# -*- coding: utf-8 -*-
# 用 NLTK 替换 LLM，实现英文语法高亮（启发式规则）
#
# 说明：
# - 保留原 FastAPI 接口与前端行为（/analyze 与 /analyze/stream）
# - 不调用第三方大模型，仅使用 NLTK 的词性标注与简单分块/规则
# - 严格保持原始文本每个字符不变，仅在其外层插入 <span> 标签
# - 规则为启发式，仅能覆盖常见结构，复杂句法可能不完全准确
#
# 依赖：
# - pip install fastapi uvicorn nltk
# - 首次运行会尝试下载所需的 NLTK 数据（averaged_perceptron_tagger 等）
#
# 运行：
# - uvicorn app:app --reload
#
# 注意：
# - 本实现不进行彻底的短语/从句语法分析，标签可能不完善
# - 若需要更高准确度，建议使用依存句法分析器（例如 spaCy）或语法分析模型

import html
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from nltk import pos_tag
from nltk.chunk import RegexpParser
from pydantic import BaseModel, Field

from style_config import SENTENCE_HELPER_ENABLED, STYLE_BLOCK

ANALYSIS_HELPER_STATE = "on" if SENTENCE_HELPER_ENABLED else "off"

# 尝试下载 NLTK 数据（若未安装）
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    try:
        nltk.download("averaged_perceptron_tagger", quiet=True)
    except Exception:
        pass

# 新版 NLTK 有时需要 averaged_perceptron_tagger_eng
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    try:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except Exception:
        pass

# 请求/响应模型保持不变
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw English text to highlight")

class AnalyzeResponse(BaseModel):
    highlighted_html: str

# 词元结构（保持原字符与索引）
@dataclass
class Token:
    text: str
    start: int
    end: int
    kind: str  # 'word' | 'space' | 'punct'

# Span 结构：用于插入标签（基于 token 边界）
@dataclass
class Span:
    start_token: int  # 开始 token 索引（含）
    end_token: int    # 结束 token 索引（不含）
    cls: str          # class 名称
    attrs: Optional[Dict[str, str]] = None

@dataclass
class SentenceInsights:
    subject: Optional[str] = None
    predicate: Optional[str] = None
    direct_object: Optional[str] = None
    indirect_object: Optional[str] = None
    complement: Optional[str] = None
    adverbial_count: int = 0
    token_count: int = 0
    clause_notes: Optional[List[str]] = None

    def __post_init__(self):
        if self.clause_notes is None:
            self.clause_notes = []

    def is_complex(self) -> bool:
        if self.token_count >= 25:
            return True
        return len(self.clause_notes) >= 1 or self.adverbial_count >= 2

    def build_note(self) -> str:
        parts: List[str] = []
        parts.append(f"主语：{self.subject or '未识别'} / 谓语：{self.predicate or '未识别'}")
        obj_bits: List[str] = []
        if self.direct_object:
            obj_bits.append(f"宾语：{self.direct_object}")
        if self.indirect_object:
            obj_bits.append(f"间宾：{self.indirect_object}")
        if obj_bits:
            parts.append("，".join(obj_bits))
        if self.complement:
            parts.append(f"补语：{self.complement}")
        if self.adverbial_count:
            parts.append(f"状语×{self.adverbial_count}")
        if self.clause_notes:
            clause_text = "、".join(self.clause_notes[:2])
            if len(self.clause_notes) > 2:
                clause_text += " 等"
            parts.append("从句：" + clause_text)
        if self.is_complex():
            parts.append("提示：句子结构较复杂")
        note = "；".join(parts)
        return note[:180]

# 词法划分（保留精确字符与偏移）
TOKEN_REGEX = re.compile(
    r"""
    (?:\s+)                                          # 空白
    |(?:\d+(?:[\.,]\d+)*)                            # 数字（含小数/千位分隔）
    |(?:\w+(?:[-']\w+)*)                             # 扩展单词：Unicode 字母/数字及内部连接符
    |(?:.)                                           # 兜底：任何单字符（含特殊符号）
    """,
    re.VERBOSE | re.UNICODE
)

WORD_LIKE_RE = re.compile(r"\w+(?:[-']\w+)*\Z", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)*\Z", re.UNICODE)

CIRCLED_DIGITS = [
    "①", "②", "③", "④", "⑤",
    "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮",
    "⑯", "⑰", "⑱", "⑲", "⑳",
]


def format_sentence_marker(index: int) -> str:
    if 1 <= index <= len(CIRCLED_DIGITS):
        return CIRCLED_DIGITS[index - 1]
    return f"({index})"


def _classify_segment(seg: str) -> str:
    if not seg:
        return "punct"
    if seg.isspace():
        return "space"
    if NUMBER_RE.fullmatch(seg) or WORD_LIKE_RE.fullmatch(seg):
        return "word"
    return "punct"


def _append_fallback_tokens(text: str, start: int, end: int, tokens: List["Token"]) -> None:
    """逐字符兜底，确保所有特殊符号都能被保留。"""
    for idx in range(start, end):
        ch = text[idx]
        if ch.isspace():
            kind = "space"
        elif ch.isalnum() or ch == "_":
            kind = "word"
        else:
            kind = "punct"
        tokens.append(Token(ch, idx, idx + 1, kind))


def tokenize_preserve(text: str) -> List[Token]:
    tokens: List[Token] = []
    if not text:
        return tokens

    last_end = 0
    for m in TOKEN_REGEX.finditer(text):
        if m.start() > last_end:
            _append_fallback_tokens(text, last_end, m.start(), tokens)
        seg = text[m.start():m.end()]
        kind = _classify_segment(seg)
        tokens.append(Token(seg, m.start(), m.end(), kind))
        last_end = m.end()

    if last_end < len(text):
        _append_fallback_tokens(text, last_end, len(text), tokens)

    # 若文本为空或未匹配（极少见），退化为整段一个 token
    if not tokens and text:
        tokens = [Token(text, 0, len(text), "word" if text[0].isalnum() else "punct")]
    return tokens

# 将 token 序列切分为句子范围（简单启发式：以 . ! ? 终止）
def sentence_token_ranges(tokens: List[Token]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    n = len(tokens)
    if n == 0:
        return ranges
    start = 0
    i = 0
    while i < n:
        t = tokens[i]
        if t.kind == "punct" and t.text in (".", "!", "?"):
            end = i + 1
            # 吸纳紧随其后的右引号/右括号
            while end < n and tokens[end].kind == "punct" and tokens[end].text in ('"', "'", "”", "’", ")", "]"):
                end += 1
            ranges.append((start, end))
            # 跳过后续空白作为下句起始
            while end < n and tokens[end].kind == "space":
                end += 1
            start = end
            i = end
            continue
        i += 1
    if start < n:
        ranges.append((start, n))
    return ranges

# 获取句子内的词 token 索引及对应词列表
def words_in_range(tokens: List[Token], start: int, end: int) -> Tuple[List[int], List[str]]:
    idxs = [i for i in range(start, end) if tokens[i].kind == "word"]
    words = [tokens[i].text for i in idxs]
    return idxs, words

# POS 标注（对词列表）
def tag_words(words: List[str]) -> List[Tuple[str, str]]:
    try:
        return pos_tag(words)
    except Exception:
        # 兜底：若标注器不可用，则将所有词标记为 NN
        return [(w, "NN") for w in words]

# 简单短语分块（NP/VP/PP/ADJP/ADVP）
CHUNK_GRAMMAR = r"""
    NP: {<DT|PRP\$|CD>?<JJ.*>*<NN.*|PRP>+}
    PP: {<IN><NP>}
    ADJP: {<RB.*>*<JJ.*>+}
    ADVP: {<RB.*>+}
    VP: {<MD>?<VB.*>+<RB.*>*}
"""

chunk_parser = RegexpParser(CHUNK_GRAMMAR)

@dataclass
class Chunk:
    label: str
    start_w: int  # 词索引（句内）
    end_w: int    # 词索引（句内，不含）

def build_chunks(pos_tags: List[Tuple[str, str]]) -> List[Chunk]:
    tree = chunk_parser.parse(pos_tags)
    chunks: List[Chunk] = []
    w_idx = 0

    def traverse(t):
        nonlocal w_idx
        if hasattr(t, "label"):
            label = t.label()
            start_here = w_idx
            for child in t:
                if isinstance(child, tuple):
                    w_idx += 1
                else:
                    traverse(child)
            end_here = w_idx
            # 过滤只保留定义的 chunk
            if label in {"NP", "PP", "VP", "ADJP", "ADVP"} and end_here > start_here:
                chunks.append(Chunk(label=label, start_w=start_here, end_w=end_here))
        else:
            # 叶子词
            for _ in t:
                w_idx += 1

    traverse(tree)
    return chunks

# 工具：找到某词索引范围映射到 token 索引范围（仅覆盖词，不包含额外空白）
def word_range_to_token_range(word_idxs: List[int], start_w: int, end_w: int) -> Tuple[int, int]:
    # 词到 token 的映射：word_token_idxs[w] -> token_idx
    if start_w >= end_w or not word_idxs:
        return -1, -1
    start_token = word_idxs[start_w]
    end_token = word_idxs[end_w - 1] + 1
    return start_token, end_token

# 添加 span（安全属性）
def add_span(spans: List[Span], start_token: int, end_token: int, cls: str, attrs: Optional[Dict[str, str]] = None):
    if start_token < 0 or end_token < 0 or end_token <= start_token:
        return
    safe_attrs = None
    if attrs:
        safe_attrs = {k: html.escape(v, quote=True) for k, v in attrs.items() if v is not None}
    spans.append(Span(start_token=start_token, end_token=end_token, cls=cls, attrs=safe_attrs))

def annotate_sentence(tokens: List[Token], start: int, end: int, *, depth: int = 0, max_depth: int = 2) -> Tuple[List[Span], SentenceInsights]:
    spans: List[Span] = []
    word_token_idxs, words = words_in_range(tokens, start, end)
    if not words:
        return spans, SentenceInsights(token_count=0)

    pos_tags = tag_words(words)
    chunks = build_chunks(pos_tags)
    info = SentenceInsights(token_count=len(words))

    def word_text(w_idx: Optional[int]) -> Optional[str]:
        if w_idx is None or w_idx < 0 or w_idx >= len(word_token_idxs):
            return None
        return tokens[word_token_idxs[w_idx]].text

    def range_text(sw: int, ew: int) -> Optional[str]:
        s_tok, e_tok = word_range_to_token_range(word_token_idxs, sw, ew)
        if s_tok < 0 or e_tok <= s_tok:
            return None
        return "".join(tokens[i].text for i in range(s_tok, e_tok)).strip()

    # 词性集合
    finite_tags = {"VB", "VBD", "VBP", "VBZ"}
    linking_verbs = {
        "am","is","are","was","were","be","been","being",
        "seem","seems","seemed","become","becomes","became",
        "appear","appears","appeared","feel","feels","felt",
        "smell","smells","smelled","look","looks","looked",
        "sound","sounds","sounded","taste","tastes","tasted",
        "remain","remains","remained","stay","stays","stayed",
        "turn","turns","turned","grow","grows","grew"
    }
    subordinators_to_function = {
        # TIME
        "when":"TIME","while":"TIME","after":"TIME","before":"TIME","until":"TIME","as":"TIME",
        "once":"TIME","since":"TIME",
        # REASON
        "because":"REASON","since":"REASON","as":"REASON","now that":"REASON",
        # CONDITION
        "if":"CONDITION","unless":"CONDITION","provided":"CONDITION","provided that":"CONDITION",
        # CONCESSION
        "although":"CONCESSION","though":"CONCESSION","even though":"CONCESSION","whereas":"CONCESSION",
        # PURPOSE
        "so that":"PURPOSE","in order that":"PURPOSE","lest":"PURPOSE",
        # RESULT
        "so that":"RESULT","so":"RESULT"
    }
    adverbial_function_labels = {
        "TIME":"时间","REASON":"原因","CONDITION":"条件","CONCESSION":"让步","PURPOSE":"目的","RESULT":"结果"
    }
    relative_prons = {"which","that","who","whom","whose","where","when"}

    # 辅助：查找第一个有限动词（predicate）
    pred_w_idx = None
    for i, (_, tag) in enumerate(pos_tags):
        if tag in finite_tags:
            pred_w_idx = i
            break
    if pred_w_idx is None:
        # 若无有限动词，尝试把第一个情态动词或动词当作谓语
        for i, (_, tag) in enumerate(pos_tags):
            if tag in {"MD", "VBG", "VBN"} or tag.startswith("VB"):
                pred_w_idx = i
                break

    # 主语：选择谓语之前的第一个 NP 的中心词（最后一个名词/代词）
    subject_w_idx = None
    if pred_w_idx is not None:
        candidate_nps = [c for c in chunks if c.label == "NP" and c.end_w <= pred_w_idx]
        if candidate_nps:
            np = candidate_nps[0]
            head = None
            for j in range(np.end_w - 1, np.start_w - 1, -1):
                if pos_tags[j][1].startswith("NN") or pos_tags[j][1] == "PRP":
                    head = j
                    break
            subject_w_idx = head or (np.end_w - 1)

    # 直接/间接宾语与补语（简单启发式）
    do_w_idx = None
    io_w_idx = None
    complement_range = None

    if pred_w_idx is not None:
        nps_after = [c for c in chunks if c.label == "NP" and c.start_w > pred_w_idx]
        pps_after = [c for c in chunks if c.label == "PP" and c.start_w > pred_w_idx]

        # give him a book -> 两个 NP 连续，前者为间接宾语，后者为直接宾语
        if len(nps_after) >= 2 and nps_after[0].end_w <= nps_after[1].start_w:
            # IO head
            head_io = None
            for j in range(nps_after[0].end_w - 1, nps_after[0].start_w - 1, -1):
                if pos_tags[j][1].startswith("NN") or pos_tags[j][1] == "PRP":
                    head_io = j
                    break
            io_w_idx = head_io or (nps_after[0].end_w - 1)

            # DO head
            head_do = None
            for j in range(nps_after[1].end_w - 1, nps_after[1].start_w - 1, -1):
                if pos_tags[j][1].startswith("NN") or pos_tags[j][1] == "PRP":
                    head_do = j
                    break
            do_w_idx = head_do or (nps_after[1].end_w - 1)
        else:
            # PP to/for + NP 作为 IO
            io_found = False
            for pp in pps_after:
                prep_word = words[pp.start_w].lower() if pp.start_w < len(words) else ""
                if prep_word in {"to", "for"}:
                    # NP 应跟随在 PP 内
                    # 简化：将 PP 里的 NP 的中心词作为 IO
                    np_in_pp = None
                    for c in chunks:
                        if c.label == "NP" and c.start_w >= pp.start_w and c.end_w <= pp.end_w:
                            np_in_pp = c
                            break
                    if np_in_pp:
                        head_io = None
                        for j in range(np_in_pp.end_w - 1, np_in_pp.start_w - 1, -1):
                            if pos_tags[j][1].startswith("NN") or pos_tags[j][1] == "PRP":
                                head_io = j
                                break
                        io_w_idx = head_io or (np_in_pp.end_w - 1)
                        io_found = True
                        break
            # DO：第一个 NP（在谓语之后）且不属于 IO
            if nps_after:
                candidate_do = nps_after[0]
                if io_found and io_w_idx is not None:
                    # 如果 IO 存在且 DO 与 IO 是同一 NP，尝试下一个
                    if candidate_do.start_w <= io_w_idx < candidate_do.end_w and len(nps_after) > 1:
                        candidate_do = nps_after[1]
                head_do = None
                for j in range(candidate_do.end_w - 1, candidate_do.start_w - 1, -1):
                    if pos_tags[j][1].startswith("NN") or pos_tags[j][1] == "PRP":
                        head_do = j
                        break
                do_w_idx = head_do or (candidate_do.end_w - 1)

        # 补语：系动词后面的 NP 或 ADJP
        pred_word_lower = words[pred_w_idx].lower()
        if pred_word_lower in linking_verbs:
            # 找到谓语后最近的 NP/ADJP
            cands = [c for c in chunks if c.start_w > pred_w_idx and c.label in {"NP", "ADJP"}]
            if cands:
                comp = cands[0]
                complement_range = (comp.start_w, comp.end_w)

    # 修饰语（PP 与 ADVP）
    adverbial_ranges: List[Tuple[int, int]] = []
    for c in chunks:
        if c.label in {"PP", "ADVP"}:
            adverbial_ranges.append((c.start_w, c.end_w))
    info.adverbial_count = len(adverbial_ranges)

    # 同位语：NP , NP
    apposition_ranges: List[Tuple[int, int]] = []
    # 查找逗号分隔的 NP 对
    comma_token_idxs = [i for i in range(start, end) if tokens[i].kind == "punct" and tokens[i].text == ","]
    for comma_tok in comma_token_idxs:
        # 找逗号前后的最近 NP（词索引）
        left_nps = [c for c in chunks if c.label == "NP" and word_token_idxs[c.end_w - 1] < comma_tok]
        right_nps = [c for c in chunks if c.label == "NP" and word_token_idxs[c.start_w] > comma_tok]
        if left_nps and right_nps:
            apposition_ranges.append((right_nps[0].start_w, right_nps[0].end_w))

    # 插入主语、谓语、宾语、补语、修饰语、同位语
    def add_head_span(role_cls: str, w_idx: Optional[int]):
        if w_idx is None:
            return
        s_tok = word_token_idxs[w_idx]
        e_tok = s_tok + 1
        add_span(spans, s_tok, e_tok, role_cls)

    add_head_span("role-subject", subject_w_idx)
    info.subject = word_text(subject_w_idx)

    add_head_span("role-predicate", pred_w_idx)
    info.predicate = word_text(pred_w_idx)

    add_head_span("role-object-do", do_w_idx)
    info.direct_object = word_text(do_w_idx)

    add_head_span("role-object-io", io_w_idx)
    info.indirect_object = word_text(io_w_idx)

    if complement_range:
        s_tok, e_tok = word_range_to_token_range(word_token_idxs, complement_range[0], complement_range[1])
        add_span(spans, s_tok, e_tok, "role-complement")
        info.complement = range_text(complement_range[0], complement_range[1])

    for (sw, ew) in adverbial_ranges:
        s_tok, e_tok = word_range_to_token_range(word_token_idxs, sw, ew)
        add_span(spans, s_tok, e_tok, "role-adverbial")

    for (sw, ew) in apposition_ranges:
        s_tok, e_tok = word_range_to_token_range(word_token_idxs, sw, ew)
        add_span(spans, s_tok, e_tok, "role-apposition")

    # 括注（parenthetical）：( ... ) 的内容
    # 将包含括号与内部全部内容包裹
    stack_paren = []
    for i in range(start, end):
        if tokens[i].text == "(":
            stack_paren.append(i)
        elif tokens[i].text == ")" and stack_paren:
            s_tok = stack_paren.pop()
            add_span(spans, s_tok, i + 1, "role-parenthetical")

    # 绝对结构（absolute）：简化为 NP, VBG... 或 NP, NP... 逗号包裹
    for idx, i in enumerate(comma_token_idxs):
        if idx + 1 < len(comma_token_idxs):
            j = comma_token_idxs[idx + 1]
            _, wlist = words_in_range(tokens, i + 1, j)
            pos_mid = tag_words(wlist) if wlist else []
            has_vbg = any(tag == "VBG" for _, tag in pos_mid)
            if has_vbg:
                add_span(spans, i, j + 1, "role-absolute")

    if depth < max_depth:
        # 从句标注（启发式）
        # 1) 关系从句
        def annotate_clause_within(range_sw: int, range_ew: int,
                                   clause_cls: str, attrs: Optional[Dict[str, str]] = None):
            s_tok, e_tok = word_range_to_token_range(word_token_idxs, range_sw, range_ew)
            add_span(spans, s_tok, e_tok, clause_cls, attrs)
            sub_spans, _ = annotate_sentence(tokens, s_tok, e_tok, depth=depth + 1, max_depth=max_depth)
            for sp in sub_spans:
                if sp.cls.startswith("role-"):
                    spans.append(sp)

        for i, (w, _) in enumerate(pos_tags):
            lw = w.lower()
            if lw in relative_prons:
                antecedent = ""
                prev_nps = [c for c in chunks if c.label == "NP" and c.end_w <= i]
                if prev_nps:
                    head = None
                    for j in range(prev_nps[-1].end_w - 1, prev_nps[-1].start_w - 1, -1):
                        if pos_tags[j][1].startswith("NN") or pos_tags[j][1] == "PRP":
                            head = j
                            break
                    if head is None:
                        head = prev_nps[-1].end_w - 1
                    antecedent = words[head]
                end_w = len(words)
                for k in range(word_token_idxs[i] + 1, end):
                    if tokens[k].kind == "punct" and tokens[k].text == ",":
                        end_w = next((x for x, tk in enumerate(word_token_idxs) if tk >= k), len(words))
                        break
                attrs: Dict[str, str] = {}
                if antecedent:
                    attrs["data-modifies"] = antecedent
                annotate_clause_within(i, end_w, "clause-relative", attrs)
                label = "关系从句"
                if antecedent:
                    label += f"（修饰 {antecedent}）"
                info.clause_notes.append(label)

        # 2) 副词从句
        def find_longest_match(start_idx: int) -> Tuple[Optional[str], int]:
            candidates = [
                ("in order that", 3), ("even though", 2), ("so that", 2), ("now that", 2),
                ("as though", 2), ("as if", 2), ("provided that", 2),
            ]
            for phrase, length in candidates:
                parts = phrase.split()
                if start_idx + length <= len(words):
                    if [words[start_idx+i].lower() for i in range(length)] == parts:
                        return phrase, length
            single = words[start_idx].lower()
            if single in subordinators_to_function:
                return single, 1
            return None, 0

        i = 0
        while i < len(words):
            key, length = find_longest_match(i)
            if key:
                func = subordinators_to_function.get(key, "TIME" if key in {"when","while","after","before","until","as","once","since"} else "")
                end_w = len(words)
                start_tok = word_token_idxs[i]
                for k in range(start_tok + 1, end):
                    if tokens[k].kind == "punct" and tokens[k].text == ",":
                        end_w = next((x for x, tk in enumerate(word_token_idxs) if tk >= k), len(words))
                        break
                attrs: Dict[str, str] = {}
                if func:
                    attrs["data-function"] = func
                annotate_clause_within(i, end_w, "clause-adverbial", attrs)
                label = "状语从句"
                if func:
                    label += f"（{adverbial_function_labels.get(func, func)}）"
                info.clause_notes.append(label)
                i += length
            else:
                i += 1

        # 3) 名词从句
        for i, (w, _) in enumerate(pos_tags):
            lw = w.lower()
            if lw in {"that", "whether", "if"}:
                clause_role = ""
                if i == 0:
                    clause_role = "subject"
                elif pred_w_idx is not None and i > pred_w_idx:
                    clause_role = "object"
                else:
                    clause_role = "complement"
                end_w = len(words)
                start_tok = word_token_idxs[i]
                for k in range(start_tok + 1, end):
                    if tokens[k].kind == "punct" and tokens[k].text == ",":
                        end_w = next((x for x, tk in enumerate(word_token_idxs) if tk >= k), len(words))
                        break
                attrs = {"data-clause-role": clause_role}
                annotate_clause_within(i, end_w, "clause-noun", attrs)
                role_cn = {"subject": "主语", "object": "宾语", "complement": "补语"}.get(clause_role, clause_role)
                info.clause_notes.append(f"名词从句（{role_cn}）")

        # 4) 非限定从句
        for i, (w, tag) in enumerate(pos_tags):
            lw = w.lower()
            is_nonfinite_start = False
            if lw == "to" and i + 1 < len(pos_tags) and pos_tags[i + 1][1] == "VB":
                is_nonfinite_start = True
            elif tag in {"VBG", "VBN"}:
                is_nonfinite_start = True
            if is_nonfinite_start:
                end_w = len(words)
                start_tok = word_token_idxs[i]
                for k in range(start_tok + 1, end):
                    if tokens[k].kind == "punct" and tokens[k].text == ",":
                        end_w = next((x for x, tk in enumerate(word_token_idxs) if tk >= k), len(words))
                        break
                annotate_clause_within(i, end_w, "clause-nonfinite", None)
                info.clause_notes.append("非限定从句 / 非谓语结构")

    return spans, info

# 渲染：将 spans 插入到原文本 token 流中（不改变任何字符，仅包裹）
def render_with_spans(tokens: List[Token], spans: List[Span], range_start: int = 0, range_end: Optional[int] = None) -> str:
    if range_end is None:
        range_end = len(tokens)
    # 仅保留与范围相交的 spans，并将边界裁剪到范围内
    effective_spans: List[Span] = []
    for sp in spans:
        if sp.end_token <= range_start or sp.start_token >= range_end:
            continue
        s = max(sp.start_token, range_start)
        e = min(sp.end_token, range_end)
        if e > s:
            effective_spans.append(Span(start_token=s, end_token=e, cls=sp.cls, attrs=sp.attrs))

    # 构建起止映射
    opens: Dict[int, List[Span]] = {}
    closes: Dict[int, List[Span]] = {}
    for sp in effective_spans:
        opens.setdefault(sp.start_token, []).append(sp)
        closes.setdefault(sp.end_token, []).append(sp)

    # 运行时维护栈，确保嵌套合法
    out_parts: List[str] = []
    active_stack: List[Span] = []

    for i in range(range_start, range_end):
        # 先关闭在此边界结束的 spans（健壮处理，避免无限循环）
        if i in closes:
            closing_list = list(closes[i])  # 拷贝以便安全修改

            # 优先关闭与栈顶匹配的
            while active_stack and closing_list:
                top = active_stack[-1]
                if top in closing_list:
                    out_parts.append("</span>")
                    active_stack.pop()
                    closing_list.remove(top)
                else:
                    # 若不匹配（交叉嵌套），先关闭栈顶以恢复正确嵌套
                    out_parts.append("</span>")
                    active_stack.pop()

            # 若仍有未关闭（不在栈中的），补充输出关闭标签（容错）
            for _ in closing_list:
                out_parts.append("</span>")

        # 再打开新的 spans
        if i in opens:
            for sp in opens[i]:
                attrs_str = ""
                if sp.attrs:
                    attrs_str = "".join([f' {k}="{v}"' for k, v in sp.attrs.items()])
                out_parts.append(f'<span class="{sp.cls}"{attrs_str}>')
                active_stack.append(sp)

        # 输出当前 token 文本（HTML 转义）
        out_parts.append(html.escape(tokens[i].text, quote=True))

    # 关闭剩余未闭合（理论上不应该存在，但容错处理）
    while active_stack:
        out_parts.append("</span>")
        active_stack.pop()

    return "".join(out_parts)

# 核心高亮函数（非流式）
def highlight_text_with_nltk(text: str) -> str:
    tokens = tokenize_preserve(text)
    sent_ranges = sentence_token_ranges(tokens)
    # 对每个句子做标注，聚合 spans
    all_spans: List[Span] = []
    sentence_counter = 0
    for (s, e) in sent_ranges:
        spans, info = annotate_sentence(tokens, s, e)
        if info.token_count > 0:
            sentence_counter += 1
            attrs = {"data-sid": format_sentence_marker(sentence_counter)}
            note = info.build_note()
            if note:
                attrs["data-note"] = note
            if info.is_complex():
                attrs["data-complex"] = "1"
            add_span(all_spans, s, e, "sentence-scope", attrs)
        all_spans.extend(spans)
    # 渲染整段
    return render_with_spans(tokens, all_spans)

# FastAPI 应用
app = FastAPI(title="Grammar Highlight API (NLTK)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    text = req.text
    if text is None or not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        sanitized_fragment = highlight_text_with_nltk(text)
        return AnalyzeResponse(highlighted_html=f"{STYLE_BLOCK}<div class='analysis' data-helper='{ANALYSIS_HELPER_STATE}'>{sanitized_fragment}</div>")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

# 可选：提供一个简单的流式接口，保持与原接口一致性（一次性输出）
@app.post("/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    text = req.text
    if text is None or not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    def gen():
        try:
            fragment = highlight_text_with_nltk(text)
            html_block = f"{STYLE_BLOCK}<div class='analysis' data-helper='{ANALYSIS_HELPER_STATE}'>{fragment}</div>"
            yield html_block
        except Exception as exc:
            yield f"Error: {html.escape(str(exc))}"

    return StreamingResponse(gen(), media_type="text/html")

@app.get("/health")
async def health():
    return {"status": "ok"}

UI_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Grammar Highlighter</title>
<style>
body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; line-height: 1.6; }
textarea { width: 100%; min-height: 140px; font-size: 1rem; padding: 0.75rem; border: 1px solid #d0d7de; border-radius: 0.5rem; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1rem; margin-top: 0.75rem; }
button { padding: 0.6rem 1.4rem; font-size: 1rem; cursor: pointer; border: none; border-radius: 999px; background: #1f7a8c; color: #fff; }
button + button { background: #6b7280; }
button:disabled { opacity: 0.6; cursor: wait; }
.helper-toggle { display: flex; align-items: center; font-size: 0.92rem; color: #475569; gap: 0.35rem; }
.helper-toggle input { width: 1rem; height: 1rem; }
#result { margin-top: 1.5rem; border-top: 1px solid #e5e7eb; padding-top: 1rem; min-height: 2rem; }
#status { color: #3b82f6; }
.err { color: #b00020; }
.muted { color: #6b7280; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>Grammar Highlighter </h1>
<textarea id="text" placeholder="Type the English text you want to analyze... For example: While the committee was reviewing the proposal, several members raised concerns that the timeline might be unrealistic."></textarea>
<div class="controls">
  <button type="button" id="submit">Analyze</button>
  <button type="button" id="clear">清空输入</button>
  <label class="helper-toggle"><input type="checkbox" id="helperToggle" />显示句子辅助说明</label>
  <span id="status"></span>
</div>
<div id="result"></div>

<script>
const btn = document.getElementById('submit');
const btnClear = document.getElementById('clear');
const textarea = document.getElementById('text');
const statusEl = document.getElementById('status');
const result = document.getElementById('result');
const helperToggle = document.getElementById('helperToggle');
const helperAvailable = __HELPER_AVAILABLE__;

function resetUI() {
  result.innerHTML = '';
  statusEl.textContent = '';
  statusEl.classList.remove('err');
}

function applyHelperState() {
  if (!helperToggle) return;
  const analysisEl = result.querySelector('.analysis');
  if (!analysisEl) return;
  if (!helperAvailable) {
    analysisEl.dataset.helper = 'off';
    return;
  }
  analysisEl.dataset.helper = helperToggle.checked ? 'on' : 'off';
}

btn.addEventListener('click', async () => {
  resetUI();
  const value = textarea.value.trim();
  if (!value) {
    statusEl.textContent = '请输入要分析的英文文本。';
    statusEl.classList.add('err');
    return;
  }

  btn.disabled = true;
  statusEl.textContent = 'Analyzing ...';

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: value })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || 'Request failed');
    }

    const data = await response.json();
    result.innerHTML = data.highlighted_html || '';
    statusEl.textContent = '';
    applyHelperState();
  } catch (err) {
    statusEl.textContent = '错误：' + (err.message || 'Unknown error');
    statusEl.classList.add('err');
  } finally {
    btn.disabled = false;
  }
});

if (helperToggle) {
  if (!helperAvailable) {
    helperToggle.checked = false;
    helperToggle.disabled = true;
    const helperLabel = helperToggle.closest('.helper-toggle');
    if (helperLabel) helperLabel.style.display = 'none';
  } else {
    helperToggle.addEventListener('change', applyHelperState);
  }
}

btnClear.addEventListener('click', () => {
  textarea.value = '';
  resetUI();
  textarea.focus();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def ui():
    helper_flag = "true" if SENTENCE_HELPER_ENABLED else "false"
    return UI_TEMPLATE.replace("__HELPER_AVAILABLE__", helper_flag)

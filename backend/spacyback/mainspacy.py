# -*- coding: utf-8 -*-
"""Grammar highlighter powered by spaCy + benepar constituency parsing."""

import html
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import benepar
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from spacy.cli import download as spacy_download
from spacy.language import Language
from spacy.tokens import Span as SpacySpan, Token as SpacyToken
from style_config import SENTENCE_HELPER_ENABLED, STYLE_BLOCK

BENE_PAR_WARNING: Optional[str] = None
HAS_BENEPAR: bool = False  # new: track whether benepar was successfully attached


def _ensure_benepar_warning(message: str) -> None:
    """Record a warning once when benepar annotations are unavailable."""
    global BENE_PAR_WARNING
    if not BENE_PAR_WARNING:
        BENE_PAR_WARNING = message


def _load_spacy_pipeline(
    model_name: str = "en_core_web_sm", benepar_model: str = "benepar_en3"
) -> Language:
    global BENE_PAR_WARNING, HAS_BENEPAR
    BENE_PAR_WARNING = None
    HAS_BENEPAR = False
    try:
        nlp = spacy.load(model_name)
    except OSError:
        try:
            spacy_download(model_name)
            nlp = spacy.load(model_name)
        except Exception as exc:  # pragma: no cover - install helper
            raise RuntimeError(
                f"spaCy model '{model_name}' is required. Install via `python -m spacy download {model_name}`."
            ) from exc

    # Ensure we have sentence segmentation available
    pipe_names = set(nlp.pipe_names)
    if not ({"parser", "senter", "sentencizer"} & pipe_names):
        try:
            nlp.add_pipe("sentencizer")
        except Exception:
            pass  # if already present or unavailable, ignore

    # Try to add benepar
    if "benepar" not in nlp.pipe_names:
        try:
            nlp.add_pipe("benepar", config={"model": benepar_model}, last=True)
            HAS_BENEPAR = True
        except ValueError:
            try:
                benepar.download(benepar_model)
                nlp.add_pipe("benepar", config={"model": benepar_model}, last=True)
                HAS_BENEPAR = True
            except Exception as exc:  # pragma: no cover - install helper
                HAS_BENEPAR = False
                BENE_PAR_WARNING = (
                    "Benepar model '{model}' unavailable ({err}). Falling back to dependency-based spans."
                ).format(model=benepar_model, err=exc)
        except Exception as exc:
            HAS_BENEPAR = False
            BENE_PAR_WARNING = (
                "Failed to attach benepar parser to spaCy pipeline. Falling back to dependency-based spans ({err})."
            ).format(err=exc)
    else:
        HAS_BENEPAR = True

    return nlp


try:
    NLP: Optional[Language] = _load_spacy_pipeline()
    NLP_LOAD_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - import-time diagnostics
    NLP = None
    NLP_LOAD_ERROR = exc


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw English text to highlight")


class AnalyzeResponse(BaseModel):
    highlighted_html: str


@dataclass
class Token:
    text: str
    start: int
    end: int
    kind: str  # 'word' | 'space' | 'punct'


@dataclass
class Span:
    start_token: int
    end_token: int
    cls: str
    attrs: Optional[Dict[str, str]] = None


@dataclass
class SentenceSummary:
    subjects: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    complements: List[str] = field(default_factory=list)
    clauses: List[str] = field(default_factory=list)
    clause_functions: List[str] = field(default_factory=list)
    connectors: List[str] = field(default_factory=list)
    residual_roles: List[str] = field(default_factory=list)
    sentence_length: int = 0


TOKEN_REGEX = re.compile(
    r"""
    (?:\s+)
    |(?:\d+(?:[\.,]\d+)*)
    |(?:\w+(?:[-']\w+)*)
    |(?:.)
    """,
    re.VERBOSE | re.UNICODE,
)

WORD_LIKE_RE = re.compile(r"\w+(?:[-']\w+)*\Z", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)*\Z", re.UNICODE)
PARAGRAPH_BREAK_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")


SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
DIRECT_OBJECT_DEPS = {"dobj", "obj"}
INDIRECT_OBJECT_DEPS = {"iobj", "dative"}
COMPLEMENT_DEPS = {"attr", "oprd", "acomp", "ccomp", "xcomp"}
ADVERBIAL_DEPS = {"advmod", "npadvmod", "advcl", "obl", "prep", "pcomp"}
RELATIVE_PRONOUNS = {"which", "that", "who", "whom", "whose", "where", "when"}
SUBORDINATORS_TO_FUNCTION = {
    "when": "TIME",
    "while": "TIME",
    "after": "TIME",
    "before": "TIME",
    "until": "TIME",
    "as": "TIME",
    "once": "TIME",
    "since": "TIME",
    "because": "REASON",
    "now that": "REASON",
    "if": "CONDITION",
    "unless": "CONDITION",
    "provided": "CONDITION",
    "provided that": "CONDITION",
    "although": "CONCESSION",
    "though": "CONCESSION",
    "even though": "CONCESSION",
    "whereas": "CONCESSION",
    "so that": "RESULT",
    "so": "RESULT",
    "lest": "PURPOSE",
    "in order that": "PURPOSE",
}
FINITE_VERB_TAGS = {"VBD", "VBP", "VBZ"}
NONFINITE_VERB_TAGS = {"VBG", "VBN"}
FIXED_MULTIWORD_PHRASES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (
        re.compile(pattern, re.IGNORECASE),
        label,
    )
    for pattern, label in [
        (r"\bas well as\b", "as well as"),
        (r"\brather than\b", "rather than"),
        (r"\bin addition to\b", "in addition to"),
        (r"\bin spite of\b", "in spite of"),
        (r"\baccording to\b", "according to"),
        (r"\bas soon as\b", "as soon as"),
    ]
)
CLAUSE_FUNCTION_LABELS = {
    "TIME": "时间",
    "REASON": "原因",
    "CONDITION": "条件",
    "CONCESSION": "让步",
    "RESULT": "结果",
    "PURPOSE": "目的",
}
RESIDUAL_DEP_LABELS = {
    "det": "限定词",
    "prep": "介词",
    "case": "介词标记",
    "cc": "并列连词",
    "mark": "从属连词",
    "poss": "所有格标记",
    "nummod": "数量修饰语",
    "aux": "助动词",
    "prt": "小品词",
}
RESIDUAL_POS_LABELS = {
    "ADJ": "形容词修饰语",
    "ADV": "副词",
    "NUM": "数词",
    "PRON": "代词",
}


def _classify_segment(seg: str) -> str:
    if not seg:
        return "punct"
    if seg.isspace():
        return "space"
    if NUMBER_RE.fullmatch(seg) or WORD_LIKE_RE.fullmatch(seg):
        return "word"
    return "punct"


def _append_fallback_tokens(text: str, start: int, end: int, tokens: List[Token]) -> None:
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
    for match in TOKEN_REGEX.finditer(text):
        if match.start() > last_end:
            _append_fallback_tokens(text, last_end, match.start(), tokens)
        seg = text[match.start() : match.end()]
        tokens.append(Token(seg, match.start(), match.end(), _classify_segment(seg)))
        last_end = match.end()

    if last_end < len(text):
        _append_fallback_tokens(text, last_end, len(text), tokens)

    if not tokens and text:
        tokens = [Token(text, 0, len(text), "word" if text[0].isalnum() else "punct")]
    return tokens


def build_char_to_token_map(tokens: List[Token]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for idx, tok in enumerate(tokens):
        for pos in range(tok.start, tok.end):
            mapping[pos] = idx
    return mapping


def char_span_to_token_span(
    char_start: int, char_end: int, mapping: Dict[int, int]
) -> Tuple[int, int]:
    if char_end <= char_start:
        return -1, -1
    start_idx = mapping.get(char_start)
    end_idx = mapping.get(char_end - 1)
    if start_idx is None or end_idx is None:
        return -1, -1
    return start_idx, end_idx + 1


def add_char_based_span(
    spans: List[Span],
    char_start: int,
    char_end: int,
    cls: str,
    mapping: Dict[int, int],
    attrs: Optional[Dict[str, str]] = None,
) -> None:
    s_tok, e_tok = char_span_to_token_span(char_start, char_end, mapping)
    if s_tok < 0 or e_tok < 0:
        return
    safe_attrs = None
    if attrs:
        safe_attrs = {k: html.escape(v, quote=True) for k, v in attrs.items() if v}
    spans.append(Span(start_token=s_tok, end_token=e_tok, cls=cls, attrs=safe_attrs))


def add_span(spans: List[Span], start_token: int, end_token: int, cls: str, attrs: Optional[Dict[str, str]] = None):
    if start_token < 0 or end_token < 0 or end_token <= start_token:
        return
    spans.append(Span(start_token=start_token, end_token=end_token, cls=cls, attrs=attrs))


def subtree_char_span(token: SpacyToken) -> Tuple[int, int]:
    subtree = list(token.subtree)
    if not subtree:
        return token.idx, token.idx + len(token.text)
    return subtree[0].idx, subtree[-1].idx + len(subtree[-1].text)


def _subtree_text(token: SpacyToken) -> str:
    span = token.doc[token.left_edge.i : token.right_edge.i + 1]
    return span.text


def _find_antecedent_word(sentence: SpacySpan, clause_start_char: int) -> Optional[str]:
    candidate = None
    for tok in sentence:
        if tok.idx >= clause_start_char:
            break
        if tok.pos_ in {"NOUN", "PROPN", "PRON"}:
            candidate = tok.text
    return candidate


def _is_nonfinite_clause(span: SpacySpan) -> bool:
    tags = {tok.tag_ for tok in span if tok.tag_}
    if tags & FINITE_VERB_TAGS:
        return False
    if "TO" in tags or tags & NONFINITE_VERB_TAGS:
        return True
    return False


def _classify_noun_clause(span: SpacySpan) -> Optional[str]:
    deps = {tok.dep_ for tok in span}
    if deps & {"csubj", "csubjpass"}:
        return "subject"
    if deps & {"ccomp", "xcomp"}:
        return "complement"
    if deps & {"dobj", "obj"}:
        return "object"
    return None


def _split_paragraph_ranges(text: str) -> List[Tuple[int, int]]:
    """Return inclusive paragraph ranges, keeping separators intact."""
    if not text:
        return [(0, 0)]
    ranges: List[Tuple[int, int]] = []
    start = 0
    for match in PARAGRAPH_BREAK_RE.finditer(text):
        ranges.append((start, match.start()))
        start = match.end()
    ranges.append((start, len(text)))
    # Ensure at least one range and sorted order
    if not ranges:
        ranges = [(0, len(text))]
    return ranges


def _circled_number(value: int) -> str:
    """Return the circled number style for sentence numbering."""
    if value <= 0:
        return ""
    if value <= 20:
        return chr(ord("\u2460") + value - 1)
    if 21 <= value <= 35:
        return chr(ord("\u3251") + value - 21)
    if 36 <= value <= 50:
        return chr(ord("\u32B1") + value - 36)
    return f"({value})"


def annotate_constituents(
    sentence: SpacySpan,
    spans: List[Span],
    mapping: Dict[int, int],
    sentence_start_char: int,
    sentence_end_char: int,
    summary: Optional[SentenceSummary] = None,
) -> None:
    # If benepar is not attached or a previous warning indicates fallback, skip.
    if not HAS_BENEPAR or BENE_PAR_WARNING:
        _ensure_benepar_warning(
            "Benepar component missing or unavailable. Using dependency-based spans."
        )
        return

    # If the extension is not present, skip
    if not SpacySpan.has_extension("constituents"):
        _ensure_benepar_warning(
            "Benepar component missing from spaCy pipeline. Falling back to dependency spans."
        )
        return
    try:
        constituents = sentence._.constituents
    except Exception as exc:
        # Catch any error while accessing benepar results and fallback safely
        _ensure_benepar_warning(
            f"Benepar constituency parse unavailable: {exc}. Falling back to dependency spans."
        )
        return

    seen_ranges = set()
    for const in constituents:
        label = getattr(const, "label_", None)
        if not label:
            continue
        start_char, end_char = const.start_char, const.end_char
        if start_char == sentence_start_char and end_char == sentence_end_char:
            continue  # skip the entire sentence span itself

        key = (start_char, end_char, label)
        is_relative = False

        if label in {"PP", "ADVP"}:
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            add_char_based_span(spans, start_char, end_char, "role-adverbial", mapping)
            continue

        if label == "SBAR" and const:
            first_token = const[0]
            lowered = first_token.text.lower()
            if lowered in RELATIVE_PRONOUNS:
                antecedent = _find_antecedent_word(sentence, start_char)
                attrs = {"data-modifies": antecedent} if antecedent else None
                add_char_based_span(spans, start_char, end_char, "clause-relative", mapping, attrs)
                if summary:
                    summary.clauses.append("定语从句")
                is_relative = True
            else:
                function = SUBORDINATORS_TO_FUNCTION.get(lowered)
                attrs = {"data-function": function}
                add_char_based_span(spans, start_char, end_char, "clause-adverbial", mapping, attrs)
                if summary:
                    summary.clauses.append("状语从句")
                    if function:
                        summary.clause_functions.append(function)
            continue

        if label in {"S", "VP"}:
            if _is_nonfinite_clause(const):
                add_char_based_span(spans, start_char, end_char, "clause-nonfinite", mapping)
                if summary:
                    summary.clauses.append("非限定结构")
                continue
            if label == "S" and not is_relative:
                role = _classify_noun_clause(const)
                if role:
                    attrs = {"data-clause-role": role}
                    add_char_based_span(spans, start_char, end_char, "clause-noun", mapping, attrs)
                    if summary:
                        summary.clauses.append(f"名词性从句({role})")


def _predicate_span_bounds(head: SpacyToken) -> Tuple[int, int]:
    """Return a character range covering predicate head + functional dependents."""
    tokens = [head]
    for child in head.children:
        if child.dep_ in {"aux", "auxpass", "prt", "cop", "neg"}:
            tokens.append(child)
    start_char = min(tok.idx for tok in tokens)
    end_char = max(tok.idx + len(tok.text) for tok in tokens)
    return start_char, end_char


def _predicate_heads(sentence: SpacySpan) -> List[SpacyToken]:
    """Collect predicate heads including coordinated verbs."""
    candidates: List[SpacyToken] = []
    for tok in sentence:
        if tok.pos_ not in {"VERB", "AUX"} and tok.tag_ not in FINITE_VERB_TAGS:
            continue
        if tok.dep_ == "ROOT":
            candidates.append(tok)
            continue
        if tok.dep_ == "conj" and tok.head.pos_ in {"VERB", "AUX"}:
            candidates.append(tok)
            continue
        if tok.dep_ in {"ccomp", "xcomp", "advcl", "acl", "relcl", "parataxis"}:
            candidates.append(tok)
    seen = set()
    ordered: List[SpacyToken] = []
    for tok in sorted(candidates, key=lambda t: t.i):
        if tok.i in seen:
            continue
        seen.add(tok.i)
        ordered.append(tok)
    return ordered


def _add_fixed_phrases(
    sentence: SpacySpan, mapping: Dict[int, int], spans: List[Span], summary: SentenceSummary
) -> None:
    base = sentence.start_char
    text = sentence.text
    for pattern, label in FIXED_MULTIWORD_PHRASES:
        for match in pattern.finditer(text):
            start_char = base + match.start()
            end_char = base + match.end()
            add_char_based_span(
                spans,
                start_char,
                end_char,
                "phrase-fixed",
                mapping,
                attrs={"data-phrase": label},
            )
            summary.connectors.append(label.lower())


def annotate_sentence(
    tokens: List[Token],
    sentence: SpacySpan,
    mapping: Dict[int, int],
) -> Tuple[List[Span], SentenceSummary]:
    spans: List[Span] = []
    summary = SentenceSummary(sentence_length=len(sentence))
    sent_bounds = char_span_to_token_span(sentence.start_char, sentence.end_char, mapping)
    sent_start_tok, sent_end_tok = sent_bounds

    def add_subtree(token: SpacyToken, cls: str):
        start_char, end_char = subtree_char_span(token)
        add_char_based_span(spans, start_char, end_char, cls, mapping)

    def add_token(token: SpacyToken, cls: str):
        add_char_based_span(spans, token.idx, token.idx + len(token.text), cls, mapping)

    for tok in sentence:
        if tok.dep_ in SUBJECT_DEPS:
            add_subtree(tok, "role-subject")
            summary.subjects.append(_subtree_text(tok))

    for head in _predicate_heads(sentence):
        start_char, end_char = _predicate_span_bounds(head)
        add_char_based_span(spans, start_char, end_char, "role-predicate", mapping)
        predicate_text = sentence.doc.text[start_char:end_char].strip()
        summary.predicates.append(predicate_text or head.text)

    for tok in sentence:
        if tok.dep_ in DIRECT_OBJECT_DEPS:
            add_subtree(tok, "role-object-do")
            summary.objects.append(_subtree_text(tok))
            break

    io_token = next((tok for tok in sentence if tok.dep_ in INDIRECT_OBJECT_DEPS), None)
    if io_token is None:
        for tok in sentence:
            if tok.dep_ == "pobj" and tok.head.dep_ == "prep" and tok.head.lemma_.lower() in {"to", "for"}:
                io_token = tok
                break
    if io_token:
        add_subtree(io_token, "role-object-io")
        summary.objects.append(_subtree_text(io_token))

    for tok in sentence:
        if tok.dep_ in COMPLEMENT_DEPS:
            add_subtree(tok, "role-complement")
            summary.complements.append(_subtree_text(tok))
            break

    for tok in sentence:
        lowered = tok.text.lower()
        if tok.dep_ in {"cc", "mark", "preconj"} or tok.pos_ in {"CCONJ", "SCONJ"}:
            add_token(tok, "role-connector")
            summary.connectors.append(lowered)
        if tok.dep_ == "det" or tok.pos_ == "DET":
            add_token(tok, "role-determiner")
        if tok.dep_ in {"amod", "poss", "compound", "nummod"}:
            add_token(tok, "role-modifier")

    adverbial_ranges = set()
    for tok in sentence:
        if tok.dep_ in ADVERBIAL_DEPS:
            adverbial_ranges.add(subtree_char_span(tok))
    for start_char, end_char in adverbial_ranges:
        add_char_based_span(spans, start_char, end_char, "role-adverbial", mapping)

    for tok in sentence:
        if tok.dep_ == "appos":
            add_subtree(tok, "role-apposition")

    if sent_start_tok >= 0 and sent_end_tok >= 0:
        stack = []
        for idx in range(sent_start_tok, sent_end_tok):
            token = tokens[idx]
            if token.text == "(":
                stack.append(idx)
            elif token.text == ")" and stack:
                add_span(spans, stack.pop(), idx + 1, "role-parenthetical")

        comma_token_idxs = [
            i
            for i in range(sent_start_tok, sent_end_tok)
            if tokens[i].kind == "punct" and tokens[i].text == ","
        ]
        for idx, first_comma in enumerate(comma_token_idxs):
            if idx + 1 >= len(comma_token_idxs):
                break
            second_comma = comma_token_idxs[idx + 1]
            start_char = tokens[first_comma].start
            end_char = tokens[second_comma].end
            span = sentence.doc.char_span(start_char, end_char, alignment_mode="expand")
            if span and any(tok.tag_ == "VBG" for tok in span):
                add_span(spans, first_comma, second_comma + 1, "role-absolute")

    annotate_constituents(
        sentence,
        spans,
        mapping,
        sentence.start_char,
        sentence.end_char,
        summary,
    )
    _add_fixed_phrases(sentence, mapping, spans, summary)

    return spans, summary


def _label_residual_token(token: SpacyToken) -> Optional[str]:
    dep_label = RESIDUAL_DEP_LABELS.get(token.dep_)
    if dep_label:
        return dep_label
    return RESIDUAL_POS_LABELS.get(token.pos_)


def _collect_residual_roles(
    sentence: SpacySpan,
    tokens: List[Token],
    spans: List[Span],
    sent_bounds: Tuple[int, int],
    summary: SentenceSummary,
    mapping: Dict[int, int],
) -> None:
    sent_start, sent_end = sent_bounds
    if sent_start < 0 or sent_end < 0 or sent_start >= sent_end:
        return
    coverage = [False] * (sent_end - sent_start)
    for span in spans:
        lo = max(span.start_token, sent_start)
        hi = min(span.end_token, sent_end)
        for idx in range(lo, hi):
            coverage[idx - sent_start] = True
    doc = sentence.doc
    for offset, covered in enumerate(coverage):
        if covered:
            continue
        token = tokens[sent_start + offset]
        if token.kind != "word":
            continue
        span = doc.char_span(token.start, token.end, alignment_mode="expand")
        if not span or not span.text.strip():
            continue
        label = _label_residual_token(span[0])
        if label and label not in summary.residual_roles:
            summary.residual_roles.append(label)
        if label:
            add_char_based_span(
                spans,
                token.start,
                token.end,
                "role-residual",
                mapping,
                attrs={"data-role": label},
            )


def _classify_sentence_complexity(summary: SentenceSummary) -> Tuple[str, bool]:
    clause_count = len(summary.clauses)
    connector_count = len(summary.connectors)
    word_count = summary.sentence_length
    if clause_count >= 2:
        return "多重复杂句", True
    if clause_count == 1:
        return "主从复合句", True
    if connector_count >= 2:
        return "并列复合句", True
    if word_count >= 25:
        return "长句", True
    return "简单句", False


def _translate_clause_functions(functions: List[str]) -> List[str]:
    translated = []
    for item in functions:
        label = CLAUSE_FUNCTION_LABELS.get(item, item)
        if label not in translated:
            translated.append(label)
    return translated


def build_sentence_note(summary: SentenceSummary) -> Tuple[str, bool]:
    note_parts: List[str] = []
    clause_label = "无"
    if summary.clauses:
        counts = Counter(summary.clauses)
        clause_label = "、".join(
            f"{name}×{count}" if count > 1 else name for name, count in counts.items()
        )
    functions = _translate_clause_functions(summary.clause_functions)
    connectors = list(dict.fromkeys(summary.connectors))
    residual = summary.residual_roles
    subjects_seq = list(dict.fromkeys(summary.subjects))
    predicates_seq = list(dict.fromkeys(summary.predicates))
    objects_seq = list(dict.fromkeys(summary.objects))
    complements_seq = list(dict.fromkeys(summary.complements))
    subjects = "、".join(subjects_seq) if subjects_seq else "未识别"
    predicates = "、".join(predicates_seq) if predicates_seq else "未识别"
    objects = "、".join(objects_seq) if objects_seq else "无"
    complements = "、".join(complements_seq) if complements_seq else "无"
    note_parts.append(f"主语：{subjects}")
    note_parts.append(f"谓语：{predicates}")
    note_parts.append(f"宾语：{objects}")
    if complements != "无":
        note_parts.append(f"补语：{complements}")
    note_parts.append(f"从句：{clause_label}")
    if functions:
        note_parts.append(f"从句功能：{'、'.join(functions)}")
    connector_text = "、".join(connectors) if connectors else "未检测到典型连接词"
    note_parts.append(f"连接词：{connector_text}")
    if residual:
        note_parts.append(f"未高亮：{'、'.join(residual)}")
    complexity_label, is_complex = _classify_sentence_complexity(summary)
    note_parts.insert(0, f"句型：{complexity_label}")
    note_parts.append(f"词数：{summary.sentence_length}")
    return "；".join(note_parts), is_complex


def render_with_spans(tokens: List[Token], spans: List[Span]) -> str:
    spans = sorted(spans, key=lambda s: (s.start_token, -s.end_token))
    out_parts: List[str] = []
    active_stack: List[Span] = []
    span_queue = list(spans)
    current_idx = 0

    def open_span(span: Span):
        attrs = ""
        if span.attrs:
            attrs = " " + " ".join(
                f"{k}='" + html.escape(v, quote=True) + "'" for k, v in span.attrs.items()
            )
        out_parts.append(f"<span class='{span.cls}'{attrs}>")

    def close_span():
        out_parts.append("</span>")

    while current_idx < len(tokens):
        opening = [sp for sp in span_queue if sp.start_token == current_idx]
        for sp in opening:
            open_span(sp)
            active_stack.append(sp)
            span_queue.remove(sp)

        token = tokens[current_idx]
        out_parts.append(html.escape(token.text))
        current_idx += 1

        while active_stack and active_stack[-1].end_token == current_idx:
            active_stack.pop()
            close_span()

    while active_stack:
        active_stack.pop()
        close_span()

    return "".join(out_parts)


def _run_pipeline_without_benepar(text: str) -> "spacy.tokens.Doc":
    """Run the spaCy pipeline skipping benepar, for robust fallback."""
    assert NLP is not None
    doc = NLP.make_doc(text)
    for name, proc in NLP.pipeline:
        if name == "benepar":
            continue
        doc = proc(doc)
    return doc


def highlight_text_with_spacy(text: str) -> str:
    if NLP is None:
        raise RuntimeError(f"spaCy pipeline unavailable: {NLP_LOAD_ERROR}")
    tokens = tokenize_preserve(text)
    if not tokens:
        return ""
    mapping = build_char_to_token_map(tokens)

    # Robust doc creation: if benepar causes any error, skip it and fallback.
    try:
        doc = NLP(text)
    except Exception as exc:
        _ensure_benepar_warning(
            f"Benepar failed during processing: {exc}. Falling back to dependency-based spans."
        )
        doc = _run_pipeline_without_benepar(text)

    paragraph_ranges = _split_paragraph_ranges(text)
    paragraph_counters = [0 for _ in paragraph_ranges]
    paragraph_idx = 0
    paragraph_spans: List[Span] = []
    for start, end in paragraph_ranges:
        add_char_based_span(paragraph_spans, start, end, "paragraph-scope", mapping)

    spans: List[Span] = list(paragraph_spans)

    for sent in doc.sents:
        while paragraph_idx < len(paragraph_ranges) and paragraph_ranges[paragraph_idx][1] <= sent.start_char:
            paragraph_idx += 1
        current_idx = min(paragraph_idx, len(paragraph_ranges) - 1)
        paragraph_counters[current_idx] += 1
        sentence_label = _circled_number(paragraph_counters[current_idx])

        sentence_spans, summary = annotate_sentence(tokens, sent, mapping)
        sent_bounds = char_span_to_token_span(sent.start_char, sent.end_char, mapping)
        sent_start, sent_end = sent_bounds
        if sent_start >= 0 and sent_end >= 0:
            _collect_residual_roles(sent, tokens, sentence_spans, sent_bounds, summary, mapping)
            helper_note, is_complex = build_sentence_note(summary)
            attrs = {
                "data-sid": sentence_label,
                "data-note": helper_note,
                "data-complex": "1" if is_complex else "0",
            }
            sentence_spans.append(Span(start_token=sent_start, end_token=sent_end, cls="sentence-scope", attrs=attrs))
        spans.extend(sentence_spans)
    return render_with_spans(tokens, spans)


app = FastAPI(title="Grammar Highlight API (spaCy + benepar)")
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
        sanitized_fragment = highlight_text_with_spacy(text)
        helper_state = "on" if SENTENCE_HELPER_ENABLED else "off"
        return AnalyzeResponse(
            highlighted_html=f"{STYLE_BLOCK}<div class='analysis' data-helper='{helper_state}'>{sanitized_fragment}</div>"
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/health")
async def health():
    status = "ok" if NLP is not None else "failed"
    detail = None if NLP is not None else str(NLP_LOAD_ERROR)
    payload = {"status": status}
    if detail:
        payload["detail"] = detail
    if BENE_PAR_WARNING:
        payload["warning"] = BENE_PAR_WARNING
    payload["benepar_attached"] = HAS_BENEPAR
    return payload


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"UTF-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Grammar Highlighter</title>
<style>
body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; line-height: 1.6; }
textarea { width: 100%; min-height: 140px; font-size: 1rem; padding: 0.75rem; border: 1px solid #d0d7de; border-radius: 0.5rem; }
button { margin-top: 0.75rem; padding: 0.6rem 1.4rem; font-size: 1rem; cursor: pointer; border: none; border-radius: 999px; background: #1f7a8c; color: #fff; }
button + button { margin-left: 0.5rem; background: #6b7280; }
button:disabled { opacity: 0.6; cursor: wait; }
#result { margin-top: 1.5rem; border-top: 1px solid #e5e7eb; padding-top: 1rem; min-height: 2rem; }
#status { margin-left: 0.75rem; color: #3b82f6; }
.err { color: #b00020; }
.muted { color: #6b7280; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>Grammar Highlighter (spaCy + benepar)</h1>
<textarea id=\"text\" placeholder=\"Type the English text you want to analyze...\"></textarea>
<div>
<button type=\"button\" id=\"submit\">Analyze</button>
<button type=\"button\" id=\"clear\">清空输入</button>
<span id=\"status\"></span>
</div>
<div id=\"result\"></div>

<script>
const btn = document.getElementById('submit');
const btnClear = document.getElementById('clear');
const textarea = document.getElementById('text');
const statusEl = document.getElementById('status');
const result = document.getElementById('result');

function resetUI() {
  result.innerHTML = '';
  statusEl.textContent = '';
  statusEl.classList.remove('err');
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
  } catch (err) {
    statusEl.textContent = '错误：' + (err.message || 'Unknown error');
    statusEl.classList.add('err');
  } finally {
    btn.disabled = false;
  }
});

btnClear.addEventListener('click', () => {
  textarea.value = '';
  resetUI();
  textarea.focus();
});
</script>
</body>
</html>"""
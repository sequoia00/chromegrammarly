"""可配置的样式与辅助开关。

将样式集中到本文件，方便按需注释或将 `enabled` 置为 False。
每条规则都给出了针对的语法成分及效果说明，帮助快速定位高亮条件。
"""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class StyleRule:
    selector: str
    target: str
    description: str
    css: str
    enabled: bool = True

    def to_css(self) -> str:
        return f"{self.selector}{{{self.css}}}"


def build_style_block(rules: Iterable["StyleRule"]) -> str:
    body = "".join(rule.to_css() for rule in rules if rule.enabled)
    return f"<style>{body}</style>"


# 统一的句子辅助说明开关：置为 False 时仅保留词汇/成分高亮。
SENTENCE_HELPER_ENABLED: bool = False

STYLE_RULES: List[StyleRule] = [
    StyleRule(
        selector=".analysis",
        target="语法分析容器",
        description="设置整体行距与字号，并保持换行，保证输出易读。",
        css="line-height:1.65;font-size:1rem;white-space:pre-wrap",
    ),
    StyleRule(
        selector=".analysis span",
        target="所有高亮片段",
        description="为每个高亮片段加入适度内边距和圆角，提升视觉分隔感。",
        css="padding:.04rem .08rem;border-radius:.15rem",
    ),
    StyleRule(
        selector=".sentence-scope",
        target="句子外层容器",
        description="包裹整句，方便显示序号与复杂度指示。",
        css="position:relative;display:inline;padding:0;margin-right:.12rem",
    ),
    StyleRule(
        selector=".sentence-scope::before",
        target="句子编号",
        description="在句首展示圈号，快速定位句子编号。",
        css="content:attr(data-sid)' ';color:#475569;font-weight:600;font-size:.85em;margin-right:.25rem",
    ),
    # StyleRule(
    #     selector=".sentence-scope[data-complex='1']",
    #     target="复杂句提示",
    #     description="复杂句底部加淡橙色阴影，以提示结构较复杂。",
    #     css="box-shadow:inset 0 -0.2rem 0 rgba(250,209,155,.6)",
    # ),
    StyleRule(
        selector=".analysis[data-helper='on'] .sentence-scope::after",
        target="句子辅助说明",
        description="在句后输出中文提示，解释成分与从句情况。",
        css="content:attr(data-note);display:block;font-size:.85rem;color:#64748b;margin:.2rem 0 .45rem 1.5rem;line-height:1.4",
    ),
    StyleRule(
        selector=".analysis[data-helper='off'] .sentence-scope::after",
        target="关闭辅助说明",
        description="当 helper 关闭时隐藏说明，避免额外占位。",
        css="content:'';display:none",
    ),
    StyleRule(
        selector=".role-subject",
        target="主语",
        description="淡黄色底纹突出主语位置。",
        css="background-color:#fff3bf",
    ),
    StyleRule(
        selector=".role-predicate",
        target="谓语动词",
        description="深玫红字体并加粗，强调谓语中心。",
        css="color:#000000!important;font-weight:700;background-color:rgba(255,235,239,.8)",
    ),
    StyleRule(
        selector=".role-object-do",
        target="直接宾语",
        description="浅绿底色显示直接宾语。",
        css="background-color:#d4ffaa",
    ),
    StyleRule(
        selector=".role-object-io",
        target="间接宾语",
        description="黄绿底色区分间接宾语。",
        css="background-color:#bcea81",
    ),
    StyleRule(
        selector=".role-complement",
        target="表语/主补语",
        description="实线下划线指示补语区域。",
        css="border-bottom:2px solid #e6a04c",
    ),
    StyleRule(
        selector=".role-object-complement",
        target="宾补",
        description="虚线下划线提示补充说明的宾补。",
        css="border-bottom:2px dashed #e6a04c",
    ),
    StyleRule(
        selector=".role-apposition",
        target="同位语",
        description="蓝色立线和缩进强调同位语说明。",
        css="border-left:2px solid #63a4d4;padding-left:.15rem",
    ),
    StyleRule(
        selector=".role-adverbial",
        target="状语短语",
        description="浅蓝底色突出状语信息。",
        css="background-color:#dce9ff",
    ),
    StyleRule(
        selector=".role-parenthetical",
        target="插入语",
        description="灰色虚线边框表示插入语。",
        css="border:1px dotted #888",
    ),
    # StyleRule(
    #     selector=".role-absolute",
    #     target="独立主格",
    #     description="淡紫底色展示独立主格结构。",
    #     css="background-color:#f0e8ff",
    # ),
    # StyleRule(
    #     selector=".clause-noun,.clause-relative,.clause-adverbial,.clause-nonfinite",
    #     target="从句容器（公共样式）",
    #     description="统一使用彩色立线和左内边距包裹从句内容。",
    #     css="border-left:2px solid currentColor;padding-left:.25rem;margin-left:.1rem",
    # ),
    # StyleRule(
    #     selector=".clause-noun",
    #     target="名词从句",
    #     description="绿色配色突出名词性从句。",
    #     css="color:#5c8f1d;background-color:rgba(158,201,134,.18)",
    # ),
    # StyleRule(
    #     selector=".clause-relative",
    #     target="定语从句",
    #     description="紫色底色标记定语从句，便于和主句区分。",
    #     css="color:#6b4fa1;background-color:rgba(146,132,189,.15)",
    # ),
    StyleRule(
        selector=".clause-adverbial",
        target="状语从句",
        description="灰色底色展示状语从句，配合数据属性显示功能类别。",
        css="color:#b2bd6d6d62b2;background-color:rgba(128,203,196,.18)",
    ),
    StyleRule(
        selector=".clause-nonfinite",
        target="非限定从句 / 非谓语",
        description="橙色底纹提示非限定结构。",
        css="color:#c7780a;background-color:rgba(253,203,110,.18)",
    ),
    
    
    # StyleRule(
    #     selector=".analysis[data-helper='on'] .clause-relative[data-modifies]::before,.analysis[data-helper='on'] .clause-adverbial[data-modifies]::before",
    #     target="从句修饰箭头",
    #     description="在辅助开启时显示“→”指向被修饰的成分。",
    #     css="content:'→'attr(data-modifies)' ';color:#666;font-size:.85em",
    # ),
    # StyleRule(
    #     selector=".analysis[data-helper='on'] .clause-adverbial[data-function]::after",
    #     target="状语从句功能标签",
    #     description="在尾部追加方括号说明（时间/原因等）。",
    #     css="content:' ['attr(data-function)']';color:#1b5e20;font-size:.85em",
    # ),
    # StyleRule(
    #     selector=".analysis[data-helper='on'] .clause-noun[data-clause-role]::after",
    #     target="名词从句句法角色",
    #     description="括号提示该名词从句在句中的角色（主语/宾语）。",
    #     css="content:' ('attr(data-clause-role)')';color:#3f6212;font-size:.78em",
    # ),
    StyleRule(
        selector=".phrase-fixed",
        target="固定搭配",
        description="米色底与虚线强调固定表达或习语。",
        css="background-color:#fff8f0;border-bottom:1px dashed #c28150",
    ),
    StyleRule(
        selector=".lex-rare",
        target="低频词",
        description="深蓝色字体加粗提示低频或重点词汇。",
        css="color:#000080;font-weight:600",
    ),
]

STYLE_BLOCK = build_style_block(STYLE_RULES)

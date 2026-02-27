#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文標點符號自動恢復模組

基於規則的中文標點恢復，適用於 ASR 輸出的無標點文本。
不依賴任何 ML 模型，純規則實現。

使用方式：
    from punctuation import add_punctuation
    result = add_punctuation("我跟你說啊這個東西真的很厲害")
    # 輸出: "我跟你說啊，這個東西真的很厲害。"
"""

import re
from typing import List, Tuple

__version__ = "1.0.0"

# =============================================================================
# 規則配置（可從外部修改擴展）
# =============================================================================

# 問句結尾詞（在句末時觸發問號）
QUESTION_ENDINGS: List[str] = [
    '嗎', '吗', '呢', '麼', '么'
]

# 疑問詞（出現在文中即視為疑問句）
QUESTION_WORDS: List[str] = [
    # 什麼類
    '什麼', '什么', '怎麼', '怎么', '為什麼', '为什么',
    # 哪裡類
    '哪裡', '哪里', '哪個', '哪个',
    # 誰/多少類
    '誰', '谁', '幾個', '几个', '多少',
    # 是否類
    '是否', '能否', '可否', '有沒有', '有没有', '是不是', '會不會', '会不会',
    # 如何類
    '怎樣', '怎样', '如何', '為何', '为何',
]

# 語氣詞（後面加逗號）
PARTICLES_AFTER: List[str] = [
    '嘛', '啦', '呀', '囉', '咯', '噢', '唷',
    '哎', '欸', '啊', '喔', '哦', '嗯', '呃',
]

# 疑問短語（後面加逗號）
QUESTION_PHRASES_AFTER: List[str] = [
    '不是嗎', '不是吗', '對不對', '对不对', '是不是',
    '好不好', '行不行', '可不可以', '對吧', '对吧',
    '是吧', '好嗎', '好吗',
]

# 句子連接詞（前面加逗號）
# 重要：長詞必須放在短詞前面！
SENTENCE_STARTERS: List[str] = [
    # 「可」開頭的轉折
    '可現在', '可现在', '可他', '可她', '可我', '可你',
    '可這', '可这', '可那', '可誰', '可谁', '可當', '可当',
    # 長詞優先（這些很安全）
    '換句話說', '换句话说', '例如說', '例如说', '比如說', '比如说',
    '沒想到', '没想到', '想不到', '不是嗎', '不是吗',
    '要不是', '此時此刻', '此时此刻',
    # 轉折/連接
    '然後', '然后', '接著', '接着', '之後', '之后',
    '所以', '但是', '不過', '不过', '可是',
    '而且', '並且', '并且', '或者', '還是', '还是',
    '雖然', '虽然', '即使',
    '首先', '其次', '最後', '最后', '另外', '此外',
    '總之', '总之', '反正', '難怪', '难怪',
    '其實', '其实', '原來', '原来', '後來', '后来',
    '不然', '否則', '否则',
    '於是', '于是',
    '畢竟', '毕竟', '終於', '终于',
    '當然', '当然', '幸好', '幸虧', '幸亏',
    '竟是', '竟然', '居然',
    # 轉折代詞
    '而他', '而她', '而我', '而你', '而它',
    '但他', '但她', '但我', '但你', '但它',
    # 強調詞
    '至少', '起碼', '起码',
    # 主詞+副詞（新句子開頭的強信號）
    # X就
    '我就', '你就', '他就', '她就', '它就',
    '我們就', '我们就', '你們就', '你们就', '他們就', '他们就',
    # X也
    '我也', '你也', '他也', '她也', '它也',
    '我們也', '我们也', '你們也', '你们也', '他們也', '他们也',
    # X又
    '我又', '你又', '他又', '她又', '它又',
    # X才
    '我才', '你才', '他才', '她才', '它才',
    # X都
    '我都', '你都', '他都', '她都', '它都',
    # X會/要/能/可
    '我會', '我会', '你會', '你会', '他會', '他会', '她會', '她会',
    '我要', '你要', '他要', '她要',
    '我能', '你能', '他能', '她能',
    '我可', '你可', '他可', '她可',
    # X便/正
    '我便', '你便', '他便', '她便',
    '我正', '你正', '他正', '她正',
    # 這就/那就
    '這就', '这就', '那就',
    # 讓步/對比
    '卻', '却', '反而', '偏偏',
    # 時間/條件開頭
    '自從', '自从', '直到', '等到', '過了', '过了',
    '當他', '當她', '當我', '當你', '當它', '当他', '当她', '当我', '当你', '当它',
    # 條件/假設
    '不需要', '不管',
    # 補充說明
    '也沒有', '也没有',
]

# 保護詞（不應被拆開的複合詞）
PROTECTED_WORDS: List[str] = [
    '自然而然', '理所當然', '理所当然', '順其自然', '顺其自然',
    '因此', '為此', '为此',
]

# 結構性斷句模式（長模式優先）
# 格式: (正則, 關鍵詞, 保護標記)
LONG_PATTERNS: List[Tuple[str, str, str]] = [
    (r'([^，。？！、；：])(可每當[^，。？！、；：]{1,15}的時候)', '可每當', '__KEMEIDANG__'),
    (r'([^，。？！、；：])(可每当[^，。？！、；：]{1,15}的时候)', '可每当', '__KEMEIDANG2__'),
    (r'([^，。？！、；：])(可每次)', '可每次', '__KEMEICI__'),
    (r'([^，。？！、；：])(每當[^，。？！、；：]{1,15}的時候)', '每當', '__MEIDANG__'),
    (r'([^，。？！、；：])(每当[^，。？！、；：]{1,15}的时候)', '每当', '__MEIDANG2__'),
]

# 結構性斷句模式（短模式）
SHORT_PATTERNS: List[str] = [
    r'([^，。？！、；：])(每次)',
    r'([^，。？！、；：])(每回)',
    r'([^，。？！、；：])(當[^，。？！、；：]{1,15}的時候)',
    r'([^，。？！、；：])(当[^，。？！、；：]{1,15}的时候)',
    r'([^，。？！、；：])(如果[^，。？！、；：]{1,15}的話)',
    r'([^，。？！、；：])(如果[^，。？！、；：]{1,15}的话)',
]

# 標點符號集合（用於判斷是否已有標點）
PUNCTUATION_SET = '，。？！、；：'


# =============================================================================
# 主函數
# =============================================================================

def add_punctuation(text: str) -> str:
    """
    為無標點的中文文本添加標點符號

    Args:
        text: 無標點的中文文本

    Returns:
        添加標點後的文本

    Example:
        >>> add_punctuation("我跟你說啊這個東西真的很厲害")
        '我跟你說啊，這個東西真的很厲害。'
    """
    if not text or not text.strip():
        return text

    text = text.strip()

    # 檢查是否為問句
    is_question = (
        any(text.endswith(w) for w in QUESTION_ENDINGS) or
        any(w in text for w in QUESTION_WORDS)
    )

    result = text

    # 0. 保護複合詞（避免被錯誤斷開）
    for i, word in enumerate(PROTECTED_WORDS):
        result = result.replace(word, f'__PROTECTED_{i}__')

    # 1. 在語氣詞後面加逗號
    for word in PARTICLES_AFTER:
        pattern = f'({word})([^{PUNCTUATION_SET}])'
        result = re.sub(pattern, r'\1，\2', result)

    # 2. 在疑問短語後面加逗號
    for phrase in QUESTION_PHRASES_AFTER:
        pattern = f'({phrase})([^{PUNCTUATION_SET}])'
        result = re.sub(pattern, r'\1，\2', result)

    # 3. 在句子連接詞前加逗號
    for word in SENTENCE_STARTERS:
        pattern = f'([^{PUNCTUATION_SET}])({word})'
        result = re.sub(pattern, r'\1，\2', result)

    # 4. 結構性斷句
    # 4.1 長模式優先處理並保護
    for pattern, word, placeholder in LONG_PATTERNS:
        result = re.sub(pattern, r'\1，\2', result)
        result = result.replace(word, placeholder)

    # 4.2 短模式處理
    for pattern in SHORT_PATTERNS:
        result = re.sub(pattern, r'\1，\2', result)

    # 4.3 還原長模式保護詞
    for pattern, word, placeholder in LONG_PATTERNS:
        result = result.replace(placeholder, word)

    # 還原複合詞保護
    for i, word in enumerate(PROTECTED_WORDS):
        result = result.replace(f'__PROTECTED_{i}__', word)

    # 5. 後處理
    # 清理連續逗號
    result = re.sub(r'，+', '，', result)

    # 移除開頭逗號
    if result.startswith('，'):
        result = result[1:]

    # 加句末標點
    if result and result[-1] not in PUNCTUATION_SET:
        result += '？' if is_question else '。'

    return result


def add_punctuation_batch(texts: List[str]) -> List[str]:
    """
    批量處理多個文本

    Args:
        texts: 文本列表

    Returns:
        處理後的文本列表
    """
    return [add_punctuation(t) for t in texts]


# =============================================================================
# 規則擴展 API
# =============================================================================

def add_particle(word: str) -> None:
    """添加語氣詞（後面會加逗號）"""
    if word not in PARTICLES_AFTER:
        PARTICLES_AFTER.append(word)


def add_sentence_starter(word: str) -> None:
    """添加句子連接詞（前面會加逗號）"""
    if word not in SENTENCE_STARTERS:
        SENTENCE_STARTERS.append(word)


def add_question_word(word: str) -> None:
    """添加疑問詞"""
    if word not in QUESTION_WORDS:
        QUESTION_WORDS.append(word)


def add_protected_word(word: str) -> None:
    """添加保護詞（不會被拆開）"""
    if word not in PROTECTED_WORDS:
        PROTECTED_WORDS.append(word)


# =============================================================================
# 測試與 CLI
# =============================================================================

TEST_CASES = [
    # 語氣詞
    ("我跟你說啊這個東西真的很厲害", "我跟你說啊，這個東西真的很厲害。"),
    # 轉折詞
    ("這個工具完全免費但是有每天一萬字的額度", "這個工具完全免費，但是有每天一萬字的額度。"),
    # 疑問句
    ("你看這樣是不是比較好如果不行的話我們再改", "你看這樣是不是比較好，如果不行的話，我們再改？"),
    # 結構性斷句
    ("每當我想起你的時候心裡就會難過", "每當我想起你的時候，心裡就會難過。"),
    # 複合測試
    ("不是我真的不會那個但是我可以學啊", "不是我真的不會那個，但是我可以學啊。"),
]


def run_tests() -> bool:
    """運行測試用例"""
    passed = 0
    failed = 0

    print("=" * 60)
    print("標點符號規則測試")
    print("=" * 60)

    for i, (input_text, expected) in enumerate(TEST_CASES, 1):
        result = add_punctuation(input_text)
        is_pass = result == expected

        print(f"\n【測試 {i}】{'✓ PASS' if is_pass else '✗ FAIL'}")
        print(f"輸入：{input_text}")
        print(f"期望：{expected}")
        print(f"實際：{result}")

        if is_pass:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"結果：{passed} 通過，{failed} 失敗")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(add_punctuation(text))
    else:
        run_tests()

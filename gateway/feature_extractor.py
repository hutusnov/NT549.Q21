import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyvi import ViTokenizer
from functools import lru_cache

CRITICAL_KEYWORDS = [
    "suy thận",
    "tôphi", "tophi",
    "loét",
    "biến chứng",
    "dịch khớp",
    "xquang",
    "bạch cầu cấp",
    "phẫu thuật",
    "chống chỉ định",
    "corticoid",
    "allopurinol",
    "suy gan",
    "acid uric"
]

def _keyword_found(keyword: str, tokenized_text: str, raw_text: str) -> bool:
    """
    Kiểm tra keyword theo 2 tầng:
    1. Tìm dạng đã tokenize (PyVi ghép bằng _): "suy thận" -> "suy_thận"
    2. Fallback: tìm thẳng trong raw text (lowercase) để bắt trường hợp PyVi tách sai cụm
    """
    tokenized_form = keyword.replace(" ", "_")
    if tokenized_form in tokenized_text:
        return True
    # Fallback: substring match trực tiếp trên text gốc đã lowercase
    if keyword in raw_text:
        return True
    return False

@lru_cache(maxsize=1000)
def analyze_query(query_text: str):
    """
    Chuyển đổi câu hỏi text thành State vector cho RL Agent.
    Input nên được strip() trước khi gọi để tránh cache miss không cần thiết.
    """
    normalized = query_text.lower().strip()
    tokenized_text = ViTokenizer.tokenize(normalized)

    # word_count dựa trên raw text để phản ánh độ dài thực tế của câu
    word_count = len(normalized.split())

    critical_count = sum(
        1 for kw in CRITICAL_KEYWORDS
        if _keyword_found(kw, tokenized_text, normalized)
    )

    return {
        "critical_count": critical_count,
        "is_complex": word_count > 25 or critical_count > 0
    }

if __name__ == "__main__":
    samples = [
        "Bệnh nhân gút có hạt tôphi bị loét, kèm suy thận thì dùng thuốc gì?",
        "Bệnh gút là gì?",
        "Bệnh nhân có biến chứng bạch cầu cấp sau phẫu thuật xử lý thế nào?",
        "Uống allopurinol có chống chỉ định gì không?",
    ]
    for s in samples:
        print(f"[{analyze_query(s.strip())}] {s[:50]}")

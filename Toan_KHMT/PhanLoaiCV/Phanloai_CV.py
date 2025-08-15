# pip install scikit-learn pandas numpy pdfminer.six

import os
import re
import pandas as pd
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ===== 1. Hàm đọc & tiền xử lý =====
def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except:
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # bỏ ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== 2. Cấu hình thư mục dữ liệu =====
train_dir = "train_cv"  # 50 CV train
test_dir = "test_cv"    # 20 CV test

# Đọc dữ liệu train
train_texts = []
for filename in os.listdir(train_dir):
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(os.path.join(train_dir, filename))
        train_texts.append(clean_text(text))

# ===== 3. TF-IDF cho train =====
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
V_train = vectorizer.fit_transform(train_texts)

# ===== 4. NMF =====
n_topics = 5  # số nhóm kỹ năng
nmf_model = NMF(n_components=n_topics, random_state=42)
W_train = nmf_model.fit_transform(V_train)
H = nmf_model.components_
feature_names = vectorizer.get_feature_names_out()

def print_topics(H, feature_names, num_top_words):
    for topic_idx, topic in enumerate(H):
        top_features = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        print(f"Nhóm kỹ năng {topic_idx+1}: {', '.join(top_features)}")

print("=== Nhóm kỹ năng từ NMF (train) ===")
print_topics(H, feature_names, num_top_words=10)

# ===== 5. Hard-code JD =====
jd_skills = [
    "management", "systems", "network", "windows", "server", "microsoft",
    "sql", "database", "security", "communication","oracle", "net"
]

# ===== 6. Áp dụng cho test + so khớp JD =====
results = []
for filename in os.listdir(test_dir):
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(os.path.join(test_dir, filename))
        text_clean = clean_text(text)

        # TF-IDF → NMF transform
        V_test = vectorizer.transform([text_clean])
        W_test = nmf_model.transform(V_test)

        # Xác định nhóm kỹ năng chính
        main_topic_idx = W_test.argmax()
        main_topic_skills = [feature_names[i] for i in H[main_topic_idx].argsort()[:-6:-1]]

        # So khớp JD
        matched = [skill for skill in jd_skills if skill.lower() in text_clean]
        match_percent = (len(matched) / len(jd_skills)) * 100
        status = "Match" if match_percent >= 70 else "Not Match"

        results.append({
            "CV": filename,
            "Main Skills": main_topic_skills,
            "Skills match with JD": matched,
            "Tỉ lệ khớp (%)": round(match_percent, 2),
            "Result": status
        })

# ===== 7. Xuất kết quả =====
df_results = pd.DataFrame(results)
print("\n=== Kết quả phân tích CV test ===")
print(df_results)

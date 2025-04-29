import argparse
import json
import re
import os
from collections import Counter
import numpy as np
import sys

# Để có thể import từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cần clean_report_mimic_cxr từ EnhancedTokenizer
def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def create_synonym_mapping(word_counts, common_patterns):
    """Tạo synonym mapping dựa trên phân tích tần suất từ và pattern"""
    # Bắt đầu với mapping cơ bản dựa trên kinh nghiệm y khoa
    synonyms = {
        # Nhóm mô tả tim
        "cardiomediastinal": "heart",
        "cardiac": "heart", 
        "cardio": "heart",
        "cardiovascular": "heart",
        
        # Nhóm mô tả phổi
        "lungs": "lung",
        "pulmonary": "lung",
        "airspace": "lung", 
        "lung fields": "lung",
        
        # Nhóm mô tả trung thất
        "mediastinal": "mediastinum",
        "mediastinal contour": "mediastinum",
        "mediastinal silhouette": "mediastinum",
        
        # Nhóm mô tả các tình trạng bất thường
        "opacities": "opacity",
        "opacification": "opacity", 
        "infiltration": "infiltrate",
        "airspace disease": "infiltrate",
        "focal airspace disease": "infiltrate", 
        "consolidation": "infiltrate",
        "pneumonia": "infiltrate",
        
        # Cách diễn đạt về "bình thường"
        "unremarkable": "normal",
        "within normal limits": "normal",
        "within limits of normal": "normal",
        "stable": "normal",
        "unchanged": "normal",
        "appear normal": "normal", 
        "appears normal": "normal",
        "intact": "normal",
        "grossly normal": "normal",
        
        # Cách diễn đạt về "sạch/không có bất thường"
        "free of": "clear",
        "without infiltrate": "clear", 
        "without consolidation": "clear",
        "without evidence of": "clear",
        "no evidence of": "clear",
        "negative for": "clear",
        
        # Từ phủ định
        "without": "no",
        "absence": "no",
        "absent": "no",
        "negative": "no",
        "not seen": "no",
        
        # Từ mô tả cấu trúc
        "contour": "silhouette",
        "contours": "silhouette",
        "border": "silhouette",
        "margins": "silhouette",
        
        # Cách diễn đạt trạng thái xương
        "bony": "osseous",
        "skeletal": "osseous", 
        "rib": "osseous", 
        "thoracic spine": "osseous"
    }
    
    # Mở rộng synonym mapping dựa trên pattern phát hiện được
    # Nhưng chỉ với các cụm có độ dài thích hợp (<= 50 ký tự)
    MAX_PHRASE_LENGTH = 50
    
    # Danh sách các pattern không nên mapping (tránh mất thông tin y tế quan trọng)
    blacklist_patterns = [
        " and ", " with ", " along with ", " as well as ",  # tránh mapping câu có nhiều thông tin
        ". ", " . ",  # tránh mapping câu có nhiều mệnh đề
        "effusion", "pneumothorax"  # không map câu có thông tin bệnh lý quan trọng
    ]
    
    for pattern_type, variations in common_patterns.items():
        for phrase, count in variations:
            # Chỉ xem xét cụm được dùng nhiều lần (≥ 10) để đảm bảo phổ biến
            if count < 10:
                continue
                
            # Bỏ qua cụm quá dài
            if len(phrase) > MAX_PHRASE_LENGTH:
                continue
                
            # Kiểm tra xem cụm có chứa mẫu blacklist không
            should_skip = False
            for pattern in blacklist_patterns:
                if pattern in phrase:
                    should_skip = True
                    break
                    
            if should_skip:
                continue
                
            # Thêm vào map nếu hợp lệ
            if pattern_type == "normal_heart" and phrase not in synonyms:
                synonyms[phrase] = "heart is normal"
            elif pattern_type == "clear_lungs" and phrase not in synonyms:
                synonyms[phrase] = "lungs are clear"
            elif pattern_type == "no_effusion" and phrase not in synonyms:
                synonyms[phrase] = "no pleural effusion"
    
    return synonyms

def extract_common_phrases(reports, patterns):
    """Trích xuất các cụm từ thường gặp từ reports dựa trên regex patterns"""
    # Tìm các mẫu câu và từ đồng nghĩa
    phrase_variations = {}
    for pattern_type, pattern_list in patterns.items():
        variations = []
        for pattern in pattern_list:
            for report in reports:
                matches = re.findall(pattern, report.lower())
                if matches:
                    # Trích xuất đoạn văn bản phù hợp
                    match_text = re.search(pattern, report.lower())
                    if match_text:
                        variations.append(match_text.group(0))
        
        if variations:
            phrase_variations[pattern_type] = Counter(variations).most_common()
    
    # Xây dựng medical_phrases dựa trên phân tích
    common_phrases = {
        "heart size is normal": ["normal heart size", "heart is normal in size", 
                                "normal sized heart", "heart size normal"],
        "lungs are clear": ["clear lungs", "lung fields are clear", "both lungs are clear", 
                          "lungs clear", "no focal airspace opacity", "no focal consolidation"],
        "no pleural effusion": ["without pleural effusion", "no effusion", "no evidence of pleural effusion", 
                              "no pleural fluid", "absence of pleural fluid"],
        "no pneumothorax": ["without pneumothorax", "pneumothorax is absent", "no evidence of pneumothorax"],
        "within normal limits": ["normal", "unremarkable", "within normal parameters", "within normal range"],
        "no evidence of": ["without", "negative for", "no sign of", "no finding of", "absent"],
        "bony structures are intact": ["osseous structures are intact", "no acute bony abnormality",
                                     "no fracture", "visualized osseous structures appear intact"],
        "cardiomediastinal silhouette is normal": ["normal cardiomediastinal silhouette", 
                                                 "cardiomediastinal contours are normal"]
    }
    
    # Mở rộng common_phrases dựa trên pattern phát hiện được
    for pattern_type, variations in phrase_variations.items():
        if pattern_type == "normal_heart" and "heart size is normal" in common_phrases:
            for phrase, _ in variations[:5]:  # Chỉ lấy 5 biến thể phổ biến nhất
                if phrase not in common_phrases["heart size is normal"]:
                    common_phrases["heart size is normal"].append(phrase)
                    
        elif pattern_type == "clear_lungs" and "lungs are clear" in common_phrases:
            for phrase, _ in variations[:5]:
                if phrase not in common_phrases["lungs are clear"]:
                    common_phrases["lungs are clear"].append(phrase)
                    
        elif pattern_type == "no_effusion" and "no pleural effusion" in common_phrases:
            for phrase, _ in variations[:5]:
                if phrase not in common_phrases["no pleural effusion"]:
                    common_phrases["no pleural effusion"].append(phrase)
    
    return common_phrases

def create_semantic_groups(word_counts):
    """Tạo các nhóm ngữ nghĩa từ dữ liệu phân tích"""
    semantic_groups = {
        "heart_terms": ["heart", "cardiac", "cardiomediastinal", "cardiovascular", "silhouette", "contour"],
        "lung_terms": ["lung", "lungs", "pulmonary", "airspace", "bronchial"],
        "mediastinum_terms": ["mediastinum", "mediastinal", "hilum", "hilar"],
        "abnormal_terms": ["opacity", "opacities", "infiltrate", "consolidation", "effusion", 
                          "pneumothorax", "edema", "mass", "nodule", "atelectasis", "pneumonia"],
        "normal_terms": ["normal", "unremarkable", "clear", "stable", "intact", "limits"],
        "negation_terms": ["no", "without", "not", "absence", "negative"],
        "descriptor_terms": ["focal", "diffuse", "mild", "moderate", "severe", "small", "large", "acute"]
    }
    
    # Thêm các term phổ biến từ MIMIC-CXR vào nhóm phù hợp
    for term, count in word_counts.most_common(100):  # Chỉ xét 100 từ phổ biến nhất
        # Phân loại từ vào nhóm phù hợp
        if any(term.lower() in group for group in semantic_groups.values()):
            continue  # Đã có trong một nhóm nào đó
            
        if term.lower() in ["chest", "thorax", "ct", "pa", "lateral", "radiograph", "xray", "film"]:
            # Thêm nhóm imaging_terms nếu chưa có
            if "imaging_terms" not in semantic_groups:
                semantic_groups["imaging_terms"] = []
            semantic_groups["imaging_terms"].append(term.lower())
            
        elif term.lower() in ["device", "tube", "catheter", "line", "port", "pacemaker", "lead"]:
            # Thêm nhóm device_terms nếu chưa có
            if "device_terms" not in semantic_groups:
                semantic_groups["device_terms"] = []
            semantic_groups["device_terms"].append(term.lower())
    
    return semantic_groups

def main():
    parser = argparse.ArgumentParser(description='Extract vocabulary info for MIMIC-CXR')
    parser.add_argument('--ann_path', type=str, required=True, help='Path to MIMIC-CXR annotation JSON')
    parser.add_argument('--output', type=str, required=True, help='Output JSON config file')
    parser.add_argument('--threshold', type=int, default=3, help='Threshold for token occurrence')
    args = parser.parse_args()

    # 1. Đọc annotations từ file JSON
    print(f"Loading annotations from {args.ann_path}...")
    with open(args.ann_path, 'r') as f:
        annotations = json.load(f)
    
    # Extract reports (điều chỉnh key phù hợp với cấu trúc của MIMIC-CXR annotation)
    # Đối với MIMIC-CXR, cấu trúc có thể khác so với IU X-ray
    reports = []
    if isinstance(annotations, dict) and "train" in annotations:
        # Cấu trúc giống IU X-ray với các split
        for split in ["train", "val", "test"]:
            if split in annotations:
                for item in annotations[split]:
                    if "report" in item:
                        reports.append(item["report"])
    elif isinstance(annotations, list):
        # Cấu trúc là danh sách các đối tượng
        for item in annotations:
            if "report" in item:
                reports.append(item["report"])
    else:
        # Trích xuất từ bất kỳ trường nào chứa report
        def extract_reports_recursive(obj):
            nonlocal reports
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "report" and isinstance(v, str):
                        reports.append(v)
                    elif isinstance(v, (dict, list)):
                        extract_reports_recursive(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_reports_recursive(item)
                    
        extract_reports_recursive(annotations)
    
    print(f"Found {len(reports)} reports.")
    if len(reports) == 0:
        print("No reports found in the annotation file. Check the structure of the JSON.")
        return
    
    # 2. Clean reports
    print("Cleaning reports...")
    cleaned_reports = [clean_report_mimic_cxr(report) for report in reports]
    
    # 3. Đếm tần suất từ
    all_words = []
    for report in cleaned_reports:
        words = report.split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    print(f"Total tokens: {len(all_words)}, unique tokens: {len(word_counts)}")
    
    # 4. Phát hiện mẫu câu phổ biến bằng regex
    print("Analyzing common patterns...")
    patterns = {
        "normal_heart": [
            r"heart size (is|appears) normal",
            r"normal (heart|cardiac) size", 
            r"heart size normal",
            r"heart is normal in size"
        ],
        "clear_lungs": [
            r"lungs are clear",
            r"clear lungs", 
            r"lung fields are clear",
            r"lungs clear"
        ],
        "no_effusion": [
            r"no pleural effusion",
            r"without pleural effusion",
            r"no effusion"
        ]
    }
    
    # 5. Extract từng phần của config
    print("Extracting synonym mappings...")
    common_patterns = {pattern_type: [] for pattern_type in patterns}
    for pattern_type, pattern_list in patterns.items():
        variations = []
        for pattern in pattern_list:
            for report in cleaned_reports:
                matches = re.findall(pattern, report.lower())
                if matches:
                    match_text = re.search(pattern, report.lower())
                    if match_text:
                        variations.append(match_text.group(0))
        if variations:
            common_patterns[pattern_type] = Counter(variations).most_common()
    
    # 6. Tạo các thành phần config
    synonym_mapping = create_synonym_mapping(word_counts, common_patterns)
    medical_phrases = extract_common_phrases(cleaned_reports, patterns)
    semantic_groups = create_semantic_groups(word_counts)
    
    # 7. Build config và lưu vào file JSON
    config = {
        'synonym_mapping': synonym_mapping,
        'medical_phrases': medical_phrases,
        'semantic_groups': semantic_groups
    }
    
    print(f"Writing config to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Done! Extracted {len(synonym_mapping)} synonyms, {len(medical_phrases)} phrases, {len(semantic_groups)} semantic groups.")

if __name__ == '__main__':
    main() 
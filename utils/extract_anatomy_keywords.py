import json
import re
import sys
import os
from collections import Counter
from tqdm import tqdm

# Thêm đường dẫn đến thư mục gốc của dự án vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Thêm import cho Tokenizer
from modules.tokenizers import Tokenizer

# --- 1. Định nghĩa các nhóm từ khóa ban đầu ---
# Tên nhóm tạm thời, sẽ được cập nhật sau dựa trên tần suất
# Đây là danh sách khởi tạo, có thể cần tinh chỉnh thêm
INITIAL_KEYWORD_GROUPS = {
    "group_heart": ["heart", "cardiac", "cardiomediastinal", "pericardial"],
    "group_lung": ["lung", "pulmonary", "hilar", "bronchovascular", "parenchyma", "interstitial", "airspace", "pleural", "pleura", "lobe", "lingula", "apex", "apices", "base"],
    "group_mediastinum": ["mediastinum", "mediastinal"],
    "group_aorta": ["aorta", "aortic"],
    "group_vasculature": ["vascular", "vasculature", "vessel", "artery", "vein"],
    "group_bone": ["bone", "osseous", "rib", "spine", "vertebra", "vertebral", "thoracic", "clavicle", "sternum", "scapula", "costal", "costophrenic"],
    "group_diaphragm": ["diaphragm", "hemidiaphragm", "subphrenic"],
    "group_abdomen": ["abdomen", "abdominal", "bowel", "hepatic", "splenic", "stomach", "gastric"],
    "group_soft_tissue": ["soft tissue", "subcutaneous", "axilla", "axillary", "breast", "chest wall"],
    "group_tube_device": ["tube", "line", "catheter", "pacemaker", "device", "hardware", "clip", "valve", "stent", "wire", "drain"],
    "group_neck": ["neck", "trachea", "thyroid", "carotid"],
    "group_general": ["opacity", "consolidation", "effusion", "edema", "atelectasis", "pneumothorax", "nodule", "mass", "lesion", "density", "scarring", "fibrosis", "calcification"] # Các từ mô tả chung, không phải bộ phận cụ thể
}

# Flatten the dictionary to get a list of all keywords for searching
ALL_KEYWORDS = [keyword for group in INITIAL_KEYWORD_GROUPS.values() for keyword in group]
# Sort keywords by length descending to match longer phrases first (e.g., "soft tissue" before "tissue")
ALL_KEYWORDS.sort(key=len, reverse=True)

# Compile regex pattern for finding keywords (case-insensitive, word boundaries)
# Pattern ensures we match whole words/phrases
KEYWORD_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in ALL_KEYWORDS) + r')\b', re.IGNORECASE)

def preprocess_text(text):
    """Lowercases and tokenizes text."""
    if not isinstance(text, str):
        return ""
    return text.lower()

def extract_keywords(text, keyword_pattern):
    """Extracts keywords from a single text using regex."""
    found_keywords = []
    if not isinstance(text, str):
        return found_keywords
        
    processed_text = preprocess_text(text)
    matches = keyword_pattern.finditer(processed_text)
    for match in matches:
        # Normalize the found keyword to lowercase for consistent counting
        found_keywords.append(match.group(1).lower())
    return found_keywords

def process_annotations(json_path, tokenizer=None, report_key='report', splits=None):
    """
    Reads the annotation JSON, extracts keywords from reports,
    counts keyword frequencies, and determines final group names.

    Args:
        json_path (str): Path to the annotations JSON file.
        tokenizer (Tokenizer, optional): Tokenizer để chuyển đổi từ khóa thành token ID.
        report_key (str): The key in the JSON dictionary that contains the report text.
        splits (list): List of data splits to process (e.g., ['train', 'val', 'test']).
                      If None, all available splits will be processed.

    Returns:
        tuple: Một tuple chứa:
            - text_groups: Dictionary với khóa là tên nhóm (từ khóa phổ biến nhất) và giá trị là danh sách từ khóa văn bản
            - token_groups: Dictionary với khóa là token ID và giá trị là danh sách token ID (nếu có tokenizer)
            - keyword_counts: Đếm tần suất của các từ khóa
    """
    print(f"Reading annotations from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {json_path}")
        return {}, {}, Counter()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return {}, {}, Counter()

    all_found_keywords = Counter()
    
    # Xác định các splits cần xử lý
    if splits is None and isinstance(annotations, dict):
        # Nếu không chỉ định splits, sử dụng tất cả các splits có sẵn
        splits = list(annotations.keys())
        print(f"Processing all available splits: {splits}")
    elif splits is None:
        # Nếu annotations không phải là dict, xử lý toàn bộ
        splits = ["all"]

    # Lấy danh sách các mẫu cần xử lý từ tất cả các splits
    items_to_process = []
    if isinstance(annotations, dict):
        for split in splits:
            if split in annotations:
                if isinstance(annotations[split], list):
                    items_to_process.extend(annotations[split])
                    print(f"Added {len(annotations[split])} samples from '{split}' split")
                else:
                    print(f"Warning: Split '{split}' is not a list. Skipping.")
            else:
                print(f"Warning: Split '{split}' not found in annotations.")
    elif isinstance(annotations, list):
        items_to_process = annotations
        print(f"Annotations is a list with {len(annotations)} items")
    else:
        print("Error: Unexpected JSON structure. Expected list or dict.")
        return {}, {}, Counter()

    print(f"Processing {len(items_to_process)} reports...")
    for item in tqdm(items_to_process):
        if not isinstance(item, dict):
            # print(f"Warning: Skipping non-dictionary item: {item}")
            continue
        report_text = item.get(report_key)
        if report_text:
            keywords_in_report = extract_keywords(report_text, KEYWORD_PATTERN)
            all_found_keywords.update(keywords_in_report)
        # else:
            # print(f"Warning: Report key '{report_key}' not found or empty in item: {item.keys()}")


    print("\nKeyword extraction complete. Determining group names...")
    
    text_keyword_groups = {}  # Dictionary với khóa là văn bản, giá trị là danh sách từ văn bản
    token_keyword_groups = {}  # Dictionary với khóa là token ID, giá trị là danh sách token ID
    keyword_to_group_map = {kw: group_name for group_name, kws in INITIAL_KEYWORD_GROUPS.items() for kw in kws}

    processed_initial_groups = set()

    # Iterate through keywords found, ordered by frequency
    for keyword, count in all_found_keywords.most_common():
        if keyword not in keyword_to_group_map:
            # This keyword might have been found due to substring issues or needs adding to INITIAL_GROUPS
            # print(f"Warning: Found keyword '{keyword}' not in predefined groups. Skipping.")
            continue

        initial_group_name = keyword_to_group_map[keyword]
        
        # If we haven't processed this initial group yet
        if initial_group_name not in processed_initial_groups:
            # Sử dụng keyword làm tên nhóm cho text_groups
            text_group_name = keyword
            # Get all keywords belonging to the same initial group
            group_members = INITIAL_KEYWORD_GROUPS[initial_group_name]
            
            # Lưu vào text_keyword_groups
            text_keyword_groups[text_group_name] = group_members
            
            # Lưu vào token_keyword_groups nếu có tokenizer
            if tokenizer is not None:
                token_id = tokenizer.get_id_by_token(keyword)
                # Chuyển đổi tất cả từ khóa trong group_members thành token ID
                token_group_members = [tokenizer.get_id_by_token(member) for member in group_members]
                token_keyword_groups[token_id] = token_group_members
                print(f"  Group '{keyword}' (token ID: {token_id}) established with token members")
            else:
                print(f"  Group '{text_group_name}' (from {initial_group_name}) established with members: {group_members}")
                
            processed_initial_groups.add(initial_group_name)

    # Check for any initial groups that had zero hits
    for initial_group_name, members in INITIAL_KEYWORD_GROUPS.items():
        if initial_group_name not in processed_initial_groups:
            print(f"Warning: No keywords found for initial group '{initial_group_name}' with members: {members}. Group omitted.")

    print("\nProcessing finished.")
    return text_keyword_groups, token_keyword_groups, all_found_keywords

if __name__ == "__main__":
    ANNOTATION_FILE = "/home/sowwn/Workspace/ws/2025/Dataset/MIMIC-CXR/mimic_cxr/annotations.json"
    # IMPORTANT: Verify this key matches your JSON structure!
    REPORT_TEXT_KEY = "report" # Or 'impression', 'findings', etc. 
    # Xử lý tất cả các splits (train, val, test)
    SPLITS_TO_PROCESS = ["train", "val", "test"]

    # Tạo một Mock Config để khởi tạo tokenizer
    from types import SimpleNamespace
    args = SimpleNamespace(
        ann_path=ANNOTATION_FILE,
        threshold=3,  # Ngưỡng tần suất từ
        dataset_name='mimic_cxr'  # Tên dataset
    )
    
    # Khởi tạo tokenizer (optional)
    try:
        tokenizer = Tokenizer(args)
        print("Tokenizer initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize tokenizer: {e}")
        print("Will only generate text-based keyword groups.")
        tokenizer = None

    # Xử lý các annotations và trả về cả text_groups và token_groups
    text_groups, token_groups, keyword_counts = process_annotations(
        ANNOTATION_FILE, tokenizer, report_key=REPORT_TEXT_KEY, splits=SPLITS_TO_PROCESS
    )

    print("\n--- Text-based Keyword Groups ---")
    if not text_groups:
        print("No text-based keyword groups were generated.")
    else:
        for group_name, members in text_groups.items():
            print(f"'{group_name}': {members}")

    # Hiển thị token groups nếu có
    if tokenizer is not None:
        print("\n--- Token-based Keyword Groups ---")
        if not token_groups:
            print("No token-based keyword groups were generated.")
        else:
            for token_id, token_members in token_groups.items():
                keyword = tokenizer.get_token_by_id(token_id)
                text_members = [tokenizer.get_token_by_id(token) for token in token_members]
                print(f"Token '{token_id}' (for '{keyword}') -> token members: {token_members}")
                print(f"  Text equivalents: {text_members}")

    print("\n--- Top 50 Keyword Frequencies ---")
    if not keyword_counts:
        print("No keywords were counted.")
    else:
        for keyword, count in keyword_counts.most_common(50):
            print(f"{keyword}: {count}")

    # Lưu text groups
    output_text_groups_file = "utils/anatomical_keyword_groups.json"
    print(f"\nSaving text-based groups to {output_text_groups_file}")
    try:
        with open(output_text_groups_file, 'w') as f:
            json.dump(text_groups, f, indent=4)
        print("Successfully saved text-based groups.")
    except Exception as e:
        print(f"Error saving text-based groups: {e}")
    
    # Lưu token groups nếu có
    if tokenizer is not None:
        output_token_groups_file = "utils/anatomical_token_groups.json"
        print(f"\nSaving token-based groups to {output_token_groups_file}")
        try:
            with open(output_token_groups_file, 'w') as f:
                # Chuyển đổi int keys thành strings để JSON hợp lệ
                token_groups_serializable = {str(k): v for k, v in token_groups.items()}
                json.dump(token_groups_serializable, f, indent=4)
            print("Successfully saved token-based groups.")
        except Exception as e:
            print(f"Error saving token-based groups: {e}")
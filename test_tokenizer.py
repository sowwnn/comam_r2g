import argparse
import os
import json
from collections import Counter
from modules.tokenizers_enhanced import EnhancedTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Tham số từ mô hình gốc
    parser.add_argument('--dataset_name', type=str, default='iu_xray', help='dataset name')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/iu_xray.json', help='path to annotation file')
    parser.add_argument('--threshold', type=int, default=3, help='threshold')
    parser.add_argument('--save_dir', type=str, default='weights/iu_xray/con_mam_v3', help='path to save dir')
    
    args = parser.parse_args()
    return args

def analyze_vocabulary_stats(tokenizer, gts_path=None):
    """Phân tích thống kê từ vựng để so sánh trước và sau khi tối ưu"""
    
    # Lấy các báo cáo từ tập annotate
    reports = []
    for split in ['train', 'val', 'test']:
        for example in tokenizer.ann[split]:
            reports.append(tokenizer.clean_report(example['report']))
    
    # Thêm từ gts.csv nếu có
    if gts_path and os.path.exists(gts_path):
        with open(gts_path, 'r') as f:
            gts_reports = f.readlines()
            for report in gts_reports:
                reports.append(tokenizer.clean_report(report))
    
    print(f"Tổng số báo cáo: {len(reports)}")
    
    # Phân tích từ vựng trước khi áp dụng mapping từ đồng nghĩa
    original_tokens = []
    for report in reports:
        tokens = report.split()
        original_tokens.extend(tokens)
    
    original_counter = Counter(original_tokens)
    
    # Phân tích từ vựng sau khi áp dụng mapping từ đồng nghĩa
    normalized_tokens = []
    for report in reports:
        tokens = report.split()
        normalized = []
        for token in tokens:
            if token.lower() in tokenizer.synonym_mapping:
                normalized.append(tokenizer.synonym_mapping[token.lower()])
            else:
                normalized.append(token)
        normalized_tokens.extend(normalized)
    
    normalized_counter = Counter(normalized_tokens)
    
    # Phân tích so sánh
    print(f"Tổng số từ trước khi mapping: {len(original_tokens)}")
    print(f"Số từ khác nhau trước khi mapping: {len(original_counter)}")
    
    print(f"\nTổng số từ sau khi mapping: {len(normalized_tokens)}")
    print(f"Số từ khác nhau sau khi mapping: {len(normalized_counter)}")
    
    print(f"\nTỷ lệ giảm từ vựng: {(1.0 - len(normalized_counter)/len(original_counter))*100:.2f}%")
    
    # Top từ phổ biến nhất trước và sau khi mapping
    print("\nTop 20 từ phổ biến nhất trước khi mapping:")
    for token, count in original_counter.most_common(20):
        print(f"  {token}: {count}")
    
    print("\nTop 20 từ phổ biến nhất sau khi mapping:")
    for token, count in normalized_counter.most_common(20):
        print(f"  {token}: {count}")
    
    # Phân tích theo nhóm ngữ nghĩa
    print("\nPhân phối từ vựng theo nhóm ngữ nghĩa:")
    for group_name, terms in tokenizer.semantic_groups.items():
        group_count = sum(normalized_counter.get(term, 0) for term in terms)
        print(f"  {group_name}: {group_count} ({group_count/len(normalized_tokens)*100:.2f}%)")
    
    # Phân tích các từ đồng nghĩa được map
    print("\nCác từ được ánh xạ phổ biến nhất:")
    mapping_usage = Counter()
    for token in original_tokens:
        if token.lower() in tokenizer.synonym_mapping:
            mapping_usage[f"{token.lower()} -> {tokenizer.synonym_mapping[token.lower()]}"] += 1
    
    for mapping, count in mapping_usage.most_common(20):
        print(f"  {mapping}: {count}")
    
    # Phân tích các cụm từ phổ biến
    print("\nCác cụm từ phổ biến và tần suất:")
    phrase_counts = {}
    for report in reports:
        for phrase in tokenizer.medical_phrases:
            if phrase.lower() in report.lower():
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    
    for phrase, count in sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{phrase}': {count}")
    
    # Xuất thống kê cho báo cáo
    stats = {
        "vocab_size_before": len(original_counter),
        "vocab_size_after": len(normalized_counter),
        "reduction_percentage": (1.0 - len(normalized_counter)/len(original_counter))*100,
        "top_tokens_before": {t: c for t, c in original_counter.most_common(50)},
        "top_tokens_after": {t: c for t, c in normalized_counter.most_common(50)},
        "semantic_groups": {g: sum(normalized_counter.get(t, 0) for t in terms) 
                           for g, terms in tokenizer.semantic_groups.items()},
        "mapping_usage": {m: c for m, c in mapping_usage.most_common(50)},
        "phrase_usage": phrase_counts
    }
    
    # Lưu báo cáo thống kê
    with open(os.path.join(tokenizer.ann_path.rsplit('/', 1)[0], "vocabulary_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats

def compare_tokenization(tokenizer, sample_reports):
    """So sánh tokenization trước và sau khi tối ưu hóa tokenizer"""
    print("\nSo sánh tokenization trên các mẫu:")
    
    for i, report in enumerate(sample_reports):
        print(f"\nMẫu {i+1}: {report}")
        
        # Tokenize theo cách thông thường
        tokens = tokenizer.clean_report(report).split()
        print(f"Tokenize thông thường: {' '.join(tokens)}")
        
        # Tokenize với enhanced tokenizer
        ids = tokenizer(report)
        decoded = tokenizer.decode(ids)
        print(f"Tokenize với enhanced tokenizer: {decoded}")
        
        # Hiển thị ánh xạ từ đồng nghĩa nếu có
        mappings = []
        for token in tokens:
            if token.lower() in tokenizer.synonym_mapping:
                mappings.append(f"{token} -> {tokenizer.synonym_mapping[token.lower()]}")
        
        if mappings:
            print(f"Các từ được map: {', '.join(mappings)}")

def test_similar_words(tokenizer):
    """Kiểm tra chức năng tìm từ tương tự ngữ nghĩa"""
    print("\nKiểm tra chức năng tìm từ tương tự ngữ nghĩa:")
    
    test_words = ["heart", "lung", "normal", "infiltrate", "opacity", "no", "mild"]
    
    for word in test_words:
        similar_words = tokenizer.get_semantically_similar_tokens(word)
        print(f"Từ tương tự với '{word}': {', '.join(similar_words)}")

def save_reports_with_normalized_vocabulary(tokenizer, output_path):
    """Lưu lại báo cáo với từ vựng chuẩn hóa"""
    print("\nĐang chuẩn hóa và lưu báo cáo với từ vựng đã mapping...")
    
    normalized_reports = {}
    for split in ['train', 'val', 'test']:
        normalized_reports[split] = []
        for example in tokenizer.ann[split]:
            report = tokenizer.clean_report(example['report'])
            
            # Tokenize và decode để áp dụng hết các quy tắc chuẩn hóa
            ids = tokenizer(report)
            normalized_report = tokenizer.decode(ids)
            
            # Tạo bản copy của example và cập nhật report
            new_example = example.copy()
            new_example['report'] = normalized_report
            normalized_reports[split].append(new_example)
    
    # Lưu vào file
    with open(output_path, 'w') as f:
        json.dump(normalized_reports, f, indent=2)
    
    print(f"Đã lưu báo cáo chuẩn hóa vào: {output_path}")

def main():
    args = parse_args()
    
    print("Khởi tạo EnhancedTokenizer...")
    tokenizer = EnhancedTokenizer(args)
    
    # Phân tích và báo cáo thống kê từ vựng
    gts_path = os.path.join(args.save_dir, "gts.csv")
    stats = analyze_vocabulary_stats(tokenizer, gts_path)
    
    # Kiểm tra trên một số mẫu báo cáo
    sample_reports = [
        "The heart size and pulmonary vascularity appear within normal limits. No focal airspace consolidation, pleural effusion or pneumothorax identified.",
        "Heart size is normal. Lungs are clear without focal airspace disease. No pleural effusion or pneumothorax. No acute bony abnormality.",
        "Cardiomediastinal silhouette is unremarkable. Pulmonary vasculature is within normal limits. No focal consolidation, effusion, or pneumothorax is identified. No acute osseous abnormality.",
        "Cardiac silhouette is mildly enlarged. Mild pulmonary edema with small bilateral pleural effusions. No focal consolidation or pneumothorax.",
        "The cardiomediastinal silhouette is normal in size and contour. The lungs are clear. There is no pneumothorax or pleural effusion. The osseous structures are intact."
    ]
    
    compare_tokenization(tokenizer, sample_reports)
    
    # Kiểm tra chức năng tìm từ tương tự
    test_similar_words(tokenizer)
    
    # Lưu báo cáo với từ vựng chuẩn hóa
    normalized_path = args.ann_path.replace('.json', '_normalized.json')
    save_reports_with_normalized_vocabulary(tokenizer, normalized_path)
    
    print("\nĐã hoàn thành kiểm tra tokenizer!")

if __name__ == "__main__":
    main() 
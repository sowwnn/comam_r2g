import torch
import numpy as np
import pandas as pd
import os
import argparse
import re
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from dev import build_model, get_model_config
from modules.metrics import compute_scores
from config.configs import Test_Config
from test import generate_batch as original_generate_batch
from vocabulary_enhancement import VocabularyEnhancement
from collections import Counter

def main(args):
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cố định seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Tạo tokenizer
    tokenizer = Tokenizer(args)
    
    # Tạo data loader
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    
    # Xây dựng model
    model_config = get_model_config()
    model = build_model(args, tokenizer, model_config)

    # Load checkpoint với weights_only=True để tránh các vấn đề bảo mật
    checkpoint = torch.load(args.load, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Lấy hàm tính metrics
    metrics = compute_scores

    # Cấu hình cải thiện từ vựng
    enhance_config = {
        'temperature': args.temperature,
        'repeat_penalty': args.repeat_penalty,
        'top_p': args.top_p,
        'min_length': args.min_length,
        'use_synonyms': args.use_synonyms
    }
    
    # Tạo đối tượng cải thiện từ vựng
    vocab_enhancer = VocabularyEnhancement(tokenizer, enhance_config)
    
    # Sửa đổi hàm generate_batch
    if args.enhance_generation:
        generate_batch = vocab_enhancer.modify_generate_batch(original_generate_batch)
    else:
        generate_batch = original_generate_batch
    
    # Thực hiện test
    print(f"Running enhanced test with settings:")
    print(f"Temperature: {args.temperature}")
    print(f"Repeat penalty: {args.repeat_penalty}")
    print(f"Top-p: {args.top_p}")
    print(f"Min length: {args.min_length}")
    print(f"Use synonyms: {args.use_synonyms}")
    print(f"Enhance generation: {args.enhance_generation}")
    
    # Thực hiện test
    test_gts, test_res = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch[1].to(device)
            reports_ids = batch[2].to(device)
            
            # Sinh báo cáo
            reports, _ = generate_batch(
                model, images, tokenizer, device, 
                max_length=args.max_length,
                temperature=args.temperature,
                repeat_penalty=args.repeat_penalty,
                top_p=args.top_p,
                min_length=args.min_length
            )
            
            # Giải mã ground truth
            ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)
            
        # Tính metrics
        test_met = metrics({i: [gt] for i, gt in enumerate(test_gts)},
                         {i: [re] for i, re in enumerate(test_res)})
        log = {**{'test_' + k: v for k, v in test_met.items()}}
        print(log)

        # Tạo thư mục output nếu chưa tồn tại
        output_dir = os.path.join(args.save_dir, f"enhanced_t{args.temperature}_r{args.repeat_penalty}_p{args.top_p}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu kết quả
        test_res_df, test_gts_df = pd.DataFrame(test_res), pd.DataFrame(test_gts)
        res_path = os.path.join(output_dir, "res.csv")
        gts_path = os.path.join(output_dir, "gts.csv")
        
        test_res_df.to_csv(res_path, index=False, header=False)
        test_gts_df.to_csv(gts_path, index=False, header=False)
        
        # Lưu metrics
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            for k, v in log.items():
                f.write(f"{k}: {v}\n")
                
        # Lưu một số mẫu kết quả để so sánh
        samples_path = os.path.join(output_dir, "samples.txt")
        with open(samples_path, 'w') as f:
            for i in range(min(10, len(test_gts))):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Ground Truth: {test_gts[i]}\n")
                f.write(f"Generated: {test_res[i]}\n")
                f.write(f"---\n")
                
        # Phân tích từ vựng
        vocab_analysis = {
            "ground_truth": analyze_vocabulary(test_gts),
            "generated": analyze_vocabulary(test_res)
        }
        
        vocab_path = os.path.join(output_dir, "vocab_analysis.txt")
        with open(vocab_path, 'w') as f:
            f.write("Ground Truth Vocabulary Analysis:\n")
            f.write(f"Total words: {vocab_analysis['ground_truth']['total_words']}\n")
            f.write(f"Unique words: {vocab_analysis['ground_truth']['unique_words']}\n")
            f.write(f"Vocabulary richness: {vocab_analysis['ground_truth']['vocabulary_richness']:.4f}\n\n")
            
            f.write("Generated Vocabulary Analysis:\n")
            f.write(f"Total words: {vocab_analysis['generated']['total_words']}\n")
            f.write(f"Unique words: {vocab_analysis['generated']['unique_words']}\n")
            f.write(f"Vocabulary richness: {vocab_analysis['generated']['vocabulary_richness']:.4f}\n\n")
            
            f.write("Top 20 words in ground truth:\n")
            for word, count in vocab_analysis['ground_truth']['word_counts'].most_common(20):
                f.write(f"{word}: {count}\n")
            
            f.write("\nTop 20 words in generated text:\n")
            for word, count in vocab_analysis['generated']['word_counts'].most_common(20):
                f.write(f"{word}: {count}\n")
                
            # Báo cáo bổ sung về từ vựng chuyên ngành
            f.write("\nPhân tích thuật ngữ y khoa:\n")
            medical_terms = [
                "heart", "cardiac", "lung", "lungs", "mediastinum", "pleural", 
                "pneumothorax", "effusion", "consolidation", "infiltrate", "opacity"
            ]
            
            gt_medical_count = sum(vocab_analysis['ground_truth']['word_counts'].get(term, 0) for term in medical_terms)
            gen_medical_count = sum(vocab_analysis['generated']['word_counts'].get(term, 0) for term in medical_terms)
            
            f.write(f"Tỷ lệ thuật ngữ y khoa trong ground truth: {gt_medical_count/vocab_analysis['ground_truth']['total_words']:.4f}\n")
            f.write(f"Tỷ lệ thuật ngữ y khoa trong generated: {gen_medical_count/vocab_analysis['generated']['total_words']:.4f}\n")
            
            # So sánh cấu trúc câu
            f.write("\nSo sánh cấu trúc câu:\n")
            gt_sentence_patterns = analyze_sentence_patterns(test_gts)
            gen_sentence_patterns = analyze_sentence_patterns(test_res)
            
            f.write(f"Số lượng mẫu câu trong ground truth: {len(gt_sentence_patterns)}\n")
            f.write(f"Số lượng mẫu câu trong generated: {len(gen_sentence_patterns)}\n")
            
            f.write("\nCác mẫu câu phổ biến trong ground truth:\n")
            for pattern, count in sorted(gt_sentence_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"{pattern}: {count}\n")
                
            f.write("\nCác mẫu câu phổ biến trong generated:\n")
            for pattern, count in sorted(gen_sentence_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"{pattern}: {count}\n")
        
        return log


def analyze_vocabulary(texts):
    """Phân tích từ vựng trong một tập hợp văn bản"""
    all_words = []
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counts)
    
    # Phân tích thêm về cấu trúc câu
    sentence_length = sum(len(text.split('.')) for text in texts) / len(texts)
    
    # Phân loại từ theo loại
    anatomical_terms = ["heart", "lung", "lungs", "pleural", "mediastinum", "bony", "chest"]
    finding_terms = ["opacity", "effusion", "pneumothorax", "consolidation", "edema", "fracture"]
    descriptor_terms = ["normal", "clear", "free", "unremarkable", "intact"]
    
    anatomical_count = sum(word_counts.get(term, 0) for term in anatomical_terms)
    finding_count = sum(word_counts.get(term, 0) for term in finding_terms)
    descriptor_count = sum(word_counts.get(term, 0) for term in descriptor_terms)
    
    # Các cụm từ phổ biến
    common_phrases = []
    for text in texts:
        # Tìm các cụm từ phổ biến như "no evidence of", "within normal limits"
        phrases = ["no evidence of", "within normal limits", "lungs are clear", 
                  "normal in size", "no pneumothorax", "no pleural effusion"]
        for phrase in phrases:
            if phrase in text.lower():
                common_phrases.append(phrase)
    
    phrase_counts = Counter(common_phrases)
    
    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "vocabulary_richness": unique_words / total_words if total_words > 0 else 0,
        "word_counts": word_counts,
        "avg_sentence_length": sentence_length,
        "anatomical_term_ratio": anatomical_count / total_words if total_words > 0 else 0,
        "finding_term_ratio": finding_count / total_words if total_words > 0 else 0,
        "descriptor_term_ratio": descriptor_count / total_words if total_words > 0 else 0,
        "common_phrases": phrase_counts
    }


def analyze_sentence_patterns(texts):
    """Phân tích các mẫu câu phổ biến"""
    patterns = {}
    
    # Các mẫu câu phổ biến trong báo cáo X-quang
    pattern_templates = [
        r"(heart|cardiac|cardiomediastinal) (size|silhouette) (is|appears) (normal|within normal limits)",
        r"(lungs|lung fields) (are|is) (clear|free of)",
        r"no (pneumothorax|pleural effusion|effusion|focal consolidation)",
        r"(normal|unremarkable) (cardiomediastinal|cardiac) (silhouette|contour)",
        r"no (evidence|sign) of (pneumonia|infiltrate|edema|fracture)"
    ]
    
    for text in texts:
        text_lower = text.lower()
        for pattern in pattern_templates:
            if re.search(pattern, text_lower):
                if pattern not in patterns:
                    patterns[pattern] = 0
                patterns[pattern] += 1
                
    return patterns


if __name__ == '__main__':
    from collections import Counter
    
    # Tạo parser để nhận tham số từ command line
    parser = argparse.ArgumentParser(description='Enhanced test for medical report generation')
    
    # Các tham số từ Test_Config
    args = Test_Config()
    
    # Thêm các tham số tuỳ chỉnh
    parser.add_argument('--temperature', type=float, default=args.temperature,
                        help='Temperature parameter for generation')
    parser.add_argument('--repeat_penalty', type=float, default=1.5,
                        help='Repetition penalty')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--min_length', type=int, default=30,
                        help='Minimum length of generated text')
    parser.add_argument('--use_synonyms', action='store_true',
                        help='Use synonym replacement')
    parser.add_argument('--enhance_generation', action='store_true',
                        help='Enable enhanced generation methods')
    
    # Parse tham số
    cmd_args = parser.parse_args()
    
    # Cập nhật các tham số từ command line vào args
    args.temperature = cmd_args.temperature
    args.repeat_penalty = cmd_args.repeat_penalty
    args.top_p = cmd_args.top_p
    args.min_length = cmd_args.min_length
    args.use_synonyms = cmd_args.use_synonyms
    args.enhance_generation = cmd_args.enhance_generation
    
    # Chạy main
    main(args) 
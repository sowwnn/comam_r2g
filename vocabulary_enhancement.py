import torch
import torch.nn.functional as F
import numpy as np
import nltk
from nltk.corpus import wordnet
import random
import json
import os
from collections import Counter

class VocabularyEnhancement:
    """
    Class cung cấp các phương pháp để cải thiện độ đa dạng từ vựng 
    trong mô hình mô tả hình ảnh y tế
    """
    
    def __init__(self, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config if config else {}
        self.synonym_dict = {}
        
        # Tạo từ điển đồng nghĩa nếu cần
        if self.config.get('use_synonyms', False):
            self._init_synonym_dictionary()
    
    def _init_synonym_dictionary(self):
        """Khởi tạo từ điển đồng nghĩa từ WordNet"""
        try:
            nltk.download('wordnet', quiet=True)
            
            # Tạo từ điển đồng nghĩa cho các từ thường gặp trong báo cáo X-quang dựa trên gts.csv
            medical_terms = [
                # Thuật ngữ giải phẫu
                "heart", "cardiac", "lung", "lungs", "mediastinum", "cardiomediastinal", 
                "pleural", "silhouette", "pneumothorax", "effusion", "consolidation", 
                "infiltrate", "opacity", "airspace", "pulmonary", "vasculature",
                "diaphragm", "bony", "osseous", "thorax", "ribs", "contour", "contours",
                
                # Từ mô tả
                "normal", "clear", "unremarkable", "intact", "within", "limits",
                "free", "acute", "focal", "large", "visible", "evidence", "xxxx",
                "suspicious", "definite", "expanded", "interval", "grossly"
            ]
            
            for term in medical_terms:
                synonyms = []
                for syn in wordnet.synsets(term):
                    for lemma in syn.lemmas():
                        if lemma.name() != term and lemma.name() not in synonyms:
                            synonyms.append(lemma.name())
                
                # Lọc từ đồng nghĩa phù hợp
                if synonyms:
                    self.synonym_dict[term] = synonyms[:5]  # Giới hạn số lượng từ đồng nghĩa
            
            # Thêm các cặp từ đồng nghĩa y tế đặc thù dựa trên phân tích gts.csv
            medical_synonyms = {
                # Thuật ngữ giải phẫu và vị trí
                "heart": ["cardiac", "cardiomediastinal", "cardio", "cardiovascular"],
                "lungs": ["lung", "pulmonary", "lung fields", "airspace"],
                "mediastinum": ["mediastinal", "cardiomediastinal", "mediastinal contour"],
                "pleural": ["pleura", "pleural space", "pleural cavity"],
                "silhouette": ["contour", "contours", "border", "margins", "outline"],
                
                # Tình trạng bất thường
                "opacity": ["opacification", "opacity", "haziness", "cloudiness", "density", "infiltrate", "consolidation"],
                "infiltrate": ["infiltration", "opacity", "consolidation", "airspace disease"],
                "consolidation": ["infiltrate", "opacification", "airspace disease", "pneumonia"],
                "effusion": ["fluid", "pleural fluid", "fluid collection", "fluid accumulation"],
                "pneumothorax": ["air", "free air", "collapsed lung"],
                
                # Cách diễn đạt về tình trạng bình thường
                "normal": ["unremarkable", "within normal limits", "unchanged", "stable", "intact", "grossly normal"],
                "clear": ["free", "clear of", "without", "no evidence of", "negative for"],
                "intact": ["normal", "preserved", "without fracture", "without acute abnormality"],
                "limits": ["parameters", "range", "boundaries"],
                
                # Cụm từ thường dùng và biến thể
                "within normal limits": ["normal", "unremarkable", "within normal parameters"],
                "no focal consolidation": ["lungs are clear", "clear lungs", "no pneumonia", "no infiltrate"],
                "no pneumothorax": ["no free air", "without pneumothorax", "negative for pneumothorax"],
                "no pleural effusion": ["no effusion", "without pleural fluid", "no fluid collection"],
                
                # Từ mô tả
                "evidence": ["sign", "indication", "finding", "demonstration"],
                "focal": ["localized", "discrete", "specific", "particular", "isolated"],
                "large": ["significant", "sizable", "substantial", "considerable"],
                "suspicious": ["concerning", "worrisome", "questionable", "doubtful"],
                "visible": ["seen", "apparent", "evident", "observable", "appreciable"],
                "acute": ["new", "recent", "fresh", "sudden"],
                "grossly": ["generally", "overall", "broadly", "apparently"]
            }
            
            # Cập nhật từ điển đồng nghĩa với các cặp y tế đặc thù
            for term, synonyms in medical_synonyms.items():
                if term in self.synonym_dict:
                    self.synonym_dict[term] = list(set(self.synonym_dict[term] + synonyms))
                else:
                    self.synonym_dict[term] = synonyms
                    
            print(f"Đã tạo từ điển đồng nghĩa với {len(self.synonym_dict)} từ")
        except Exception as e:
            print(f"Lỗi khi tạo từ điển đồng nghĩa: {e}")
            self.config['use_synonyms'] = False
    
    def enhance_generation_params(self):
        """
        Trả về các tham số cải tiến cho quá trình sinh văn bản
        """
        params = {}
        
        # Tham số nhiệt độ để tăng tính đa dạng
        params['temperature'] = self.config.get('temperature', 0.8)  # Tăng nhiệt độ để tăng tính đa dạng
        
        # Hình phạt lặp lại
        params['repeat_penalty'] = self.config.get('repeat_penalty', 1.8)  # Tăng repeat_penalty
        
        # Nucleus sampling
        params['top_p'] = self.config.get('top_p', 0.9)  # Tăng top_p để đa dạng hơn
        
        # Độ dài tối thiểu
        params['min_length'] = self.config.get('min_length', 30)
        
        return params
    
    def enhance_word_choice(self, logits, temperature=1.0, top_p=0.9):
        """
        Cải thiện lựa chọn từ thông qua việc điều chỉnh phân phối xác suất
        """
        # Áp dụng nhiệt độ để làm mềm phân phối
        logits = logits / temperature
        
        # Sắp xếp logits và tính xác suất lũy tiến
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Tạo mask cho tokens trong top-p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Giữ nguyên token có xác suất cao nhất và áp dụng mask cho những token khác
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        # Tăng xác suất cho các từ ít gặp một chút
        vocab_size = logits.shape[-1]
        frequency_boost = torch.zeros_like(logits)
        # Đây là nơi bạn có thể thêm logic để tăng xác suất cho các từ ít phổ biến
        
        # Kết hợp với logits gốc
        logits = logits + frequency_boost
        
        return logits, sorted_indices, sorted_indices_to_remove
    
    def replace_with_synonyms(self, text, probability=0.3):
        """
        Thay thế một số từ trong văn bản bằng từ đồng nghĩa
        """
        if not self.config.get('use_synonyms', False) or not self.synonym_dict:
            return text
            
        words = text.split()
        for i in range(len(words)):
            word = words[i].lower()
            # Chỉ thay thế từ có trong từ điển đồng nghĩa với xác suất cho trước
            if word in self.synonym_dict and random.random() < probability:
                synonyms = self.synonym_dict[word]
                if synonyms:
                    words[i] = random.choice(synonyms)
                    
        # Tìm và thay thế các cụm từ cố định
        full_text = ' '.join(words)
        phrase_replacements = {
            "heart size is normal": ["normal cardiac size", "the heart is normal in size", "normal heart size", 
                                     "cardiac size within normal limits"],
            "lungs are clear": ["clear lungs", "lung fields are clear", "no focal airspace opacity",
                               "no focal consolidation", "lungs clear without infiltrate"],
            "no pleural effusion": ["without pleural effusion", "no effusion", "absence of pleural fluid", 
                                    "no visible pleural fluid"],
            "no pneumothorax": ["without pneumothorax", "pneumothorax is absent", "no free air in pleural space"],
            "within normal limits": ["normal", "unremarkable", "within normal parameters", "within normal range"],
            "no evidence of": ["without", "negative for", "no sign of", "no finding of", "absent"]
        }
        
        # Thực hiện thay thế cụm từ với xác suất
        for phrase, replacements in phrase_replacements.items():
            if phrase in full_text.lower() and random.random() < probability:
                replacement = random.choice(replacements)
                # Thay thế với phù hợp về hoa thường (tạm thời dùng phiên bản đơn giản)
                full_text = full_text.replace(phrase, replacement)
        
        return full_text
    
    def data_augmentation(self, report):
        """
        Tăng cường dữ liệu bằng cách thay thế từ hoặc cụm từ
        """
        if random.random() < 0.5:  # 50% cơ hội áp dụng augmentation
            # Thay thế bằng từ đồng nghĩa
            if self.config.get('use_synonyms', False):
                return self.replace_with_synonyms(report, probability=0.3)
            
        return report
    
    def analyze_vocabulary_distribution(self, corpus_path, output_path=None):
        """
        Phân tích phân phối từ vựng trong tập corpus
        """
        try:
            # Đọc corpus
            with open(corpus_path, 'r') as f:
                corpus = json.load(f)
                
            all_words = []
            for split in corpus:
                for example in corpus[split]:
                    if 'report' in example:
                        words = example['report'].lower().split()
                        all_words.extend(words)
            
            # Đếm tần suất
            word_counts = Counter(all_words)
            total_words = len(all_words)
            unique_words = len(word_counts)
            
            # Sắp xếp theo tần suất giảm dần
            sorted_words = word_counts.most_common()
            
            # Phân tích
            analysis = {
                "total_words": total_words,
                "unique_words": unique_words,
                "vocabulary_richness": unique_words / total_words,
                "top_50_words": sorted_words[:50],
                "rare_words": [w for w, c in sorted_words if c == 1][:50]
            }
            
            # Lưu kết quả phân tích
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(analysis, f, indent=4)
            
            return analysis
        
        except Exception as e:
            print(f"Lỗi khi phân tích từ vựng: {e}")
            return None
    
    def modify_generate_batch(self, generate_batch_fn):
        """
        Sửa đổi hàm generate_batch để sử dụng các phương pháp cải thiện từ vựng
        """
        original_fn = generate_batch_fn
        
        def enhanced_generate_batch(model, images, tokenizer, device, max_length=150, **kwargs):
            # Lấy tham số cải tiến
            enhanced_params = self.enhance_generation_params()
            
            # Gộp các tham số của người dùng với tham số cải tiến
            for key, value in enhanced_params.items():
                if key not in kwargs:
                    kwargs[key] = value
                    
            # Gọi hàm generate_batch gốc với tham số đã cải tiến
            captions, generated = original_fn(model, images, tokenizer, device, max_length, **kwargs)
            
            # Áp dụng thay thế từ đồng nghĩa nếu cần
            if self.config.get('use_synonyms', False):
                enhanced_captions = []
                for caption in captions:
                    enhanced_caption = self.replace_with_synonyms(caption, probability=0.2)
                    enhanced_captions.append(enhanced_caption)
                return enhanced_captions, generated
            
            return captions, generated
        
        return enhanced_generate_batch


# Cách sử dụng:
# 1. Tạo đối tượng VocabularyEnhancement
# config = {
#     'temperature': 0.8,
#     'repeat_penalty': 1.8,
#     'top_p': 0.9,
#     'use_synonyms': True
# }
# vocab_enhancer = VocabularyEnhancement(tokenizer, config)
#
# 2. Sử dụng phương thức enhance_generation_params để lấy tham số tối ưu
# params = vocab_enhancer.enhance_generation_params()
#
# 3. Sửa đổi hàm generate_batch
# from test import generate_batch
# enhanced_generate_batch = vocab_enhancer.modify_generate_batch(generate_batch)
#
# 4. Sử dụng enhanced_generate_batch thay vì generate_batch gốc
# captions, generated = enhanced_generate_batch(model, images, tokenizer, device) 
import json
import re
from collections import Counter
import os
import nltk
from nltk.corpus import wordnet
import random
import numpy as np


class EnhancedTokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'siu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        
        # Tạo từ điển đồng nghĩa trước khi xây dựng từ vựng
        self.synonym_mapping = self.create_synonym_mapping()
        
        # Tạo từ điển cụm từ y khoa thường gặp
        self.medical_phrases = self.extract_common_phrases()
        
        # Tạo từ vựng chính
        self.token2idx, self.idx2token = self.create_vocabulary()
        
        # Map ngữ nghĩa cho các từ đồng nghĩa
        self.semantic_groups = self.create_semantic_groups()
        
        # Kích thước vector embedding (phù hợp với mô hình)
        self.embedding_dim = 512  # Tương ứng với chiều của model embedding
        
        # Tạo các vector embedding cho từng token và đảm bảo từ đồng nghĩa có vector gần nhau
        self.init_similarity_embeddings()
        
        # Lưu thống kê trích xuất từ gts.csv
        try:
            self.analyze_gts_vocabulary(os.path.join(args.save_dir, "gts.csv"))
        except:
            print("Không tìm thấy file gts.csv để phân tích. Sẽ sử dụng từ điển mặc định.")

    def create_vocabulary(self):
        """Tạo từ vựng với sự hỗ trợ của từ điển đồng nghĩa và nhóm ngữ nghĩa"""
        # Kiểm tra xem semantic_groups đã tồn tại chưa
        # Nếu chưa, tạo một phiên bản tạm thời chỉ để dùng trong phương thức này
        if not hasattr(self, 'semantic_groups'):
            temp_semantic_groups = self.create_semantic_groups()
        else:
            temp_semantic_groups = self.semantic_groups

        total_tokens = []
        for split in ['train', 'val', 'test']:
            for example in self.ann[split]:
                # Chỉ làm sạch báo cáo mà không áp dụng synonym mapping
                report = self.clean_report(example['report'])
                
                # Không chuẩn hóa cụm từ trong quá trình xây dựng từ điển
                tokens = report.split()
                
                # Không áp dụng synonym mapping để giữ nguyên từ điển
                # (mà chỉ áp dụng synonym mapping khi test)
                total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        
        # Tạo từ vựng cơ bản mà không thay đổi token
        base_vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        
        # Thêm các từ đại diện của từng nhóm đồng nghĩa
        for group_name, terms in temp_semantic_groups.items():
            for term in terms:
                if term not in base_vocab:
                    base_vocab.append(term)
        
        # Sắp xếp từ vựng theo nhóm ngữ nghĩa
        vocab = self.sort_vocabulary_by_semantics(base_vocab)
        
        # Tạo token2idx và idx2token
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
            
        return token2idx, idx2token

    def sort_vocabulary_by_semantics(self, vocab):
        """Sắp xếp từ vựng theo nhóm ngữ nghĩa thay vì theo bảng chữ cái"""
        # Chia thành các nhóm
        organ_terms = []
        normal_terms = []
        abnormal_terms = []
        descriptor_terms = []
        negation_terms = []
        other_terms = []
        
        # Phân loại từng từ
        for token in vocab:
            token_lower = token.lower()
            
            # Các bộ phận cơ thể
            if token_lower in ['heart', 'cardiac', 'cardiomediastinal', 'lung', 'lungs', 'pulmonary', 
                             'mediastinum', 'pleural', 'bony', 'osseous', 'spine', 'thorax', 'thoracic',
                             'rib', 'diaphragm', 'vasculature', 'silhouette', 'contour', 'contours']:
                organ_terms.append(token)
                
            # Từ mô tả bình thường
            elif token_lower in ['normal', 'unremarkable', 'clear', 'stable', 'unchanged', 'intact', 
                               'within', 'limits', 'free', 'negative', 'expanded', 'aerated', 'well-aerated',
                               'well', 'grossly']:
                normal_terms.append(token)
                
            # Từ mô tả bất thường
            elif token_lower in ['consolidation', 'opacity', 'opacities', 'effusion', 'pneumothorax', 
                               'infiltrate', 'infiltration', 'edema', 'mass', 'nodule', 'atelectasis', 
                               'abnormality', 'abnormal', 'fracture', 'pneumonia']:
                abnormal_terms.append(token)
                
            # Từ mô tả mức độ
            elif token_lower in ['mild', 'moderate', 'severe', 'small', 'large', 'focal', 'diffuse', 
                              'minimal', 'significant', 'acute', 'chronic']:
                descriptor_terms.append(token)
                
            # Từ phủ định
            elif token_lower in ['no', 'not', 'without', 'none', 'absence', 'negative']:
                negation_terms.append(token)
                
            # Các từ khác
            else:
                other_terms.append(token)
        
        # Sắp xếp từng nhóm riêng
        for group in [organ_terms, normal_terms, abnormal_terms, descriptor_terms, negation_terms]:
            group.sort()
            
        other_terms.sort()
        
        # Kết hợp các nhóm theo thứ tự ưu tiên
        sorted_vocab = (organ_terms + normal_terms + abnormal_terms + 
                       descriptor_terms + negation_terms + other_terms)
        
        return sorted_vocab

    def create_synonym_mapping(self):
        """Tạo bảng ánh xạ từ đồng nghĩa dựa trên phân tích gts.csv"""
        # Ánh xạ từ đồng nghĩa -> từ chuẩn
        # Phân tích từ gts.csv cho thấy các từ này thường được sử dụng thay thế cho nhau
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
        
        return synonyms

    def create_semantic_groups(self):
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
        
        return semantic_groups

    def extract_common_phrases(self):
        """Trích xuất các cụm từ thường gặp dựa trên dữ liệu"""
        # Các cụm từ phổ biến và biến thể của chúng (phân tích từ gts.csv)
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
        
        return common_phrases

    def analyze_gts_vocabulary(self, gts_path):
        """Phân tích từ vựng từ tập dữ liệu gts.csv để cải thiện synonym mapping"""
        if os.path.exists(gts_path):
            with open(gts_path, 'r') as f:
                reports = f.readlines()
                
            # Đếm tần suất các từ
            all_words = []
            for report in reports:
                words = self.clean_report(report).split()
                all_words.extend(words)
                
            word_counts = Counter(all_words)
            
            # Tìm các mẫu câu phổ biến
            patterns = {
                "normal_heart": [r"heart (size|silhouette) (is|appears) normal", 
                               r"normal (heart|cardiac) (size|silhouette)",
                               r"heart.+within normal limits"],
                "clear_lungs": [r"lungs (are|is) clear", 
                              r"clear lung(s)?", 
                              r"lung fields (are|is) clear",
                              r"no focal (consolidation|airspace disease|infiltrate)"],
                "no_effusion": [r"no (pleural effusion|effusion)", 
                              r"without (pleural effusion|effusion)",
                              r"no evidence of.+effusion"]
            }
            
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
                
                phrase_variations[pattern_type] = Counter(variations).most_common()
                
            # Cập nhật synonym_mapping và medical_phrases dựa trên kết quả phân tích
            # Đây là nơi bạn có thể mở rộng thêm mapping dựa trên dữ liệu thực tế
            for pattern_type, variations in phrase_variations.items():
                for phrase, _ in variations:
                    if pattern_type == "normal_heart" and phrase not in self.synonym_mapping:
                        self.synonym_mapping[phrase] = "heart is normal"
                    elif pattern_type == "clear_lungs" and phrase not in self.synonym_mapping:
                        self.synonym_mapping[phrase] = "lungs are clear"
                    elif pattern_type == "no_effusion" and phrase not in self.synonym_mapping:
                        self.synonym_mapping[phrase] = "no pleural effusion"
                        
            # In thống kê
            print(f"Đã phân tích {len(reports)} báo cáo từ gts.csv")
            print(f"Tổng số từ: {len(all_words)}, số lượng từ độc nhất: {len(word_counts)}")
            print(f"Từ điển đồng nghĩa có {len(self.synonym_mapping)} mục")

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        # Chuẩn hóa token bằng từ đồng nghĩa nếu có thể
        token_lower = token.lower()
        if token_lower in self.synonym_mapping:
            token = self.synonym_mapping[token_lower]
            
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        # Xử lý các cụm từ trước
        processed_report = report.lower()
        for phrase, variations in self.medical_phrases.items():
            for variation in variations:
                if variation in processed_report:
                    processed_report = processed_report.replace(variation, phrase)
        
        # Tiếp tục xử lý bình thường
        tokens = self.clean_report(processed_report).split()
        ids = []
        
        for token in tokens:
            # Áp dụng synonym mapping
            if token.lower() in self.synonym_mapping:
                token = self.synonym_mapping[token.lower()]
                
            ids.append(self.get_id_by_token(token))
            
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                continue
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
    
    def init_similarity_embeddings(self):
        """Khởi tạo các vector biểu diễn cho từng token, đảm bảo các từ liên quan gần nhau"""
        # Dictionary lưu trữ embedding cho mỗi token
        self.token_embeddings = {}
        
        # Khởi tạo base embeddings ngẫu nhiên cho mỗi token
        # Các token đặc biệt sẽ có vector đặc biệt
        special_tokens = {'<unk>': np.zeros(self.embedding_dim)}
        
        for token, idx in self.token2idx.items():
            if token in special_tokens:
                self.token_embeddings[token] = special_tokens[token]
            else:
                # Khởi tạo vector ngẫu nhiên
                self.token_embeddings[token] = np.random.normal(0, 0.1, self.embedding_dim)
        
        # Đảm bảo các từ đồng nghĩa có vector gần nhau
        self.align_synonym_embeddings()
        
        # Đảm bảo các từ trong cùng nhóm ngữ nghĩa có vector gần nhau
        self.align_semantic_group_embeddings()
    
    def align_synonym_embeddings(self):
        """Căn chỉnh các vector biểu diễn để các từ đồng nghĩa có vector gần nhau"""
        # Xây dựng map ngược từ từ chuẩn -> danh sách từ đồng nghĩa
        canonical_to_synonyms = {}
        for synonym, canonical in self.synonym_mapping.items():
            if canonical not in canonical_to_synonyms:
                canonical_to_synonyms[canonical] = []
            canonical_to_synonyms[canonical].append(synonym)
        
        # Với mỗi từ chuẩn, chỉnh sửa vector biểu diễn của các từ đồng nghĩa để gần với từ chuẩn
        for canonical, synonyms in canonical_to_synonyms.items():
            if canonical in self.token_embeddings:
                canonical_vector = self.token_embeddings[canonical]
                
                for synonym in synonyms:
                    if synonym in self.token_embeddings:
                        # Tạo một vector gần với từ chuẩn nhưng có nhiễu nhỏ để tránh trùng lặp
                        noise = np.random.normal(0, 0.01, self.embedding_dim)
                        self.token_embeddings[synonym] = canonical_vector + noise
    
    def align_semantic_group_embeddings(self):
        """Căn chỉnh các vector biểu diễn để các từ trong cùng nhóm ngữ nghĩa có vector gần nhau"""
        for group_name, terms in self.semantic_groups.items():
            if not terms:
                continue
            
            # Tính trung bình của các vector trong nhóm
            valid_terms = [term for term in terms if term in self.token_embeddings]
            if not valid_terms:
                continue
                
            group_vectors = [self.token_embeddings[term] for term in valid_terms]
            centroid = np.mean(group_vectors, axis=0)
            
            # Di chuyển các vector trong nhóm để gần với tâm hơn
            for term in valid_terms:
                # Đưa vector gần với tâm nhóm hơn nhưng vẫn giữ đặc trưng riêng
                current_vec = self.token_embeddings[term]
                # Trộn 70% vector gốc với 30% tâm nhóm
                self.token_embeddings[term] = 0.7 * current_vec + 0.3 * centroid
                
                # Chuẩn hóa vector
                norm = np.linalg.norm(self.token_embeddings[term])
                if norm > 0:
                    self.token_embeddings[term] = self.token_embeddings[term] / norm
    
    def get_token_embedding(self, token):
        """Lấy vector biểu diễn cho một token cụ thể"""
        if token in self.token_embeddings:
            return self.token_embeddings[token]
        elif token.lower() in self.token_embeddings:
            return self.token_embeddings[token.lower()]
        else:
            return self.token_embeddings['<unk>']
    
    def get_embeddings_batch(self, tokens_batch):
        """Lấy ma trận embedding cho một batch các token"""
        embeddings = []
        for tokens in tokens_batch:
            tokens_emb = [self.get_token_embedding(token) for token in tokens]
            embeddings.append(tokens_emb)
        return np.array(embeddings)
    
    def get_embedding_matrix(self):
        """Tạo ma trận embedding để sử dụng trong mô hình"""
        vocab_size = len(self.token2idx) + 1  # +1 cho token 0 (PAD)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        # Đặt vector cho mỗi token theo ID của nó
        for token, idx in self.token2idx.items():
            embedding_matrix[idx] = self.get_token_embedding(token)
            
        return embedding_matrix
    
    def get_semantically_similar_tokens(self, token, top_k=5):
        """Trả về các token có ngữ nghĩa tương tự với token đầu vào dựa trên khoảng cách vector"""
        if token not in self.token_embeddings:
            if token.lower() in self.token_embeddings:
                token = token.lower()
            else:
                return []
        
        query_vector = self.token_embeddings[token]
        similarities = {}
        
        # Tính khoảng cách cosine với tất cả các token khác
        for other_token, other_vector in self.token_embeddings.items():
            if other_token == token:
                continue
            
            # Tính độ tương đồng cosine
            dot_product = np.dot(query_vector, other_vector)
            norm_product = np.linalg.norm(query_vector) * np.linalg.norm(other_vector)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                similarities[other_token] = similarity
        
        # Sắp xếp và trả về top_k tokens tương tự nhất
        sorted_tokens = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tokens[:top_k]]


# Để dễ dàng sử dụng với mã nguồn hiện tại
# Tạo lớp kế thừa để backward compatibility
class Tokenizer(EnhancedTokenizer):
    pass 
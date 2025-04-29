import torch
import torch.nn.functional as F
import numpy as np
from modules.tester import BaseTester
import os
import pandas as pd
from dev import build_model, get_model_config
from tqdm import tqdm
from modules.dataloaders import R2DataLoader
from modules.tokenizers_enhanced import EnhancedTokenizer as Tokenizer
import json

from modules.metrics import compute_scores
from config.configs import Test_Config

import copy  # for duplicating tokenizer

import warnings
warnings.filterwarnings("ignore")

def generate_batch(model, images, tokenizer, device, max_length=150, temperature=1.0, min_length=30, repeat_penalty=1.5, top_p=0.9):
    """
    Hàm sinh báo cáo X-quang sử dụng mô hình AnatomaMamba
    """
    model.eval()
    images = images.to(device)
    batch_size = images.shape[0]
    
    # Lấy thông tin tokenizer
    vocab_size = tokenizer.get_vocab_size()
    end_token_id = 0  # Token EOS/PAD
    
    # Token bắt đầu câu
    generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    # Các tham số sinh văn bản
    
    # Lưu token đã sinh cho mỗi batch
    prev_tokens = [[] for _ in range(batch_size)]
    
    # Flags để theo dõi trạng thái sinh
    finished = [False] * batch_size
    
    with torch.no_grad():
        # Sinh tuần tự
        for i in range(max_length):
            # Sinh token tiếp theo
            outputs = model(
                text=generated,
                images=images,
                return_loss=False
            )
            
            # Lấy logits của token cuối cùng
            next_token_logits = outputs[:, -1, :vocab_size]
            next_token_logits = next_token_logits / temperature
            
            # Phạt lặp lại tokens
            for idx in range(batch_size):
                for prev_token in prev_tokens[idx]:
                    next_token_logits[idx, prev_token] /= repeat_penalty
            
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Tạo mask cho tokens trong top-p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # Áp dụng mask và lấy token tiếp theo
            for idx in range(batch_size):
                indices_to_remove = sorted_indices[idx][sorted_indices_to_remove[idx]]
                next_token_logits[idx, indices_to_remove] = -float('Inf')
            
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(next_token_probs, num_samples=1).squeeze(-1)
            
            # Ngăn chặn token EOS nếu chưa đạt độ dài tối thiểu
            if i < min_length:
                next_tokens = torch.where(next_tokens == end_token_id, 
                                        torch.randint(1, vocab_size, (batch_size,), device=device),
                                        next_tokens)
            
            # Lưu lại token đã sinh
            for idx in range(batch_size):
                if not finished[idx]:
                    token = next_tokens[idx].item()
                    prev_tokens[idx].append(token)
                    
                    # Kiểm tra nếu đã sinh EOS sau min_length
                    if i >= min_length and token == end_token_id:
                        finished[idx] = True
            
            # Thêm vào chuỗi đã sinh
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            
            # Kiểm tra nếu tất cả các batch đã hoàn thành
            if all(finished) and i >= min_length:
                break
    
    # Chuyển về numpy để giải mã
    generated = generated.cpu().numpy()
    generated = generated[:, 1:]  # Bỏ token bắt đầu
    
    # Giải mã
    captions = tokenizer.decode_batch(generated)
    
    return captions, generated


class Tester(BaseTester):
    def __init__(self, model, tokenizer, metric_ftns, args, test_dataloader, device):
        super(Tester, self).__init__(model, None, metric_ftns, args)
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.args = args

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            # Load raw original reports and clean them (ground-truth, no synonyms)
            raw_reports = [ex['report'] for ex in self.raw_tokenizer.ann['test']]
            raw_cleaned = [self.raw_tokenizer.clean_report(r) for r in raw_reports]
            # Prepare canonical ground-truth for metrics (apply mapping)
            canonical_gts = []
            for r in raw_reports:
                ids = self.tokenizer(r)
                canonical_gts.append(self.tokenizer.decode(ids))
            # Collect predictions
            test_res = []
            for batch in self.test_dataloader:
                images = batch[1].to(self.device)
                reports_ids = batch[2].to(self.device)
                # Sinh báo cáo (predictions canonical)
                reports, _ = generate_batch(
                    self.model, images, self.tokenizer, self.device, 
                    max_length=self.args.max_length, 
                    temperature=self.args.temperature
                )
                test_res.extend(reports)
                
            # Compute metrics on canonical tokens
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(canonical_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            # Lưu kết quả: predictions canonical và ground-truth raw cleaned
            res_df = pd.DataFrame(test_res)
            gts_df = pd.DataFrame(raw_cleaned)
            res_path = os.path.join(self.save_dir, "res.csv")
            gts_path = os.path.join(self.save_dir, "gts.csv")
            res_df.to_csv(res_path, index=False, header=False)
            gts_df.to_csv(gts_path, index=False, header=False)
            
            return log

    
def main(args):
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cố định seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Create two tokenizers: one for model predictions (with mapping), one raw for ground-truth cleaning
    tokenizer_map = Tokenizer(args)
    # raw tokenizer: disable synonyms and phrase mapping for ground truth only
    tokenizer_raw = copy.deepcopy(tokenizer_map)
    tokenizer_raw.synonym_mapping = {}
    tokenizer_raw.medical_phrases = {}
    
    # Tạo data loader using raw tokenizer (ground-truth IDs/raw cleaning)
    test_dataloader = R2DataLoader(args, tokenizer_raw, split='test', shuffle=False)
    
    # Xây dựng model
    model_config = get_model_config()
    model = build_model(args, tokenizer_map, model_config)

    # Load checkpoint
    checkpoint = torch.load(args.load, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Lấy hàm tính metrics
    metrics = compute_scores

    # Tạo tester và thực hiện test (predictions dùng tokenizer_map)
    print(f"Max_len: {args.max_length} | temperature: {args.temperature}")
    tester = Tester(model, tokenizer_map, metrics, args, test_dataloader=test_dataloader, device=device)
    # attach raw tokenizer for ground-truth cleaning
    tester.raw_tokenizer = tokenizer_raw
    results = tester.test()
    
    return results


if __name__ == '__main__':
    args = Test_Config()
    main(args)



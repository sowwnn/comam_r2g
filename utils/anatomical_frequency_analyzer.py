import json
import re
from collections import Counter
import os

def analyze_keyword_frequency(keyword_groups_path, annotations_path, output_path=None):
    """
    Phân tích tần suất xuất hiện của các nhóm từ khóa giải phẫu trong dữ liệu annotations.
    
    Args:
        keyword_groups_path (str): Đường dẫn đến file chứa các nhóm từ khóa giải phẫu
        annotations_path (str): Đường dẫn đến file annotations.json chứa dữ liệu báo cáo
        output_path (str, optional): Đường dẫn để lưu kết quả phân tích. Mặc định là None.
        
    Returns:
        dict: Kết quả phân tích tần suất của các nhóm từ khóa
    """
    # Đọc file từ khóa nhóm giải phẫu
    with open(keyword_groups_path, 'r') as f:
        keyword_groups = json.load(f)
    
    # Đọc file annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Tạo từ điển để đếm tần suất xuất hiện của mỗi nhóm
    group_frequency = {group: 0 for group in keyword_groups.keys()}
    report_counts = 0
    
    # Kết quả cho từng phần (train, test, val)
    split_results = {}
    
    # Tiền xử lý các từ khóa để tìm kiếm hiệu quả hơn
    for group in keyword_groups:
        # Sắp xếp các từ khóa theo độ dài giảm dần để khớp với các từ dài trước
        keyword_groups[group] = sorted(keyword_groups[group], key=len, reverse=True)
    
    # Xử lý từng phần (train, test, val)
    for split in annotations.keys():
        # Khởi tạo bộ đếm cho mỗi phần
        split_frequency = {group: 0 for group in keyword_groups.keys()}
        split_count = 0
        
        # Nếu là dict với key là id của báo cáo
        if isinstance(annotations[split], dict):
            items = annotations[split].items()
        # Nếu là list các báo cáo
        elif isinstance(annotations[split], list):
            items = [(item.get('id', i), item) for i, item in enumerate(annotations[split])]
        else:
            print(f"Không xác định được định dạng của phần {split}")
            continue
        
        # Xử lý từng báo cáo trong phần
        for report_id, report_data in items:
            split_count += 1
            report_counts += 1
            
            # Lấy nội dung của "report"
            if isinstance(report_data, dict) and 'report' in report_data:
                report_text = report_data['report'].lower()
            else:
                # Trường hợp khác, chuyển đổi sang string và lowercase
                report_text = str(report_data).lower() if report_data else ""
            
            # Tạo set để theo dõi các nhóm đã xuất hiện trong báo cáo này
            groups_in_report = set()
            
            # Kiểm tra sự xuất hiện của từng từ khóa trong mỗi nhóm
            for group, keywords in keyword_groups.items():
                for keyword in keywords:
                    # Tìm từ khóa dưới dạng từ hoàn chỉnh (sử dụng ranh giới từ)
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    if re.search(pattern, report_text):
                        groups_in_report.add(group)
                        break  # Nếu tìm thấy một từ khóa trong nhóm, đã đếm nhóm này
            
            # Cập nhật số lần xuất hiện cho mỗi nhóm đã tìm thấy
            for group in groups_in_report:
                split_frequency[group] += 1
                group_frequency[group] += 1
        
        # Tính phần trăm xuất hiện cho mỗi phần
        split_percentage = {group: (count / split_count * 100) if split_count > 0 else 0 
                           for group, count in split_frequency.items()}
        
        # Lưu kết quả của phần
        split_results[split] = {
            "total_reports": split_count,
            "absolute_frequency": dict(sorted(split_frequency.items(), key=lambda x: x[1], reverse=True)),
            "percentage": dict(sorted(split_percentage.items(), key=lambda x: x[1], reverse=True))
        }
    
    # Tính phần trăm xuất hiện tổng thể
    percentage = {group: (count / report_counts * 100) if report_counts > 0 else 0 
                 for group, count in group_frequency.items()}
    
    # Tạo kết quả cuối cùng
    result = {
        "overall": {
            "total_reports": report_counts,
            "absolute_frequency": dict(sorted(group_frequency.items(), key=lambda x: x[1], reverse=True)),
            "percentage": dict(sorted(percentage.items(), key=lambda x: x[1], reverse=True))
        },
        "splits": split_results
    }
    
    # Lưu kết quả vào file nếu có đường dẫn
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
    
    return result

if __name__ == "__main__":
    # Đường dẫn tới các file
    keyword_groups_path = os.path.join(os.path.dirname(__file__), "anatomical_keyword_groups.json")
    annotations_path = "/home/sowwn/Workspace/ws/2025/Dataset/MIMIC-CXR/mimic_cxr/annotations.json"
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "anatomical_frequency_results.json")
    
    # Đảm bảo thư mục results tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Chạy phân tích
    result = analyze_keyword_frequency(keyword_groups_path, annotations_path, output_path)
    
    # In kết quả tổng thể
    overall = result["overall"]
    print(f"Đã phân tích tổng cộng {overall['total_reports']} báo cáo")
    print("\nTần suất xuất hiện tổng thể của các nhóm từ khóa giải phẫu:")
    for group, count in overall['absolute_frequency'].items():
        print(f"{group}: {count} lần ({overall['percentage'][group]:.2f}%)")
    
    # In kết quả cho từng phần
    print("\nPhân tích theo từng phần:")
    for split, data in result["splits"].items():
        print(f"\n--- {split.upper()} ({data['total_reports']} báo cáo) ---")
        for group, count in list(data['absolute_frequency'].items())[:5]:
            print(f"{group}: {count} lần ({data['percentage'][group]:.2f}%)")
        if len(data['absolute_frequency']) > 5:
            print("...")

def main(annotations_path=None, output_path=None):
    """
    Hàm chính để chạy từ bên ngoài module
    
    Args:
        annotations_path (str, optional): Đường dẫn tới file annotations.json
        output_path (str, optional): Đường dẫn để lưu kết quả
        
    Returns:
        dict: Kết quả phân tích
    """
    # Nếu không có đường dẫn được cung cấp, sử dụng đường dẫn mặc định
    if not annotations_path:
        annotations_path = "/home/sowwn/Workspace/ws/2025/Dataset/MIMIC-CXR/mimic_cxr/annotations.json"
    
    if not output_path:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "anatomical_frequency_results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Đường dẫn tới file từ khóa
    keyword_groups_path = os.path.join(os.path.dirname(__file__), "anatomical_keyword_groups.json")
    
    # Chạy phân tích
    return analyze_keyword_frequency(keyword_groups_path, annotations_path, output_path) 
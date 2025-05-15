class Test_Config():
    # image_dir = "/mnt/e/Sho/iu_xray/images"
    # ann_path = "/mnt/e/Sho/iu_xray/annotation.json"
    # # ann_path = "logs/uixray_ov_annotations.json"
    # dataset_name = "siu_xray"
    # max_seq_length = 60
    # threshold = 1
    # max_length = 34
    # temperature= 0.25

    image_dir = "/mnt/e/Sho/mimic_cxr/images"
    ann_path = "/mnt/e/Sho/mimic_cxr/annotations.json"
    # ann_path = "/mnt/e/Sho/mimic_cxr/aug_annotation.json"
    dataset_name = "mimic_cxr"
    max_seq_length = 100
    threshold = 1
    max_length = 64
    temperature = 0.35

    batch_size = 64
    num_workers = 12
    learning_rate = 1e-4
    num_epochs = 30
    seed = 5401

    
    save_dir = 'weights/mimic/temp'
    n_gpu = 1
    load = "weights/mimic/con_mam_v3_gau/best_model.pth"
    # use_threshold_model = True





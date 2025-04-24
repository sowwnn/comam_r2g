class Test_Config():
    image_dir = "/mnt/e/Sho/iu_xray/images"
    ann_path = "/mnt/e/Sho/iu_xray/annotation.json"
    # ann_path = "logs/uixray_ov_annotations.json"
    dataset_name = "siu_xray"
    max_seq_length = 60
    threshold = 1
    max_length = 32
    temperature= 0.3

    # image_dir = "/mnt/e/Sho/mimic_cxr/images"
    # ann_path = "/mnt/e/Sho/mimic_cxr/annotations.json"
    # # ann_path = "/mnt/e/Sho/mimic_cxr/aug_annotation.json"
    # dataset_name = "mimic_cxr"
    # max_seq_length = 180
    # threshold = 1
    # max_length = 150
    # temperature = 0.9

    batch_size = 128
    num_workers = 12
    transform = None
    learning_rate = 1e-4
    num_epochs = 100
    seed = 5401

    
    save_dir = 'weights/iu_xray/con_mam_v3'
    n_gpu = 1
    load = "weights/iu_xray/con_mam_v3/best_model.pth"
    # use_threshold_model = True

    enable_dynamic_loss = True
    repeat_penalty = 1.5
    top_p = 0.5





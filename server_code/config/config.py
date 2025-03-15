class Config:
    Mysql_host = "localhost"
    Mysql_username = "PCB"
    Mysql_password = "020522"
    Mysql_db = "PCB_Detection_DB"
    project_path = "E:\code\Graduation_project\module\yolov7_PCB_Server\server_code"
    Image_path = f"{project_path}\dataset\images"
    Image_detect_path = f"{Image_path}\detected"
    temp_Image_path = f"{Image_path}\\temp"
    origin_Image_path = f"{Image_path}\origins"
    Text_path = f"{project_path}\dataset\labels"
    model_path = "runs/train/yolo-tiny-pcb-500/weights/best.pt"
    Coockie = {
        "cookie_name": "session",
        "identifier": "general_verifier",
        "auto_error": True,
        "secret_key": "MMDCCJBYSJFWD",  # 毕业设计服务端
    }
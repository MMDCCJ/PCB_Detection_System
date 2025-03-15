#PCB缺陷检测系统
基于YOLOv7优化的YOLOv7_PCB模型
##设置与初始化
修改`Server_code/Config.py`文件，初始化`SQL`请运行`Server_code/sql/start.sql`
###参数详情
```python
    Mysql_host = "localhost" # 数据库Host
    Mysql_username = "PCB" # 用户名
    Mysql_password = "020522" # 密码
    Mysql_db = "PCB_Detection_DB" # 所选数据库名
    project_path = "E:\code\Graduation_project\module\yolov7_PCB_Server\server_code" # 项目server_code位置
    Image_path = f"{project_path}\dataset\images" # 图片存储位置
    Image_detect_path = f"{Image_path}\detected" # 预测图片存储位置
    temp_Image_path = f"{Image_path}\\temp" # 临时图片存储位置
    origin_Image_path = f"{Image_path}\origins" # 预测后原始图片存储位置
    Text_path = f"{project_path}\dataset\labels" # 标签位置
    model_path = "runs/train/yolo-tiny-pcb-500/weights/best.pt" # 所选择模型
```
##运行
Step 1 启动Redis服务

step 2 启动Python服务器
```python
uvicorn server:app --reload"
```
step 3 启动Vue服务(开发模式)
```shell
npm run dev
```

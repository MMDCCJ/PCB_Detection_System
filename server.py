import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import ast
from numpy import random
import shutil
import datetime
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import uuid
from server_code.config import config
from server_code.py.PCB import PCB 
#server 
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
# class
from server_code.py.User import User
from server_code.py.Message import Message
#database
import redis
from server_code.py.DBService import DBService
from datetime import timedelta
import asyncio
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
# redis 管理session配置
cookie_params = CookieParameters()
cookie = SessionCookie(
    cookie_name="session",
    identifier="general_verifier",
    auto_error=True,
    secret_key="MMDCCJBYSJFWD",  # 毕业设计服务端
    cookie_params=cookie_params,
)
# 每五分钟执行一次
async def flush_daily_detecte_num_every_five_minutes():
    while True:
        data = DB.get_daily_detected_data()
        pre_data = DB.get_pre_day_detected_data()
        redis_client.set('daily_detected_data', data[0][0])
        redis_client.set('pre_daily_detected_data', pre_data[0][0])
        await asyncio.sleep(300)  # Sleep for 300 seconds (5 minutes)
def init():
    # 目录创建
    temp_path = config.Config.temp_Image_path
    origin_image_path = config.Config.origin_Image_path
    detected_image_path = config.Config.Image_detect_path
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(origin_image_path, exist_ok=True)
    os.makedirs(detected_image_path, exist_ok=True)
    # 主页数据定时器
    asyncio.create_task(flush_daily_detecte_num_every_five_minutes())
    print("初始化完成")
init()
# 加载模型
need_load = True  # 是否加载模型 
if need_load:
    # 初始化
    set_logging()  # 设置日志
    device = select_device('0')  # 选择设备（CPU或GPU）
    half = device.type != 'cpu'  # 在CUDA上启用半精度浮点数
    # 模型加载
    model = attempt_load('runs/train/yolov7-tiny-pcb-500/weights/best.pt', map_location=device)  # 加载模型权重
    stride = int(model.stride.max())  # 获取模型的步幅
    imgsz = 640  # 图像输入大小
    imgsz = check_img_size(imgsz, s=stride)  # 检查输入大小是否符合模型要求
    if half:
        model.half()  # 模型转换为半精度
    # 推理预热
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 预热模型
# 加载模型结束
def detect(img_src='dataset/images/test',save_img=False):
    source = img_src
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path('runs/detect') / 'yolo_pcb_v7', exist_ok=False))
    (save_dir / 'labels' if True else save_dir).mkdir(parents=True, exist_ok=True) 
    # Initialize
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load('runs/train/yolo_pcb_v7/weights/best.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            print("txt_path is")
            print(txt_path)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if True else (cls, *xywh)  # label format
                    print(('%g ' * len(line)).rstrip() % line + '\n')
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    print("缺陷位置"+('%g ' * len(line)).rstrip() % line + '\n')
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
def detect_single(img_src='dataset/images/test',user_id=0,save_img=True,PCB_Id=0):
    # 数据源设置
    source = img_src  # 输入源（可以是图片路径、视频路径、摄像头等）
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断输入源是否为实时流或摄像头
    # 保存目录创建
    save_path = config.Config.Image_detect_path
    original_path = config.Config.origin_Image_path
    os.makedirs(save_path, exist_ok=True) 
    os.makedirs(original_path, exist_ok=True) 
    os.makedirs(config.Config.Text_path, exist_ok=True)
    save_dir = Path(save_path)  # 增量创建保存目录
    # 数据加载
    vid_path, vid_writer = None, None
    # 视频流
    if webcam:
        view_img = check_imshow()  # 检查是否支持显示图像窗口
        cudnn.benchmark = True  # 提高推理速度（固定大小的输入图像）
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)  # 加载视频流数据
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)  # 加载静态图像数据
    # 获取类别名和颜色
    names = model.module.names if hasattr(model, 'module') else model.names  # 类别名称
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]  # 每个类别分配随机颜色
    # 初始化变量
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    # 遍历数据集
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)  # 转换图像为张量并转移到设备
        img = img.half() if half else img.float()  # 转换为半精度或浮点数
        img /= 255.0  # 归一化到 [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 增加批量维度
        # 模型预热（仅在输入大小变化时）
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]
        # 推理
        t1 = time_synchronized()
        with torch.no_grad():  # 禁用梯度计算，减少显存使用
            pred = model(img, augment=False)[0]  # 获取预测结果
        t2 = time_synchronized()
        # 非极大值抑制（NMS）
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)  # 筛选有效预测
        t3 = time_synchronized()
        # 处理检测结果
        for i, det in enumerate(pred):
            if webcam:  # 如果是实时流
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:  # 如果是静态图像
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # 转换为 Path 对象
            save_path = str(save_dir / p.name)  # 保存路径
            txt_path = str(config.Config.Text_path +"/"+ p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 标签路径
            print("txt_path is", txt_path)
            print("save_path is", save_path)
            print("img_path",path)
            # 检测结果处理
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化坐标比例
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 将预测框坐标缩放回原图尺寸

                # 打印结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别的数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 格式化输出

                # 保存结果
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 转换为 xywh 格式
                    line = (cls, *xywh, conf)  # 保存格式
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    print("缺陷位置：" + ('%g ' * len(line)).rstrip() % line)
                    print('文件名：',p.name)
                    pcb_id = DB.get_pcbId_by_name(p.name)
                    if pcb_id:
                        DB.save_defect(pcb_id,('%g ' * len(line)).rstrip() % line)
                        DB.update_image_state(Image_id=pcb_id,Image_state="存在缺陷") 
                    else:
                        DB.update_image_state(Image_id=pcb_id,Image_state="检测通过") 
                        pass
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            # 输出推理时间
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # 保存检测结果图像或视频
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)  # 保存静态图像
                    shutil.move(path,config.Config.origin_Image_path+"/"+p.name)
                    print(f"The image with the result is saved in: {save_path}")
                else:  # 视频或流
                    if vid_path != save_path:  # 如果是新视频
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)  # 写入帧
app = FastAPI()
# cors配置
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DB = DBService()
# 中间件来验证Web端的登录状态
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    # 跳过特定路径的请求
    # 无需权限的API
    if request.url.path in ["/api/login", "/api/register","/static","/api/check_login"]:
        response = await call_next(request)
        return response
    print(request.url.path,"经过中间件")
    if request.method.lower() == "options" and request.url.path in ["/api/upload"] :
        response = await call_next(request)
        return response
    session_id = request.cookies.get("session")
    print("session_id:",session_id)
    # 需要权限的API 
    if session_id:
        session_data = redis_client.get(f"session:{session_id}")
        if session_data:
            request.state.session = session_data
        else:
            return JSONResponse(status_code=401, content=Message("未登录",400).pack())
    else:
        request.state.session = None
        return JSONResponse(status_code=401, content=Message("未登录",400).pack())
    response = await call_next(request)
    return response
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images/detected",StaticFiles(directory="server_code/dataset/images/detected"), name="images")
app.mount("/images/origins",StaticFiles(directory="server_code/dataset/images/origins"), name="origins")
# 首页
@app.get("/",response_class=HTMLResponse)
async def read_html():
    with open("static/index.html", "r",encoding='utf-8') as file:
        html_content = file.read()
    return html_content

"""
    以下为API接口
"""
# 注册
@app.post("/api/register")
async def test(user:User):
    if DB.check_username_exist(user.username):
        return Message("用户名已存在",400).pack()
    if DB.check_userInfo(user.username,user.password):
        return Message("用户信息非法",400).pack()
    else:
        if DB.register_user(user.username,user.password):
            return Message("注册成功",200).pack()
        else:
            return Message("注册失败",400).pack()  
# 检查用户名是否存在
@app.get("/api/check_username_exist/{username}")
async def check_username_exist(username: str):
    if DB.check_username_exist(username):
        return Message("用户名已存在", 400).pack()
    else:
        return Message("用户名可用", 200).pack()
# 登录
@app.post("/api/login")
async def login(user:User):
    if DB.check_username_exist(user.username):
        if DB.login(user.username,user.password):
            session_id = str(uuid.uuid4())
            session_data = {"username": user.username}
            redis_client.setex(f"session:{session_id}", timedelta(hours=1), str(session_data))
            res = HTMLResponse(content=str(Message("login success",200)), status_code=200)
            res.set_cookie(key="session", value=session_id, httponly=True,samesite='Lax',secure=False)
            return res
        else:
            return Message("密码错误",400).pack()
    else:
        return Message("用户名不存在",400).pack()
# 检查登录状态
@app.get("/api/check_login")
async def check_login(request: Request):
    session_data = redis_client.get(f"session:{request.cookies.get('session')}")
    print("session-data:",session_data)
    if session_data:
        return Message("已登录", 200).pack()
    else:
        return Message("未登录", 400).pack()
# 上传图片
@app.post("/api/upload")
async def upload(request: Request):
    form = await request.form()
    file = form['file']
    session_id = request.cookies.get("session") # 获取session_id
    session_data = redis_client.get(f"session:{session_id}")
    session_data = ast.literal_eval(session_data.decode())
    user_name = session_data['username']
    # 随机生成一个文件名
    file_name = f"{uuid.uuid4()}.jpg"
    # 给文件路径加上时间前缀
    file_name = f"{datetime.date.today()}_{file_name}"
    # 文件路径
    file_path = config.Config.temp_Image_path+f"/{file_name}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    DB.upload_image(user_name,file_name)
    return Message("上传成功",200).pack()

# 预测
@app.get("/api/predict")
async def predict(request: Request):
    session_id = request.cookies.get("session") # 获取session_id
    session_data = redis_client.get(f"session:{session_id}")
    session_data = ast.literal_eval(session_data.decode())
    user_name = session_data['username']
    user_id = DB.get_user_id(user_name)
    unpredict_images_list = DB.get_unpredicted_image_list(user_id)
    if not unpredict_images_list:
        return Message("无待预测图片",400).pack()
    for image in unpredict_images_list:
        print(image)
        detect_single(img_src=config.Config.temp_Image_path+"/"+image[2],save_img=True,PCB_Id=image[0])
    return Message("预测完成",200).pack()
@app.get("/api/get_pcb_list")
async def get_result_list(request: Request):
    session_id = request.cookies.get("session")
    session_data = redis_client.get(f"session:{session_id}")
    session_data = ast.literal_eval(session_data.decode())
    user_name = session_data['username']
    user_Id = DB.get_user_id(user_name)
    print("userid:",user_Id)
    result_list = DB.get_defected_PCB_list_by_userId(user_Id)
    PCB_Info_list = []
    for i, result in enumerate(result_list):
        pcb_id = result[0]
        src_name = result[2]
        date = result[3]
        defect = DB.get_defect_by_pcbId(pcb_id)
        pcb = PCB(id=pcb_id,srcName=src_name,position=defect,date=date)
        PCB_Info_list.append(pcb)
    return Message("获取成功",200,PCB_Info_list).pack()
@app.get("/api/detected_data")
async def get_daily_detected_data(request: Request):
    data = redis_client.get('daily_detected_data')
    pre_data = redis_client.get('pre_daily_detected_data')
    return Message("获取成功",200,{"today":data,"pre_day":pre_data}).pack()
# 空白路由
# Catch-all 路由来处理所有未匹配的 URL 请求，并返回 index.html 页面
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def catch_all(full_path: str):
    with open("static/index.html", "r", encoding='utf-8') as file:
        html_content = file.read()
    return html_content




# 测试检测功能
if __name__ == "__main__":
    # detect()
    detect_single()
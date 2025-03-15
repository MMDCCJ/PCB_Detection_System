from mysql.connector import Error
import mysql.connector
import sys
sys.path.append('E:\code\Graduation_project\module\yolov7_PCB_Server\server_code')
from config.config import Config
import re
class DBService:
    _instance = None
    _connection = None
    # 单例模式
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DBService, cls).__new__(cls, *args, **kwargs)
            cls._connection = cls.create_connection()
        return cls._instance
    # 创建数据库连接
    @staticmethod
    def create_connection():
        """
        创建并返回一个 MySQL 数据库连接。

        :return: MySQL 数据库连接对象
        :rtype: mysql.connector.connection.MySQLConnection
        """
        connection = None
        try:
            connection = mysql.connector.connect(
                host=Config.Mysql_host,
                user=Config.Mysql_username,
                passwd=Config.Mysql_password,
                database=Config.Mysql_db
            )
            print("Connection to MySQL DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")

        return connection
    # 获取数据库连接
    @staticmethod
    def get_connection():
        """
        获取当前的数据库连接对象。

        :return: 当前的数据库连接对象
        :rtype: mysql.connector.connection.MySQLConnection
        """
        return DBService._connection
    # 检查用户名是否存在
    @staticmethod
    def check_username_exist(username):
        """
        检查用户名是否存在于数据库中。

        :param username: 要检查的用户名
        :type username: str
        :return: 如果用户名存在则返回 True,否则返回 False
        :rtype: bool
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM pcb_user WHERE username = '{username}'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return len(records) > 0
    @staticmethod
    # 注册用户
    def register_user(username:str,password:str):
        """
        向数据库中注册一个新用户。

        :param username: 新用户的用户名
        :type username: str
        :param password: 新用户的密码
        :type password: str
        :return: 注册成功则返回 True,否则返回 False
        :rtype: bool
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(f"INSERT INTO pcb_user (username, password,user_type) VALUES ('{username}', '{password}','user')")
            connection.commit()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return True
    # 检查用户信息是否符合要求
    def check_userInfo(self,username:str,password:str):
        """
        检查用户信息是否符合要求。

        :param user: 用户对象
        :type user: User
        :return: 如果用户信息符合要求则返回 True, 否则返回 False
        :rtype: bool
        """
        # 检验长度
        if not (3 < len(username) < 20):
            return False
        if not (6 < len(password) < 20):
            return False
        # 检验是否包含中文
        if not re.search("[\u4e00-\u9FFF]", username):
            return False
        if not re.search("[\u4e00-\u9FFF]", password):
            return False
        return True
    # 登录
    @staticmethod
    def login(username:str,password:str):
        """
        用户登录。

        :param username: 用户名
        :type username: str
        :param password: 密码
        :type password: str
        :return: 登录成功则True,否则返回 False
        :rtype: User
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM pcb_user WHERE username = '{username}' AND password = '{password}'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        if len(records) == 0:
            return False
        return True
    # 上传图片URL至数据库
    @staticmethod
    def upload_image(user_name:str,image_name:str):
        """
        上传图片URL至数据库。

        :param image_name: 图片名
        :type image_name: str
        :return: 上传成功则返回 True,否则返回 False
        :rtype: bool
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        user_id = DBService.get_user_id(user_name)
        if user_id == False:
            return False
        else:
            try:
                cursor.execute(f"INSERT INTO pcb_images (User_id,image_name,Upload_date,Detect_Date,Image_state) VALUES ('{user_id}', '{image_name}',NOW(),NOW(),'未检测')")
                connection.commit()
            except Error as e:
                print(f"The error '{e}' occurred")
                return False
            return True
    # 通过用户名获取用户id
    @staticmethod
    def get_user_id(username:str):
        """
        获取用户id。

        :param username: 用户名
        :type username: str
        :return: 用户id
        :rtype: int
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM pcb_user WHERE username = '{username}'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records[0][0]
    @staticmethod
    def save_defect(pcb_id:str,defect:str):
        """
        保存缺陷。

        :param pcb_id: 图片名
        :type pcb_id: str
        :param defect: 缺陷
        :type defect: str
        :return: 保存成功则返回 True,否则返回 False
        :rtype: bool
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"INSERT INTO pcb_defect (pcb_id,position) VALUES ('{pcb_id}', '{defect}')")
            connection.commit()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return True
    @staticmethod
    def get_pcbId_by_name(image_name:str):
        """
        通过图片名获取pcbId。

        :param image_name: 图片名
        :type image_name: str
        :return: pcbId
        :rtype: int
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM pcb_images WHERE image_name = '{image_name}'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        if len(records) == 0:
            return False
        return records[0][0]
    @staticmethod
    def get_defected_PCB_list_by_userId(user_id:str):
        """
        通过用户id获取pcb列表。

        :param user_id: 用户id
        :type user_id: int
        :return: pcb列表
        :rtype: list
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM pcb_images WHERE User_id = '{user_id}' AND Image_state != '未检测' AND Image_state != '已删除'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    @staticmethod
    def get_defect_by_pcbId(pcb_id:str):
        """
        通过pcbId获取缺陷。

        :param pcb_id: pcbId
        :type pcb_id: int
        :return: 缺陷
        :rtype: list
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT position FROM pcb_defect WHERE pcb_id = '{pcb_id}'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    # 获取用户上传的未预测的图片列表
    @staticmethod
    def get_unpredicted_image_list(user_id:str):
        """
        获取用户上传的未预测的图片列表。

        :param user_id: 用户id
        :type user_id: int
        :return: 未预测的图片列表
        :rtype: list
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM pcb_images WHERE User_id = '{user_id}' AND Image_state = '未检测'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    # 更新图片状态
    @staticmethod
    def update_image_state(Image_id=0,Image_state="NULL",Image_Name="NULL"):
        """
        更新图片状态。

        :param Image_id: 图片id
        :type Image_id: int
        :param Image_state: 图片状态
        :type Image_state: str
        :param Image_Name: 图片名
        :type Image_Name: str
        :return: 更新成功则返回 True,否则返回 False
        :rtype: bool
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            if Image_state!="NULL" and Image_Name!="NULL":
                cursor.execute(f"UPDATE pcb_images SET Image_state = '{Image_state}',Image_name = '{Image_Name}', Detect_Date = NOW() WHERE PCB_id = '{Image_id}'")
            if Image_state=="NULL":
                cursor.execute(f"UPDATE pcb_images SET Image_name = '{Image_Name}', Detect_Date = NOW() WHERE PCB_id = '{Image_id}'")
            elif Image_Name=="NULL":
                cursor.execute(f"UPDATE pcb_images SET Image_state = '{Image_state}', Detect_Date = NOW() WHERE PCB_id = '{Image_id}'")
            elif Image_state=="NULL" and Image_Name=="NULL":
                return False
            connection.commit()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return True
    @staticmethod
    def get_daily_detection_data():
        """
        获取每日检测数据。

        :return: 每日检测数据
        :rtype: list
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM pcb_images WHERE Image_state != '未检测' and DATE(pcb_images.detect_date) = CURDATE()")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    @staticmethod
    def get_pre_day_detection_data():
        """
        获取前一天检测数据。

        :return: 前一天检测数据
        :rtype: list
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM `pcb_images` WHERE DATE(pcb_images.detect_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY);")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    # 删除图片
    @staticmethod
    def delete_pcb(pcb_id):
        """
        置指定的PCB为删除状态。

        :return: 删除结果
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"UPDATE pcb_images SET Image_state = '已删除' WHERE PCB_Id = '{pcb_id}'")
            records = cursor.fetchall()
            connection.commit()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    @staticmethod
    def get_total_detection_data():
        """
        获取总检测数据。

        :return: 总检测数据
        :rtype: list
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM pcb_images WHERE Image_state != '未检测'")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    @staticmethod
    def get_total_detected_data():
        """
        获取总缺陷数据。

        :return: 总缺陷数据
        :rtype: list
        """
        connection = DBService.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM pcb_defect")
            records = cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occurred")
            return False
        return records
    
if __name__ == "__main__":
    # Example usage
    db = DBService()
    if db.check_username_exist("test"):
        print("1用户名存在检验正常")
    if not db.check_userInfo("test","12345"):
        print("2密码长短检验正常")
    if not db.check_userInfo("test男","12345"):
        print("3用户名不包含中文检验正常")
    if not db.check_userInfo("test","男12345"):
        print("4密码不包含中文检验正常")
    test_id= db.get_user_id("test")
    if db.get_user_id("test")==1:
        print(f"5获取用户test,id={test_id}正常")
    if db.get_pcbId_by_name("test.jpg")==1:
        print("6获取pcbId正常")
    PCB_list = db.get_PCB_list_by_userId(1)
    if PCB_list:
        print(PCB_list[0])
        pcbs_list = []
        for i in range(len(PCB_list)):
            pcb = {}
            pcb['pcb_id'] = PCB_list[i][0]
            pcb['user_id'] = PCB_list[i][1]
            pcb['image_name'] = PCB_list[i][2]
            pcb['update_date'] = PCB_list[i][3].strftime('%Y-%m-%d %H:%M:%S')
            pcbs_list.append(pcb)
        print(pcbs_list)
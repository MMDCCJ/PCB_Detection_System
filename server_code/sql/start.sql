DROP DATABASE IF EXISTS PCB_Detection_DB;

CREATE DATABASE PCB_Detection_DB;

USE PCB_Detection_DB;

CREATE USER IF NOT EXISTS 'PCB' @'localhost' IDENTIFIED BY '020522';

GRANT
SELECT
,
INSERT
,
UPDATE
,
    DELETE ON PCB_Detection_DB.* TO 'PCB' @'localhost';

FLUSH PRIVILEGES;

USE PCB_Detection_DB;

-- 用户表
CREATE TABLE IF NOT EXISTS PCB_User (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Username VARCHAR(50) NOT NULL,
    Password VARCHAR(50) NOT NULL,
    User_Type ENUM('admin', 'user') NOT NULL
) COMMENT = '用户信息表';

-- PCB图片表
CREATE TABLE IF NOT EXISTS PCB_Images (
    PCB_Id INT AUTO_INCREMENT PRIMARY KEY,
    User_Id INT NOT NULL,
    Image_name VARCHAR(255) not NULL,
    Upload_Date DATETIME NOT NULL,
    Detect_Date DATETIME NOT NULL,
    Image_state ENUM('未检测', '存在缺陷', '检测通过') NOT NULL,
    CONSTRAINT fk_UserId FOREIGN KEY (User_Id) REFERENCES PCB_User(id)
) COMMENT = 'PCB图像上传记录表';

-- PCB缺陷表
CREATE TABLE IF NOT EXISTS PCB_Defect (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    PCB_ID INT NOT NULL,
    position VARCHAR(255),
    CONSTRAINT fk_PCBID FOREIGN KEY (PCB_ID) REFERENCES PCB_Images(PCB_Id)
) COMMENT = 'PCB缺陷表';

INSERT INTO
    PCB_User (Username, Password, User_Type)
VALUES
    ('Test', '123456', 'user');

INSERT INTO
    PCB_User (Username, Password, User_Type)
VALUES
    ('MMDCCJ', '020522', 'admin');
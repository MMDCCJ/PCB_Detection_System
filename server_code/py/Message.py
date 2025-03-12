import json
class Message:
    def __init__(self, message: str, code: int, data=None):
        self.message = message
        self.data = data
        self.code = code
    """
    将 Message 对象打包
    :return: Message 对象的 JSON 字符串表示
    :rtype: str
    """
    def pack(self):
        return {
            "message": self.message,
            "code": self.code,
            "data": self.data
        }
    def __str__(self):
        return json.dumps({
            "message": self.message,
            "code": self.code,
            "data": self.data
        })
if __name__ == "__main__":
    # Example usage
    msg = Message("register success", 200, {"user_id": 123})
    print(str(msg))  # 输出: {"message": "register success", "code": 200, "data": {"user_id": 123}}
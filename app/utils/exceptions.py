"""全局异常定义"""


class UserStopped(Exception):
    """用户主动停止实验时抛出，由 worker 统一处理（不计为崩溃）"""

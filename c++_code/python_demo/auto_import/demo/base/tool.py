'''
Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
Date: 2022-11-04 16:12:26
LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
LastEditTime: 2022-11-04 16:17:35
FilePath: \c++_code\python_demo\base\tool.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


class Tool(object):
    """
    Base class for Sub-Tools.
    """

    def __init__(self, name=None, subtools=[]):
        self.name = name
        self.subtools = subtools
        self.parser = None

    @staticmethod
    def run():
        print("xiaoying")
        # raise NotImplementedError("run() must be implemented by child classes")

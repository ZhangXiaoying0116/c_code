'''
Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
Date: 2022-11-21 18:56:12
LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
LastEditTime: 2022-11-21 19:00:16
FilePath: \c++_code\python_demo\track\track_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import traceback


def test1():
    1/0


def test2():
    test1()


if __name__ == "__main__":
    try:
        test2()
    except Exception as ex:
        # traceback.print_exc()
        traceback.print_stack()

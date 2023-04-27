'''
Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
Date: 2022-11-04 16:13:41
LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
LastEditTime: 2022-11-04 17:14:15
FilePath: \c++_code\python_demo\demo\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from onnx_model import OnnxTool
from base.tool import Tool

MainTool = Tool('main', subtools=[OnnxTool])
print(MainTool.run())

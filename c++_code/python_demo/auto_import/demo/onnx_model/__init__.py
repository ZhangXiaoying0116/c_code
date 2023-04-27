'''
Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
Date: 2022-11-04 14:54:23
LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
LastEditTime: 2022-11-04 17:39:19
FilePath: \c++_code\python_demo\tool\__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
from base.tool import Tool
import onnx_model

module_object1 = __import__("onnx_model.batchsize", fromlist=[
                            "batchsize"])  # 将模块加载为对象
print(module_object1)
module_class1 = getattr(module_object1, "Batchsize")  # 获取模块当中的类对象


# module_object2 = __import__("onnx_model.extract")  # 将模块加载为对象 wrong
module_object2 = __import__("onnx_model.extract", fromlist=[
                            "extract"])
print(module_object2)
module_class2 = getattr(module_object2, "Extract")  # 获取模块当中的类对象
instance1 = module_class1()
instance2 = module_class2()
print(type(instance1))
print(type(instance2))


OnnxTool = Tool('onnx', subtools=[instance1, instance2])

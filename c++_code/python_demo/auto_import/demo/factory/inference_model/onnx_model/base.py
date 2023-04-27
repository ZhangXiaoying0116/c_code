'''
Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
Date: 2022-11-16 17:29:15
LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
LastEditTime: 2022-11-16 17:29:23
FilePath: \c++_code\python_demo\auto_import\demo\factory\inference_model\onnx_model\base.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from model_factory import ModelFactory


class OnnxModelFactory(ModelFactory):
    name = 'onnx'
    model = 'undefined'

'''
Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
Date: 2022-11-16 17:27:30
LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
LastEditTime: 2022-11-17 13:50:10
FilePath: \c++_code\python_demo\auto_import\demo\factory\inference_model\model_factory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


class BaseModelFactory(type):
    _framework_name = "name"
    _model_name = "model"
    _develop_stage = "stage"
    _registered_map = {}
    _model_names = []
    _develop_map = {"done": [], "beta": [], "alpha": []}
    _abstractmethods = set(["new_model"])

    def __new__(mcs, *args, **kwargs):
        cls = super(BaseModelFactory, mcs).__new__(mcs, *args, **kwargs)

        if cls.__name__ == "ModelFactory" or cls.model == "undefined":
            return cls
        print("------cls.model:", cls.model)
        mcs.__register_new_model(cls)
        return cls

    @classmethod
    def __register_new_model(mcs, cls):
        pass


class ModelFactory(metaclass=BaseModelFactory):
    pass

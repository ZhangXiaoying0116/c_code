'''
Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
Date: 2022-11-16 16:05:33
LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
LastEditTime: 2022-11-16 16:05:47
FilePath: \c++_code\python_demo\auto_import\demo\factory\简单工厂模式.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEDog
'''
from abc import ABCMeta, abstractmethod


class Animal(metaclass=ABCMeta):
    @abstractmethod
    def do_say(self):
        pass


class Dog(Animal):
    def do_say(self):
        print("Bhow Bhow")


class Cat(Animal):
    def do_say(self):
        print("Meow Meow")


class ForestFactory:
    def make_sound(self, object_type):
        return eval(object_type)().do_say()


if __name__ == '__main__':
    ff = ForestFactory()
    animal = input("Which animal should make sound, Dog or Cat\n")
    ff.make_sound(animal)

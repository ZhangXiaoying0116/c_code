
#include <iostream>
#include"string.h"
using namespace std;


class ClassA
{
public:
    ClassA(double *p, int iAge)
    {
        m_chName = p;
        m_iAge = iAge;
        std::cout<<"Constructor .... "<<m_chName[0]<<"\n";
    }
    ClassA(const ClassA &obj)
    {
        cout << "调用深度拷贝构造函数并为指针 ptr 分配内存" << endl;
        m_iAge = obj.m_iAge;
        m_chName = (double *)malloc(sizeof(double));
        memcpy(m_chName, obj.m_chName, sizeof(double)* 5);
        std::cout<<"deep copy Constructor .... "<<m_chName[0]<<"\n";
    }

public:
    double *m_chName;  //名字
    int m_iAge;  //年龄
};

int main()
{
    double runoobAarray[5] = {1000.0, 2.0, 3.4, 17.0, 50.0};
    ClassA A(runoobAarray, 18);  //定义A对象

    std::cout<<"1-copy .... \n";
    double runoobAarrayB[5] = {1111.0, 22222.0, 3.4, 17.0, 50.0};
    ClassA B(runoobAarrayB, 9);  //定义A对象
    B = A;  // 赋值拷贝A对象，产生对象B
    std::cout<<"New B .... "<<B.m_chName[0]<<"\n";
    std::cout<<"Old A .... "<<A.m_chName[0]<<"\n";
    B.m_chName[0] = 1234;
    std::cout<<"2New B .... "<<B.m_chName[0]<<"\n";
    std::cout<<"2Old A .... "<<A.m_chName[0]<<"\n";

    std::cout<<"2-deepcopy .... \n";
    ClassA C(A);  // 拷贝构造A对象，产生对象C; ClassA C=A; 默认浅拷贝，重写成深拷贝
    std::cout<<"New C .... "<<C.m_chName[0]<<"\n";
    std::cout<<"Old A .... "<<A.m_chName[0]<<"\n";
    C.m_chName[0] = 222222;
    std::cout<<"2New C .... "<<C.m_chName[0]<<"\n";
    std::cout<<"2Old A .... "<<A.m_chName[0]<<"\n";

    return 0;
}

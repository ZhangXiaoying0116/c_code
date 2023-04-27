
#include <iostream>
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

public:
    double *m_chName;  //名字
    int m_iAge;  //年龄
};


int main()
{
    double runoobAarray[5] = {1000.0, 2.0, 3.4, 17.0, 50.0};
    ClassA A(runoobAarray, 18);  //定义A对象
    ClassA B = A;  //拷贝A对象，产生对象B
    std::cout<<"New B .... "<<B.m_chName[0]<<"\n";
    std::cout<<"Old A .... "<<A.m_chName[0]<<"\n";
    B.m_chName[0] = 1234;
    std::cout<<"2New B .... "<<B.m_chName[0]<<"\n";
    std::cout<<"2Old A .... "<<A.m_chName[0]<<"\n";
    return 0;
}

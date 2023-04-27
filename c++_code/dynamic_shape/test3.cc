/*
 * @Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @Date: 2022-10-14 16:15:29
 * @LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @LastEditTime: 2022-10-14 16:20:17
 * @FilePath: \c++_code\dynamic_shape\test3.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <map>
#include <iostream>
using namespace std;

void func(map<int, int> n) {
  cout << " 最初,numbers.empty(): " << n.empty() << "\n";
}

int main() {
  map<int, int> numbers;
  func(numbers);
  std::string str = "";
  if (str.empty()) {
    cout << " str.empty(): " << str.empty() << "\n";
  }

  //   numbers[1] = 100;
  //   numbers[2] = 200;
  //   numbers[3] = 300;
  //   cout << "\n 添加元素后,number.empty(): " << numbers.empty() << "\n";
}
/*
 * @Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @Date: 2022-10-18 15:11:40
 * @LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @LastEditTime: 2022-10-18 15:14:03
 * @FilePath: \c++_code\dynamic_shape\test5.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
using namespace std;
int main() {
  int res2[10] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  int res2_dest[6] = {0};
  for (int i = 0; i < 6; i++) {
    cout << res2_dest[i] << endl;
  }
  cout << "------------------------" << endl;
  memcpy(res2_dest, res2, 6 * sizeof(int));
  for (int i = 0; i < 6; i++) {
    cout << res2_dest[i] << endl;
  }
}

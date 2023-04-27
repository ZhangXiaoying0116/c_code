/*
 * @Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @Date: 2022-10-16 23:06:35
 * @LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @LastEditTime: 2022-10-17 02:43:53
 * @FilePath: \c++_code\dynamic_shape\test4.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
#include <json/json.h>
using namespace std;

int main() {
  Json::Value op_max_val;
  Json::Value op_min_val;

  std::vector<string> max_shape_dim = {"1,1024,1024,64", "1,512,512,128",
                                       "1,256,256,256", "1,128,128,512"};
  std::vector<string> min_shape_dim = {"1,16,16,64", "1,8,8,128", "1,4,4,256",
                                       "1,2,2,512"};
  // 遍历列表长度
  for (size_t i = 0; i < max_shape_dim.size(); i++) {
    // set max shape
    op_max_val["main"].append(max_shape_dim[i]);
    // set min shape
    op_min_val["main"].append(min_shape_dim[i]);
  }
  Json::Value max_shape_range_setting;
  max_shape_range_setting.append(op_max_val);
  std::string max_setting_str = max_shape_range_setting.toStyledString();

  Json::Value min_shape_range_setting;
  min_shape_range_setting.append(op_min_val);
  std::string min_setting_str = min_shape_range_setting.toStyledString();
}

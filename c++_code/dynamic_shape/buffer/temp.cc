/*
 * @Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @Date: 2022-10-26 13:47:58
 * @LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @LastEditTime: 2022-10-26 13:52:52
 * @FilePath: \c++_code\dynamic_shape\buffer\temp.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

// 输出大小开辟
// ===================================================================
// AAA：开辟一段新的buffer(用于存放输出的最大buffer)
// Create and populate Tensor
TensorShape shape(ort_shape);
// std::unique_ptr<Tensor> tp(new Tensor(DataTypeImpl::GetType<float>(), shape,
// allocator));
// Tensor* tp = new Tensor(DataTypeImpl::GetType<float>(), shape, allocator);
// void *output_data = tp->MutableData<float>();
void *output_data = cpu_allocator->Alloc(1 * 64 * 64 * 64 * 4);

// *tp->MutableData<std::string>() = input.Get<ExperimentalType>().str_;
// exit(0);
// // ===================================================================

// OrtValue *output_tensor = ort.KernelContext_GetOutput(
//     context, i, ort_shape.data(), ort_shape.size());
// void *output_data = ort.GetTensorMutableData<void>(output_tensor);

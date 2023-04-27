/*
 * @Author: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @Date: 2022-10-27 10:38:44
 * @LastEditors: ZhangXiaoying0116 zhangxiaoying0116@gmail.com
 * @LastEditTime: 2022-10-27 10:42:59
 * @FilePath: \c++_code\dynamic_shape\buffer\temp2.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
context reset output shape == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == == == == == == == ==

std::vector<int64_t> ort_shape2{1, 64, 64, 64};
std::for_each(ort_shape2.begin(), ort_shape2.end(),
              [logger = GetLogger()](int64_t n) {
                LOGS(*logger, VERBOSE) << "- dim value: " << n;
              });
OrtValue *output_tensor2 = ort.KernelContext_GetOutput(
    context, 0, ort_shape2.data(), ort_shape2.size());
OrtTensorTypeAndShapeInfo *output_info =
    ort.GetTensorTypeAndShape(output_tensor2);

int64_t size = ort.GetTensorShapeElementCount(output_info);
const auto &tensor_shapes = ort.GetTensorShape(output_info);
const auto &tensor_type = ort.GetTensorElementType(output_info);
ort.ReleaseTensorTypeAndShapeInfo(output_info);
ort.SetDimensions(output_info, ort_shape3.data(), ort_shape3.size());

std::vector<int64_t> new_dims = {1, 64, 56, 56};
onnxruntime::TensorShape new_shape(new_dims.data(), 4);
tensor.Reshape(new_dims);
size_t newtensor_size = tensor.SizeInBytes();
auto &new_outputshape = tensor.Shape();
OrtTensorTypeAndShapeInfo *output_info3 =
    ort.GetTensorTypeAndShape(output_tensor2);
int64_t size3 = ort.GetTensorShapeElementCount(output_info3);

*((float *)outputs[0]) = 1.11111;

auto *reuse_tensor = ort_value_reuse.GetMutable<Tensor>();
auto buffer_num_elements = reuse_tensor -> Shape().Size();
auto required_num_elements = shape.Size();
int aaaa = ort.GetOutputArgIndex(0);

void *output_data = ort.GetTensorMutableData<void>(output_tensor2);
*/
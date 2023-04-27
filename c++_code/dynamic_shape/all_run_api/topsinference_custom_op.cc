// =============================================================================
//
// Copyright 2022 The Enflame-tech Company.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "topsinference_custom_op.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <fstream>
#include <sstream>

namespace onnxruntime {
namespace topsinference_ep {

void DumpOnnxModelProto(const ONNX_NAMESPACE::ModelProto &model_proto,
                        std::string file_name);
// Dump onnx model proto to file
void DumpOnnxModelProto(const ONNX_NAMESPACE::ModelProto &model_proto,
                        std::string file_name) {
  std::fstream outfile(file_name,
                       std::ios::out | std::ios::trunc | std::ios::binary);
  std::string onnx_string_buffer;
  model_proto.SerializeToString(&onnx_string_buffer);
  outfile << onnx_string_buffer;
}

TopsInferenceCustomOp::TopsInferenceCustomOp(
    const ComputeContext *context, const onnxruntime::Node *fused_node,
    const std::string &output_names, const std::string &backend_type,
    const std::string &export_executable, const std::string &load_executable,
    const logging::Logger *logger)
    : export_executable_(export_executable) {
  SetLogger(logger);

  LOGS(*logger_, INFO) << "Current chip generation: " << backend_type;

  ORT_UNUSED_PARAMETER(context);

  if (load_executable.empty()) {
    LOGS(*logger_, WARNING)
        << "No executable to load. Generate from onnx model";
    LOGS(*logger_, INFO)
        << "Create IParser only in construct function of TopsInferenceCustomOp";

    error_manager_ = TopsInference::create_error_manager();

    model_path_ = fused_node->ModelPath().ToPathString();
    LOGS(*logger_, INFO) << "model path: " << model_path_;

    parser_.reset(TopsInference::create_parser(
        TopsInference::ParserType::TIF_ONNX, error_manager_));

    if (!output_names.empty()) {
      LOGS(*logger_, INFO) << "Output names not empty. Assigned output names: "
                           << output_names;
      parser_->setOutputNames(output_names.c_str());
    }

    // TopsInference::release_parser(parser_.get());

  } else {
    LOGS(*logger_, WARNING) << "Load executable from external source: "
                            << export_executable_;
    LOGS(*logger_, INFO) << "Create IEngine and load executable in "
                            "construct function of TopsInferenceCustomOp";
    engine_.reset(TopsInference::create_engine());
    engine_->loadExecutable(load_executable.c_str());
  }
}

TopsInferenceCustomOp::~TopsInferenceCustomOp() {
  if (telemetry_.total_runs > 1) {
    double average_latency =
        telemetry_.total_run_latency / (telemetry_.total_runs - 1);
    double average_throughput =
        telemetry_.total_run_throughput / (telemetry_.total_runs - 1);

    if (int8_enable_) {
      LOGS(*logger_, WARNING) << "Run in [TOPSINFERENCE EP] INT8 mode";
    } else if (fp16_enable_) {
      LOGS(*logger_, WARNING) << "Run in [TOPSINFERENCE EP] FP16 mode";
    } else {
      LOGS(*logger_, WARNING) << "Run in [TOPSINFERENCE EP] FP32 mode";
    }

    LOGS(*logger_, WARNING) << "========================";
    LOGS(*logger_, WARNING) << "Total " << telemetry_.total_runs - 1
                            << " queries over " << telemetry_.total_run_latency
                            << "s (excluding the first query)";
    LOGS(*logger_, WARNING) << "Average latency on "
                            << telemetry_.total_runs - 1
                            << " runs: " << average_latency;
    LOGS(*logger_, WARNING) << "Average throughput on "
                            << telemetry_.total_runs - 1
                            << " runs: " << average_throughput;
    LOGS(*logger_, WARNING) << "========================";
  }

  if (!export_executable_.empty()) {
    // Executable could only be saved during deconstructed
    engine_->saveExecutable(export_executable_.c_str());
  }
}

Status
TopsInferenceCustomOp::Compute(const OrtApi *api, OrtKernelContext *context,
                               const std::vector<std::string> &input_names,
                               const std::vector<int> &has_dynamic_batch,
                               const int compiled_batchsize) {
  // ===============================================================
  // 根据用户输入的最大最小的形状确定动态维度，如[1,-1,-1,3]
  // TODO :
  // 如果模型中读取的输入形状不是动态的是否会有问题，has_dynamic_batch是通过模型文件判断的
  // TODO :
  // 最大最小算出的维度[1,-1,-1,3]是否要跟通过模型文件读到的输入[1,w,h,3]做比较，判断动态维度是否对应

  // 是否存在动态形状
  bool has_dynamic_shape = false;

  // 假设从用户侧拿到:
  // 列表 max_shape_dim = ["1,1024,1024,64", "1,512,512,128", "1,256,256,256",
  // "1,128,128,512"]
  // 列表 min_shape_dim = ["1,16,16,64", "1,8,8,128", "1,4,4,256", "1,2,2,512"]
  // TODO : 是否对应input names给定
  // max_shape_dim，min_shape_dim；还是所有输入都需要给定（目前方法）；
  // TODO : 给定所有输入name，要跟input_names做一个排序对齐绑定
  std::vector<string> test_input_names = {"133", "360", "587", "814"};
  std::vector<string> max_shape_dim = {"1,1024,1024,64", "1,512,512,128",
                                       "1,256,256,256", "1,128,128,512"};
  std::vector<string> min_shape_dim = {"1,16,16,64", "1,8,8,128", "1,4,4,256",
                                       "1,2,2,512"};

  // 判断是否存在动态形状。has_dynamic_shape =
  // True的条件是，用户设置了min_shape和max_shape,
  // 同时除了第一维度以外其他维度不相等
  // TODO: 合理性检查-->判断设置是否合理，max应大于min的形状;
  // TODO: 要求bs维度是相同的，目前不支持bs维度动态
  // TODO: 如果第一维度不是bs的情况？所有输入里面第一维度不相等的
  std::vector<string> test_input_shape_strings;
  // test_input_name_strings
  for (size_t i = 0; i < test_input_names.size(); i++) {
    std::vector<string> max_vetStr = splitStr(max_shape_dim[i], ",")
        std::vector<string> min_vetStr = splitStr(min_shape_dim[i], ",")
        // TODO : max_vetStr[0] == min_vetStr[0]
        std::string str = max_vetStr[0];
    for (size_t i = 1; i < vetStr.size(); i++) {
      if (max_shape_dim[i] - min_shape_dim[i])
        > 0 {
          str += ",-1";
          has_dynamic_shape = True;
        }
      elif(max_shape_dim[i] == min_shape_dim[i]) { str += max_shape_dim[i]; }
      else {
        exit(0);
      }
    }
    str += ":" test_input_shape_strings += str
  }

  if (has_dynamic_shape) {
    Json::Value op_max_val;
    Json::Value op_min_val;
    // 遍历列表长度
    for (size_t i = 0; i < test_input_names.size(); i++) {
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
  } else {
    // ===============================================================
    LOGS(*logger_, INFO) << "TopsInference Custom Op Compute";

    Ort::CustomOpApi ort{*api};

    std::string input_name_strings = "";
    std::string input_shape_strings = "";

    size_t data_batchsize = 0;
    bool use_legacy = false;

    /*
      When only one input
        set compiled-batch in input_shape_str to parser
        if compiled-batch equal to real data batch
          use legacy run according to performance loss in TopsInference
        else
          use run_with_batch for parallelism execution
      For two or more inputs
        if all inputs with same value at dimension 0 (assume batch)
          same as one input case
        if shape[0] differs from inputs
          set data-batch to parser
          use legacy run to execute serially
    */

    // concat strings for parser
    for (size_t i = 0; i < input_names.size(); i++) {
      auto tensor_shape = ort.GetTensorShape(
          ort.GetTensorTypeAndShape(ort.KernelContext_GetInput(context, i)));
      // Always assume 1st input holds the semantics of batch size at 1st
      // dimension.
      if (i == 0) {
        data_batchsize = (size_t)tensor_shape[0];
      } else {
        if (data_batchsize != (size_t)tensor_shape[0]) {
          LOGS(*logger_, WARNING)
              << "Values of Dimension 0 are not same for all "
                 "inputs. Serial Execution instead as unsupported in Engine";
          use_legacy = true;
        }
      }
    }
    for (size_t i = 0; i < input_names.size(); i++) {
      auto tensor_shape = ort.GetTensorShape(
          ort.GetTensorTypeAndShape(ort.KernelContext_GetInput(context, i)));
      std::string str = "";

      if (has_dynamic_batch[i]) {
        LOGS(*logger_, INFO)
            << "input: " << input_names[i]
            << " has static batch size: " << has_dynamic_batch[i];
        str += std::to_string(has_dynamic_batch[i]);
      } else {
        if (use_legacy) {
          LOGS(*logger_, INFO)
              << "input: " << input_names[i]
              << " has dynamic batch size, set read-data batchsize: "
              << (size_t)tensor_shape[0] << " in compile-time.";
          str += std::to_string((size_t)tensor_shape[0]);
        } else {
          LOGS(*logger_, INFO)
              << "input: " << input_names[i]
              << " has dynamic batch size, set compiled batchsize: "
              << compiled_batchsize << " in compile-time.";
          str += std::to_string(compiled_batchsize);
        }
      }

      for (size_t index = 1; index < tensor_shape.size(); index++) {
        str += ',' + std::to_string(tensor_shape[index]);
      }

      str += ':';
      input_shape_strings += str;

      input_name_strings += input_names[i] + ',';
    }

    if (input_name_strings.length() > 0) {
      input_name_strings =
          input_name_strings.substr(0, input_name_strings.length() - 1);
      LOGS(*logger_, INFO) << "Input Name Strings: " << input_name_strings;
    } else {
      LOGS(*logger_, WARNING) << "Input Name Strings is NULL. ";
    }

    if (input_shape_strings.length() > 0) {
      input_shape_strings =
          input_shape_strings.substr(0, input_shape_strings.length() - 1);
      LOGS(*logger_, INFO) << "Input Shape Strings: " << input_shape_strings;
    } else {
      LOGS(*logger_, WARNING) << "Input Shape Strings is NULL. ";
    }

    static std::string input_shapes_str = "";
    bool tensor_shape_changed = false;
    if (input_shapes_str == "") {
      LOGS(*logger_, WARNING) << "Trigger Compile Progress";
      // Assign input shapes and passed to TopsInference Parser
      input_shapes_str = input_shape_strings;
      tensor_shape_changed = true;
    } else if (input_shapes_str != input_shape_strings) {
      LOGS(*logger_, WARNING)
          << "Trigger Recompile Progress. Input shapes: " << input_shape_strings
          << " is different from last time: " << input_shapes_str;
      // Reassign input shapes as the string has changed
      input_shapes_str = input_shape_strings;
      tensor_shape_changed = true;
    }
  }

  /* Only when engine_ is null or tensor_shape_changed and parser_ is not null.
   * No need to recompile if IEngine has been created in previous steps.
   * Functions:
   *   Set input shapes for IParser
   *   Create INetwork by parser
   *   Create IOptimizer and set BuildFlag
   *   Optimize INetwork and produce IEngine
   */

  if (parser_ && (!engine_ || tensor_shape_changed)) {
    parser_->setInputNames(input_name_strings.c_str());
    parser_->setInputShapes(input_shapes_str.c_str());

    const Env &env_instance = Env::Default();

    bool enable_graph_builder = false;
    const std::string enable_graph_builder_env = env_instance.GetEnvironmentVar(
        onnxruntime::topsinference_env_vars::kEnableGraphBuilder);
    if (!enable_graph_builder_env.empty()) {
      enable_graph_builder =
          (std::stoi(enable_graph_builder_env) == 0 ? false : true);
    }

    if (enable_graph_builder) {
      parser_->getConfig()->enableGraphBuilder();
    }

    TopsInference::INetwork *network = parser_->readModel(model_path_.c_str());
    // network->dump();

    TopsInference::IOptimizer *optimizer = TopsInference::create_optimizer();

    // Set precision flags
    const std::string fp16_enable_env = env_instance.GetEnvironmentVar(
        onnxruntime::topsinference_env_vars::kFP16Enable);
    if (!fp16_enable_env.empty()) {
      fp16_enable_ = (std::stoi(fp16_enable_env) == 0 ? false : true);
    }

    const std::string int8_enable_env = env_instance.GetEnvironmentVar(
        onnxruntime::topsinference_env_vars::kINT8Enable);
    if (!int8_enable_env.empty()) {
      int8_enable_ = (std::stoi(int8_enable_env) == 0 ? false : true);
    }

    if (int8_enable_ && fp16_enable_) {
      ORT_ENFORCE(false,
                  "Environment variables ORT_TOPSINFERENCE_FP16_ENABLE and "
                  "ORT_TOPSINFERENCE_INT8_ENABLE cannot be set to 1 at the "
                  "same time");
    } else if (int8_enable_) {
      ORT_NOT_IMPLEMENTED("Not implemented [TOPSINFERENCE EP] INT8 mode");
    } else if (fp16_enable_) {
      optimizer->getConfig()->setBuildFlag(
          TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16);
    }

    // 如果含有动态形状，设置最大最小的config
    if (has_dynamic_shape) {
      ASSERT_TRUE(optimizer->getConfig()->setMaxShapeRange(
                      max_setting_str.c_str()) == true)
          << "[Error] set max shape range failed!";
      ASSERT_TRUE(optimizer->getConfig()->setMinShapeRange(
                      min_setting_str.c_str()) == true)
          << "[Error] set min shape range failed!";
      TopsInference::IEngine *engine = optimizer->build(network);
    }

    engine_.reset(optimizer->build(network));

    TopsInference::release_optimizer(optimizer);
    TopsInference::release_network(network);
  }

  const int num_inputs = engine_->getInputNum();
  const int num_outputs = engine_->getOutputNum();

  void *inputs[num_inputs];
  void *outputs[num_outputs];

  // set input ITensor
  std::vector<TopsInference::ITensor_t> input_tensor_list;

  for (int i = 0; i < num_inputs; ++i) {
    const OrtValue *input_tensor = ort.KernelContext_GetInput(context, i);
    auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    // auto tensor_type = ort.GetTensorElementType(tensor_info);
    auto ort_shape = ort.GetTensorShape(tensor_info);
    // std::for_each(ort_shape.begin(), ort_shape.end(), [](int64_t n){std::cout
    // << "dim: " << n << std::endl;});

    std::vector<ssize_t> tensor_shape{ort_shape.begin(), ort_shape.end()};

    void *input_data =
        const_cast<void *>(ort.GetTensorData<void>(input_tensor));

    inputs[i] = (void *)input_data;

    // set input ITensor
    if (has_dynamic_shape) {
      TopsInference::ITensor *sub_input = TopsInference::create_tensor();
      sub_input->setOpaque(
          reinterpret_cast<void *>(inputs[i])); // you should set Opaque to
                                                // transfer input data into
                                                // TopsInference
      sub_input->setDeviceType(
          TopsInference::DataDeviceType::HOST); // you should set DeviceType
      TopsInference::Dims input_shape = engine_->getInputShape(i);

      int counter = 0;
      std::for_each(ort_shape.begin(), ort_shape.end(), [&counter](int64_t n) {
        std::cout << "dim: " << n << std::endl;
        input_shape[counter] = n;
        counter++;
      });

      sub_input->setDims(input_shape); // you should set Dims for transfer
                                       // input shape information into
                                       // TopsInference
      input_tensor_list.emplace_back((TopsInference::ITensor_t)sub_input);
    }
  }

  // If any output has the same value at 0th dimension as that in the
  // first output, we assume that the specific output has semantics of
  // batchsize. Seemingly unreasonable assumptions here as the functional
  // limitations and design flaws of the engine framework.
  if (has_dynamic_shape) {
    // TODO : oxxruntime 是否对于 output要预分配空间大小
    // 那么对于动态性来说预分配的是max shape的最大空间大小

    // set output ITensors
    std::vector<std::vector<float>> outs;
    std::vector<TopsInference::ITensor_t> output_tensor_list;
    outs.resize(num_outputs);
    for (int32_t i = 0; i < num_outputs; ++i) {
      // set output ITensor
      TopsInference::ITensor *sub_output = TopsInference::create_tensor();
      TopsInference::Dims max_shape = engine_->getMaxOutputShape(i);
      int64_t out_element_size = 1;
      // max_shape.dimension[0] = 1;  //
      // batch维度是固定的，直接拿取输出中最大的bs，最大bs==最小bs
      for (int32_t j = 0; j < max_shape.nbDims; ++j) {
        out_element_size *= max_shape.dimension[j]; // sample_num*H*W*C
      }
      outs[i].resize(out_element_size);
      sub_output->setOpaque(reinterpret_cast<void *>(outs[i].data()));
      sub_output->setDims(
          max_shape); // you should use maxoutputshape to set Dims
      sub_output->setDeviceType(
          TopsInference::DataDeviceType::HOST); // you should set DeviceType
      output_tensor_list.emplace_back((TopsInference::ITensor_t)sub_output);
    }

    // TODO:需要对onnxruntime开辟内存去存取结果,
    // 在运行结束后需要对结果按照输出大小重新写回
    // TODO:推断在onnxruntime开辟最大的内存之后结果会自动写入，最后会自动根据实际的输出获取结果
    // TODO:如果返回的是最大的结果形状，可能需要其他方法处理
  } else {
    int32_t bs_in_engine =
        num_outputs > 0 ? engine_->getOutputShape(0).dimension[0] : 0;
    for (int i = 0; i < num_outputs; ++i) {
      auto output_dims = engine_->getOutputShape(i);
      std::vector<ssize_t> output_shape;
      if (use_legacy) {
        output_shape.push_back((ssize_t)output_dims.dimension[0]);
      } else {
        if (output_dims.dimension[0] == bs_in_engine) {
          LOGS(*logger_, INFO) << "Output " << i
                               << " has semantic of batchsize";
          output_shape.push_back((ssize_t)data_batchsize);
        } else {
          output_shape.push_back((ssize_t)output_dims.dimension[0]);
        }
      }

      for (int d = 1; d < output_dims.nbDims; ++d) {
        output_shape.push_back((ssize_t)output_dims.dimension[d]);
      }

      std::vector<int64_t> ort_shape{output_shape.begin(), output_shape.end()};
      LOGS(*logger_, VERBOSE) << "Output " << i << "'s shape: ";
      std::for_each(ort_shape.begin(), ort_shape.end(),
                    [logger = GetLogger()](int64_t n) {
                      LOGS(*logger, VERBOSE) << "- dim value: " << n;
                    });

      OrtValue *output_tensor = ort.KernelContext_GetOutput(
          context, i, ort_shape.data(), ort_shape.size());
      void *output_data = ort.GetTensorMutableData<void>(output_tensor);

      outputs[i] = output_data;
    }
  }

  // 如果含有动态形状，根据输入大小重新bind 动态engine的shape
  if (has_dynamic_shape) {
  }

  auto beforeTime = std::chrono::steady_clock::now();
  // 调用runV2的接口
  if (has_dynamic_shape) {
    // runV2
    engine_->runV2(input_tensor_list.data(), output_tensor_list.data());
    // get real output data and release resources
    for (int32_t i = 0; i < output_num; ++i) {
      int64_t real_element_num = 1;
      TopsInference::Dims ouput_shape =
          output_tensor_list[i]->getDims(); // you should get real output Dims
      for (int32_t index = 0; index < ouput_shape.nbDims; ++index) {
        real_element_num *= ouput_shape.dimension[index];
      }
      outs[i].resize(real_element_num);
    }
  } else {
    if (use_legacy) {
      engine_->run(reinterpret_cast<void **>(inputs),
                   reinterpret_cast<void **>(outputs),
                   TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
    } else {
      engine_->run_with_batch(
          data_batchsize, reinterpret_cast<void **>(inputs),
          reinterpret_cast<void **>(outputs),
          TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
    }
  }

  auto afterTime = std::chrono::steady_clock::now();
  double latency_seconds =
      std::chrono::duration<double>(afterTime - beforeTime).count();

  if (telemetry_.total_runs > 0) {
    telemetry_.total_run_latency += latency_seconds;
    telemetry_.total_run_throughput += (data_batchsize / latency_seconds);
  }
  ++telemetry_.total_runs;

  return Status::OK();
}

} // namespace topsinference_ep
} // namespace onnxruntime

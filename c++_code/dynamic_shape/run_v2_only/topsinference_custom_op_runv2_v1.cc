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
                               const int compiled_batchsize,
                               const std::vector<std::string> &dynamic_infos) {

  LOGS(*logger_, INFO) << "TopsInference Custom Op Compute";

  Ort::CustomOpApi ort{*api};

  // 判断是否是动态形状
  // has_dynamic_batch has_dynamic_shape
  // ===================================================
  // TODO : bs维度不能是动态
  bool has_dynamic_shape = false;
  bool use_legacy = false;
  size_t data_batchsize = 0;
  std::string input_name_strings = "";
  std::string input_shape_strings = "";

  // 确定input_str
  std::string max_setting_str;
  std::string min_setting_str;
  if (!(dynamic_infos.empty())) {
    std::string dynamic_input_names = dynamic_infos[0];
    std::string dynamic_min_shape = dynamic_infos[1];
    std::string dynamic_max_shape = dynamic_infos[2];
    std::vector<std::string> max_shape_dim;
    std::vector<std::string> min_shape_dim;

    // dynamic_min_shape
    std::stringstream s_stream_name(dynamic_input_names);
    std::stringstream s_stream_min(dynamic_min_shape);
    std::stringstream s_stream_max(dynamic_max_shape);

    while (s_stream_name.good()) {
      std::string substr_name;
      std::string substr_min;
      std::string substr_max;

      getline(s_stream_name, substr_name, ';');
      getline(s_stream_min, substr_min, ';');
      getline(s_stream_max, substr_max, ';');
      min_shape_dim.push_back(substr_min);
      max_shape_dim.push_back(substr_max);

      std::stringstream s_min(substr_min);
      std::stringstream s_max(substr_max);

      // 每个input的shape字符转
      std::string str = "";
      while (s_min.good()) {
        std::string s_a;
        std::string s_b;
        getline(s_min, s_a, ',');
        getline(s_max, s_b, ',');

        if (std::stoi(s_b) > std::stoi(s_a)) {
          str += "-1,";
          has_dynamic_shape = true;
        } else if (s_b == s_a) {
          str += s_b + ",";
        } else {
          // set error
          exit(0);
        }
      }

      // 设置字符串名称
      input_shape_strings += (str.substr(0, str.length() - 1) + ":");
      input_name_strings += substr_name + ",";
    }

    std::cout << "input_name_strings:" << input_name_strings << std::endl;
    std::cout << "input_shape_strings:" << input_shape_strings << std::endl;

    if (has_dynamic_shape) {
      // 遍历列表长度
      // op_min_val : [{\"main\":[\"1,16,16,64\", \"1,8,8,128\"]}]
      // op_max_val : [{\"main\":[\"1,16,16,64\", \"1,8,8,128\"]}]

      std::string op_max_val;
      std::string op_min_val;
      for (size_t i = 0; i < max_shape_dim.size(); i++) {
        op_max_val += ("\"" + max_shape_dim[i] + "\",");
        op_min_val += ("\"" + min_shape_dim[i] + "\",");
      }
      max_setting_str = "[{\"main\":[" +
                        op_max_val.substr(0, op_max_val.length() - 1) + "]}]";
      min_setting_str = "[{\"main\":[" +
                        op_min_val.substr(0, op_min_val.length() - 1) + "]}]";
    }
  } else {
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
  } else if ((input_shapes_str != input_shape_strings) && !has_dynamic_shape) {
    LOGS(*logger_, WARNING)
        << "Trigger Recompile Progress. Input shapes: " << input_shape_strings
        << " is different from last time: " << input_shapes_str;
    // Reassign input shapes as the string has changed
    input_shapes_str = input_shape_strings;
    tensor_shape_changed = true;
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
      optimizer->getConfig()->setMaxShapeRange(max_setting_str.c_str());
      optimizer->getConfig()->setMinShapeRange(min_setting_str.c_str());
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

    // set input ITensor, 临时设置, for runV2
    TopsInference::ITensor *sub_input = TopsInference::create_tensor();
    sub_input->setOpaque(reinterpret_cast<void *>(inputs[i]));
    sub_input->setDeviceType(
        TopsInference::DataDeviceType::HOST); // you should set DeviceType
    TopsInference::Dims input_shape = engine_->getInputShape(i);

    int counter = 0;
    std::for_each(ort_shape.begin(), ort_shape.end(),
                  [&counter, &input_shape](int64_t n) {
                    std::cout << "dim: " << n << std::endl;
                    input_shape.dimension[counter] = n;
                    counter++;
                  });

    sub_input->setDims(input_shape);
    input_tensor_list.emplace_back((TopsInference::ITensor_t)sub_input);
  }

  // If any output has the same value at 0th dimension as that in the
  // first output, we assume that the specific output has semantics of
  // batchsize. Seemingly unreasonable assumptions here as the functional
  // limitations and design flaws of the engine framework.

  std::vector<TopsInference::ITensor_t> output_tensor_list;

  int32_t bs_in_engine =
      num_outputs > 0 ? engine_->getMaxOutputShape(0).dimension[0] : 0;
  for (int i = 0; i < num_outputs; ++i) {
    auto output_dims = engine_->getMaxOutputShape(i);
    std::vector<ssize_t> output_shape;

    if (has_dynamic_shape) {
      // 动态 output_shape， 设置最大形状
      for (int32_t j = 0; j < output_dims.nbDims; ++j) {
        output_shape.push_back(output_dims.dimension[j]);
      }
    } else {
      // 非动态 output_shape, legacy mode and unlegacy mode
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

    // 转换成 ITensor_t
    sub_output->setOpaque(reinterpret_cast<void *>(outputs[i]));
    sub_output->setDims(max_shape); // you should use maxoutputshape to set Dims
    sub_output->setDeviceType(
        TopsInference::DataDeviceType::HOST); // you should set DeviceType
    output_tensor_list.emplace_back((TopsInference::ITensor_t)sub_output);
  }
}

auto beforeTime = std::chrono::steady_clock::now();
engine_->runV2(input_tensor_list.data(), output_tensor_list.data());
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

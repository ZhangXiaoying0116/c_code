# -*- coding: utf-8 -*-
import os
os.environ["COMPILE_OPTIONS_MLIR_DBG"]="-pass-timing -pass-statistics -print-ir-before=hlir-first-pass,hlir-fusion,hlir-fp16-quantize-pass,factor-first-pass,func_op_flatten,llir-sipcode-prefetch -print-ir-after=hlir-fp16-quantize-pass,hlir-fusion,hlir-last-pass,func_op_flatten,llir-sipcode-prefetch,factor-last-pass -mlir-elide-elementsattrs-if-larger=8 -log-output-path=/tmp/irdump/"
import random
from _pytest import logging
import TopsInference
import numpy as np
import pytest
import sys
import onnxruntime as ort

sys.path.append("..")
from py_test.lib.gen_data import gen_rand_data
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s : %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

HOST_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
MODEL_PATH = os.path.join(HOST_PATH, "py_test/inputs/models")

onnxruntime_topsinference_fp32_list = [
    pytest.param(
        {
            "model_name": os.path.join(MODEL_PATH, "resnet50_v1.5-torchvision-op13-fp32-N.onnx"),
            "inputs": [{"dtype": "float32", "shape": [1,3,224,224], "rand": "normal", "loc": 0.0, "scale": 1.0,}],
            "outputs" : np.full((1, 1000), 0, dtype=np.float32),
            "buffer_type": TopsInference.TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE,
            "set_build_flag": TopsInference.KDEFAULT,
            "rtol": 1e-4,
            "atol": 1e-4,
            "raise": 0,
        },
        id = "test_api_onnxruntime_topsinference_run_resnet50_fp32_normal_gcu",
        ),

]


onnxruntime_topsinference_fp16_list = [
    pytest.param(
        {
            "model_name": os.path.join(MODEL_PATH, "resnet50_v1.5-torchvision-op13-fp32-N.onnx"),
            "inputs": [{"dtype": "float32", "shape": [1,3,224,224], "rand": "normal", "loc": 0.0, "scale": 1.0,}],
            "outputs" : np.full((1, 1000), 0, dtype=np.float32),
            "buffer_type": TopsInference.TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE,
            "set_build_flag": TopsInference.KFP16_MIX,
            "rtol": 1e-4,
            "atol": 1e-4,
            "raise": 0,
        },
        id = "test_api_onnxruntime_topsinference_run_resnet50_fp16_normal_gcu",
        ),
]

class TestOnnxruntimeEngineRunApi:

    @pytest.mark.parametrize('fixture_overwatch', onnxruntime_topsinference_fp32_list)
    def test_onnxruntime_topsinference_engine_run_fp32(self, fixture_overwatch):
        para_dict = fixture_overwatch
        buffer_type = para_dict.get('buffer_type')
        model_name = para_dict.get("model_name")
        input_data = para_dict.get('inputs')
        outputs = para_dict.get('outputs')
        set_build_flag = para_dict.get('set_build_flag')
        stream = para_dict.get('stream')
        rtol = para_dict.get('rtol')
        atol = para_dict.get('atol')

        # prepare input data
        inputs = []
        if isinstance(input_data[0], dict):
            for idx, input_item in enumerate(input_data):
                random_data, seed = gen_rand_data(input_item)
                inputs.append(random_data)
        else:
            inputs = input_data

        # Run fp32 mode and disable fp16 mode by setting environment variables
        os.environ["ORT_TOPSINFERENCE_FP16_ENABLE"] = "0"

        #alloc device
        device_handle = TopsInference.set_device(0, [0])
        parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
        parser.set_input_names(["input"])
        parser.set_input_dtypes([TopsInference.DT_FLOAT32])
        parser.set_input_shapes([[1,3,224,224]])
        parser.set_output_dtypes([TopsInference.DT_FLOAT32])
        parser.set_output_names(["output"])
        parser.enable_graph_builder()
        network = parser.read(model_name)
        optimizer=TopsInference.create_optimizer()
        optimizer.set_compile_options({'hlir_options': 'tops-hlir-pipeline{non-dtu=true}'})
        optimizer.set_build_flag(set_build_flag)
        engine = optimizer.build(network)

        #prepare data for test function
        p_inputs_list = [TopsInference.mem_alloc(tensor.size * tensor.itemsize) for tensor in inputs]
        for i in range(len(p_inputs_list)):
            TopsInference.mem_h2d_copy(inputs[i], p_inputs_list[i],
                            inputs[i].size * inputs[i].itemsize)
        p_outputs = TopsInference.mem_alloc(outputs.size * outputs.itemsize)

        if para_dict["raise"]:  #exception expected
            with pytest.raises(BaseException) as e:
                engine.run(p_inputs_list, [p_outputs], buffer_type,stream)
            assert e.match(para_dict["raise"])
        else:  #pass expected
            engine.run(p_inputs_list, [p_outputs], buffer_type,stream)
            if buffer_type == TopsInference.TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE:
                TopsInference.mem_d2h_copy(p_outputs, outputs,
                                    outputs.size * outputs.itemsize,
                )
            else:
                outputs = p_outputs
            logging.debug("output data:{}".format(outputs))

            #--teardown stage--
            if buffer_type == TopsInference.TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE:
                for p_input in p_inputs_list:
                    TopsInference.mem_free(p_input)
            TopsInference.release_device(device_handle)

            sess = ort.InferenceSession(model_name, providers=['TopsInferenceExecutionProvider'], provider_options=[{}])
            output_name = sess.get_outputs()[0].name

            input_map = {}
            for i in range(len(inputs)):
                input_map[sess.get_inputs()[i].name]= inputs[i]

            result = sess.run([output_name], input_map)[0]
            logging.debug("result data:{}".format(result))
            # Compare the accuracy of the two methods
            # assert np.allclose(outputs, result)
            compare_result = list(zip(outputs, result))
            logging.info("Compare_result:{}".format(compare_result))
            for compare in compare_result:
                np.testing.assert_allclose(compare[0], compare[1], rtol=rtol, atol=atol, err_msg = "The actual and expected values are not equal to the specified precision!!!")

    @pytest.mark.parametrize('fixture_overwatch', onnxruntime_topsinference_fp16_list)
    def test_onnxruntime_topsinference_engine_run_fp16(self, fixture_overwatch):
        para_dict = fixture_overwatch
        buffer_type = para_dict.get('buffer_type')
        model_name = para_dict.get("model_name")
        input_data = para_dict.get('inputs')
        outputs = para_dict.get('outputs')
        set_build_flag = para_dict.get('set_build_flag')
        stream = para_dict.get('stream')
        rtol = para_dict.get('rtol')
        atol = para_dict.get('atol')

        # prepare input data
        inputs = []
        if isinstance(input_data[0], dict):
            for idx, input_item in enumerate(input_data):
                random_data, seed = gen_rand_data(input_item)
                inputs.append(random_data)
        else:
            inputs = input_data

        # The FP16 blended mode is run by default. You can control the fp16 mode by setting export ORT_TOPSINFERENCE_FP16_ENABLE=1
        os.environ["ORT_TOPSINFERENCE_FP16_ENABLE"] = "1"

        #alloc device
        device_handle = TopsInference.set_device(0, [0])
        parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
        parser.set_input_names(["input"])
        parser.set_input_dtypes([TopsInference.DT_FLOAT32])
        parser.set_input_shapes([[1,3,224,224]])
        parser.set_output_dtypes([TopsInference.DT_FLOAT32])
        parser.set_output_names(["output"])
        parser.enable_graph_builder()
        network = parser.read(model_name)
        optimizer=TopsInference.create_optimizer()
        optimizer.set_compile_options({'hlir_options': 'tops-hlir-pipeline{non-dtu=true}'})
        optimizer.set_build_flag(set_build_flag)
        engine = optimizer.build(network)

        #prepare data for test function
        p_inputs_list = [TopsInference.mem_alloc(tensor.size * tensor.itemsize) for tensor in inputs]
        for i in range(len(p_inputs_list)):
            TopsInference.mem_h2d_copy(inputs[i], p_inputs_list[i],
                            inputs[i].size * inputs[i].itemsize)
        p_outputs = TopsInference.mem_alloc(outputs.size * outputs.itemsize)

        if para_dict["raise"]:  #exception expected
            with pytest.raises(BaseException) as e:
                engine.run(p_inputs_list, [p_outputs], buffer_type,stream)
            assert e.match(para_dict["raise"])
        else:  #pass expected
            engine.run(p_inputs_list, [p_outputs], buffer_type,stream)
            if buffer_type == TopsInference.TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE:
                TopsInference.mem_d2h_copy(p_outputs, outputs,
                                    outputs.size * outputs.itemsize,
                )
            else:
                outputs = p_outputs
            logging.debug("output data:{}".format(outputs))

            #--teardown stage--
            if buffer_type == TopsInference.TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE:
                for p_input in p_inputs_list:
                    TopsInference.mem_free(p_input)
            TopsInference.release_device(device_handle)

            sess = ort.InferenceSession(model_name, providers=['TopsInferenceExecutionProvider'], provider_options=[{}])
            output_name = sess.get_outputs()[0].name

            input_map = {}
            for i in range(len(inputs)):
                input_map[sess.get_inputs()[i].name]= inputs[i]

            result = sess.run([output_name], input_map)[0]
            logging.debug("result data:{}".format(result))
            # Compare the accuracy of the two methods
            # assert np.allclose(outputs, result)
            compare_result = list(zip(outputs, result))
            logging.info("Compare_result:{}".format(compare_result))
            for compare in compare_result:
                np.testing.assert_allclose(compare[0], compare[1], rtol=rtol, atol=atol, err_msg = "The actual and expected values are not equal to the specified precision!!!")

if __name__ == '__main__':
    pytest.main(["-s", "test_API_engine_run.py"])

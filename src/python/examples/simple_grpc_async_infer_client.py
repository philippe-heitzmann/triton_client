#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import partial
import argparse
import time
import sys

import cv2
import numpy as np

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

def read_image(filepath) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def preprocess(img, input_shape, letter_box=True):
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds. Default is None.')

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = 'yolov7'
    model_metadata = triton_client.get_model_metadata(model_name)
    print('model_metadata ', model_metadata)
    print(type(model_metadata))
    # Infer
    inputs, outputs = [], []
    img = read_image('/mnt/c/Users/phil0/DS/triton_client/data/people.jpg')
    # img = np.float32(cv2.resize(img, (640, 640)))
    img = np.float32(preprocess(img, [640,640]))
    # img = np.expand_dims(input_image_buffer, axis=0)
    # .transpose(2,0,1)
    print('img.shape ', img.shape)
    image_data = np.stack([img for i in range(8)])
    print('image_data.shape ', image_data.shape)
    # image_data = np.ones([4,3,640,640], np.float32)
    inputs.append(grpcclient.InferInput('images', image_data.shape, "FP32"))
    # inputs.append(grpcclient.InferInput('INPUT1', image_data.shape, "FP32"))
    # outputs=[grpcclient.InferRequestedOutput(output) for output in output_names ]
    # inputs[0].set_data_from_numpy(image_data)

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    # input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    # input0_data = np.expand_dims(input0_data, axis=0)
    # input1_data = np.ones(shape=(1, 16), dtype=np.int32)

    # Initialize the data
    inputs[0].set_data_from_numpy(image_data)
    # inputs[1].set_data_from_numpy(image_data)
    # print('__dict__ ', model_metadata["outputs"].__dict__)
    # output_names = [str(i["name"]) for i in model_metadata["outputs"]]
    output_names = ["det_classes","det_scores","det_boxes","num_dets"]
    outputs=[grpcclient.InferRequestedOutput(output) for output in output_names ]
    # outputs.append(grpcclient.InferRequestedOutput('OUTPUT0'))
    # outputs.append(grpcclient.InferRequestedOutput('OUTPUT1'))

    # Define the callback function. Note the last two parameters should be
    # result and error. InferenceServerClient would povide the results of an
    # inference as grpcclient.InferResult in result. For successful
    # inference, error will be None, otherwise it will be an object of
    # tritonclientutils.InferenceServerException holding the error details
    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    # list to hold the results of inference.
    user_data = []

    # Inference call
    triton_client.async_infer(model_name=model_name,
                              inputs=inputs,
                              callback=partial(callback, user_data),
                              outputs=outputs,
                              client_timeout=FLAGS.client_timeout)

    # Wait until the results are available in user_data
    time_out = 10
    while ((len(user_data) == 0) and time_out > 0):
        time_out = time_out - 1
        time.sleep(1)
        
    print('user_data ', user_data)

    # Display and validate the available results
    if ((len(user_data) == 1)):
        # Check for the errors
        if type(user_data[0]) == InferenceServerException:
            print(user_data[0])
            sys.exit(1)

        # Validate the values by matching with already computed expected
        # values.
        det_classes = user_data[0].as_numpy("det_classes")
        print('det_classes ', det_classes)
        num_dets = user_data[0].as_numpy("num_dets")
        print('num_dets ', num_dets)
        det_boxes = user_data[0].as_numpy("det_boxes")
        print('det_boxes ', det_boxes)
        det_scores = user_data[0].as_numpy("det_scores")
        print('det_scores ', det_scores)
        # "det_scores","det_boxes","num_dets"]
        output0_data = user_data[0].as_numpy('OUTPUT0')
        output1_data = user_data[0].as_numpy('OUTPUT1')
        print('output0_data ', output0_data)
        print('output1_data ', output1_data)
        
        # for i in range(16):
        #     print(
        #         str(input0_data[0][i]) + " + " + str(input1_data[0][i]) +
        #         " = " + str(output0_data[0][i]))
        #     print(
        #         str(input0_data[0][i]) + " - " + str(input1_data[0][i]) +
        #         " = " + str(output1_data[0][i]))
        #     if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
        #         print("sync infer error: incorrect sum")
        #         sys.exit(1)
        #     if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
        #         print("sync infer error: incorrect difference")
        #         sys.exit(1)
        # print("PASS: Async infer")



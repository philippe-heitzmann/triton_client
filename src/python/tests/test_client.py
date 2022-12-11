import argparse
import numpy as np
from PIL import Image
import os
import sys
import struct

import grpc

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc


# Create gRPC stub for communicating with the server
channel = grpc.insecure_channel(FLAGS.url)
grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

# Make sure the model matches our requirements, and get some
# properties of the model that we need for preprocessing
metadata_request = service_pb2.ModelMetadataRequest(
    name=FLAGS.model_name, version=FLAGS.model_version)
metadata_response = grpc_stub.ModelMetadata(metadata_request)

config_request = service_pb2.ModelConfigRequest(name=FLAGS.model_name,
                                                version=FLAGS.model_version)
config_response = grpc_stub.ModelConfig(config_request)

max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
    metadata_response, config_response.config)
'''Example run python3 /mnt/c/Users/phil0/DS/triton_client/src/python/examples/grpc_async.py'''

from functools import partial
import numpy as np
import tritonclient.grpc as grpcclient
import logging 
import time

logging.basicConfig(level=logging.INFO)
logging.getLogger("TritonClient").setLevel(logging.INFO)
log = logging.getLogger("TritonClient")

number_of_requests=8
requests_concurrency=10
triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False) #concurrency=requests_concurrency
model_name = "yolov7"
model_metadata = triton_client.get_model_metadata(model_name)

user_data = []
def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)

input_names = ["images"]
output_names = ["det_classes","det_scores","det_boxes","num_dets"]
# input_names = [str(i["name"]) for i in model_metadata["inputs"]]
# output_names = [str(i["name"]) for i in model_metadata["outputs"]]
async_requests=[]
final_results =[]

# Loading dummy input for the model
image_data = np.ones([1,3,640,640], np.float32)
inputs=[grpcclient.InferInput(input_names[0], image_data.shape, "FP32")]
outputs=[grpcclient.InferRequestedOutput(output) for output in output_names]
inputs[0].set_data_from_numpy(image_data)


if __name__ == '__main__':
    print('inputs', input_names)
    print('outputs', output_names)
    print('model_metadata ', model_metadata)
    begining=time.time()
    for j in range(number_of_requests):
        start=time.time()
        async_requests=[]             
        for i in range(requests_concurrency) :
            async_requests.append(triton_client.async_infer(model_name=model_name, inputs=inputs, outputs=outputs, callback=partial(callback, user_data)))
        print(async_requests)
        # for async_request in async_requests:
        #     final_results.append(async_request.get_result())
        # log.info("Time taken to run {} asynchronous inference requests is {} ".format(requests_concurrency, time.time()-start))

    log.info("Total time to run {} inference requests is {}".format(requests_concurrency*number_of_requests, time.time()-begining))
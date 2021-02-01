#
# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
import numpy as np
from builtins import range

import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient import utils
import traceback

FLAGS = None
import PyNvCodec as nvc
import numpy as np
import sys, os
import cv2
from threading import Thread
from PIL import Image
from utils.logger import setup_logger

logger = setup_logger(name='boe_decode')


class Worker(Thread):
    def __init__(self, gpuID: int, encFile: str):
        Thread.__init__(self)
        try:
            logger.debug(f"gpuId is:{gpuID}, encFile is: {encFile}")
            self.nvDec = nvc.PyNvDecoder(encFile, gpuID,
                                         {'rtsp_transport': 'tcp', 'max_delay': '5000000', 'bufsize': '30000k'})
            width, height = self.nvDec.Width(), self.nvDec.Height()
            hwidth, hheight = int(width / 2), int(height / 2)

            self.nvCvt = nvc.PySurfaceConverter(width, height, self.nvDec.Format(), nvc.PixelFormat.YUV420, gpuID)
            self.nvRes = nvc.PySurfaceResizer(hwidth, hheight, self.nvCvt.Format(), gpuID)
            self.nvDwn = nvc.PySurfaceDownloader(hwidth, hheight, self.nvRes.Format(), gpuID)
            self.num_frame = 0
        except:
            logger.critical(
                f"Input with some error, should be python main.py \"0 rtsp://admin:abcd1234@192.168.8.222\" ")
            os._exit(0)

    def run(self):
        try:
            while True:
                try:
                    rawSurface = self.nvDec.DecodeSingleSurface()
                    if (rawSurface.Empty()):
                        print('No more video frames')
                        break
                except nvc.HwResetException:
                    print('Continue after HW decoder was reset')
                    continue

                cvtSurface = self.nvCvt.Execute(rawSurface)
                if (cvtSurface.Empty()):
                    print('Failed to do color conversion')
                    break

                resSurface = self.nvRes.Execute(cvtSurface)
                if (resSurface.Empty()):
                    print('Failed to resize surface')
                    break

                rawFrame = np.ndarray(shape=(resSurface.HostSize()), dtype=np.uint8)
                success = self.nvDwn.DownloadSingleSurface(resSurface, rawFrame)
                if not (success):
                    print('Failed to download surface')
                    break

                self.num_frame += 1
                if (0 == self.num_frame % self.nvDec.Framerate()):
                    print(self.num_frame)

        except Exception as e:
            print(getattr(e, 'message', str(e)))
            # decFile.close()


def create_threads(*gpuid_camera_pair):
    try:
        threads_counter = 0
        threads_works = []
        logger.info(f"Input gpuid , camera url is :{gpuid_camera_pair[1:]}")
        for argspair in gpuid_camera_pair[1:]:
            gpuid, cam_url = argspair.split(' ')
            threads_works.append(Worker(int(gpuid), str(cam_url)))
            threads_counter += 1
    except:
        traceback.print_exception(*sys.exc_info())
        os._exit(1)

    if (threads_counter > 20):
        print(f"total worker threads is: {threads_counter}, may be to much threads ....")

    for work in threads_works:
        work.start()
    for work in threads_works:
        work.join()



def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = utils.triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if protocol == "grpc":
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled
    else:
        if format == "FORMAT_NCHW":
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def Postprocess(output0_data: np.ndarray, label_filename: str, topK: int):
    if os.path.exists(label_filename):
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)  # only difference
        with open(label_filename) as fid:
            labels = list(map(str.strip, fid.readlines()))
            assert output0_data.size == len(labels)
            labelsarray = np.asarray(labels, dtype='<U20')
            inds = output0_data.argsort()[-topK:][::-1]
            return (labelsarray[inds].tolist(),softmax(output0_data[inds]).tolist())


if __name__ == "__main__":
    URL = 'localhost:8000'
    verbose = True
    verbose = False
    INFER_MODEL_NAME = "densenet_onnx"
    INFER_MODEL_VERSION = "1"
    topK = 3
    c, h, w = 3, 224, 224
    label_filename = "densenet_labels.txt"

    try:
        triton_client = httpclient.InferenceServerClient(url=URL, verbose=verbose)
    except Exception as e:
        logger.error(f"channel creation failed: {str(e)}")
        sys.exit(1)
    # To make sure no shared memory regions are registered with the
    # server.
    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()
    logger.info(f"server_live is :{triton_client.is_server_live()},server_ready is :{triton_client.is_server_ready()}")
    if (not triton_client.is_server_live() or not triton_client.is_server_ready()):
        logger.critical(
            f"server_live is :{triton_client.is_server_live()},server_ready is :{triton_client.is_server_ready()}")
        os._exit(0)

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    '''
        example model define is:
        name: "densenet_onnx"
        platform: "onnxruntime_onnx"
        max_batch_size : 0
        input [
          {
            name: "data_0"
            data_type: TYPE_FP32
            format: FORMAT_NCHW
            dims: [ 3, 224, 224 ]
            reshape { shape: [ 1, 3, 224, 224 ] }
          }
        ]
        output [
          {
            name: "fc6_1"
            data_type: TYPE_FP32
            dims: [ 1000 ]
            reshape { shape: [ 1, 1000, 1, 1 ] }
            label_filename: "densenet_labels.txt"
          }
        ]
    '''
    # mug.jpg
    img = Image.open('images/mug.jpg')
    img = Image.open('images/car.jpg')
    img = Image.open('images/vulture.jpeg')

    img.resize((w,h))

    input0_data = preprocess(img, "FORMAT_NCHW", "FP32", c, h, w, "INCEPTION", "cuda_shared_memory")
    print(input0_data.shape,input0_data.dtype)
    # input0_data = np.zeros(1 * 3 * 224 * 224, dtype=np.single)
    output0_data = np.zeros(1 * 1000 * 1 * 1, dtype=np.single)

    input_byte_size = input0_data.size * input0_data.itemsize
    output_byte_size = output0_data.size * output0_data.itemsize

    # Create Output0 and Output1 in Shared Memory and store shared memory handles
    shm_op0_handle = cudashm.create_shared_memory_region(
        "output0_data", output_byte_size, 0)
    # Register Output0 and Output1 shared memory with Triton Server
    triton_client.register_cuda_shared_memory(
        "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0,
        output_byte_size)

    # Create Input0 and Input1 in Shared Memory and store shared memory handles
    shm_ip0_handle = cudashm.create_shared_memory_region(
        "input0_data", input_byte_size, 0)
    # Put input data values into shared memory
    cudashm.set_shared_memory_region(shm_ip0_handle, [input0_data])
    # Register Input0 and Input1 shared memory with Triton Server
    triton_client.register_cuda_shared_memory(
        "input0_data", cudashm.get_raw_handle(shm_ip0_handle), 0,
        input_byte_size)
    # Set the parameters to use data from shared memory
    inputs = []
    inputs.append(httpclient.InferInput('data_0', [3, 224, 224], "FP32"))
    inputs[-1].set_shared_memory("input0_data", input_byte_size)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('fc6_1', binary_data=True))
    outputs[-1].set_shared_memory("output0_data", output_byte_size)

    # run infer
    # create_threads(*sys.argv)
    results = triton_client.infer(model_name=INFER_MODEL_NAME, inputs=inputs, outputs=outputs)
    # results = triton_client.async_infer(model_name=INFER_MODEL_NAME, inputs=inputs, outputs=outputs)

    # Read results from the shared memory.
    output0 = results.get_output("fc6_1")

    if output0 is not None:
        output0_data = cudashm.get_contents_as_numpy(shm_op0_handle, utils.triton_to_np_dtype(output0['datatype']),
                                                     output0['shape'])
        logger.debug(
            f"the output is: {output0},output type is:{type(output0)},output0_data0 type is :{type(output0_data)} output_data0 shape is :{output0_data.shape}")
        logger.debug(f"{Postprocess(output0_data, label_filename, topK)}")
    else:
        logger.error("OUTPUT0 is missing in the response.")
        sys.exit(1)

try:
    logger.debug(f"the triton-server shared memory status is:{triton_client.get_cuda_shared_memory_status()}")
    triton_client.unregister_cuda_shared_memory()
    # assert len(cudashm.allocated_shared_memory_regions()) == 4
    cudashm.destroy_shared_memory_region(shm_ip0_handle)
    cudashm.destroy_shared_memory_region(shm_op0_handle)
    # assert len(cudashm.allocated_shared_memory_regions()) == 0
except:
    pass
    # traceback.print_exception(*sys.exc_info())
    # os._exit(1)
    '''
        python main.py "0 rtsp://admin:abcd1234@192.168.8.222"
    '''

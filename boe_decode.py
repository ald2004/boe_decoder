import PyNvCodec as nvc
import numpy as np
import sys
import cv2

def decode_work(gpuID,encFile):

    nvDec = nvc.PyNvDecoder(encFile, gpuID, { # yuv 420p
        'rtsp_transport': 'tcp',
        # 'max_delay': '5000000',
        # 'bufsize': '30000k',
        # 'pixel_format':'rgb'
    })
    width, height = 1280 , 720
    nvDwn = nvc.PySurfaceDownloader(width, height, nvc.PixelFormat.BGR, 0)
    cvtNV12_BGR = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, nvc.PixelFormat.BGR, gpuID)
    nvDwn_planar = nvc.PySurfaceDownloader(width, height, nvc.PixelFormat.RGB_PLANAR, gpuID)
    print('xxxxxxxxxxxx')
    print(nvDec.Framesize())
    frameBGR = np.ndarray(shape=(nvDec.Framesize()), dtype=np.uint8)

    # for _ in range(2):
    success = nvDwn.DownloadSingleSurface(cvtNV12_BGR.Execute(nvDec.DecodeSingleSurface()), frameBGR)
    # success = nvDwn_planar.DownloadSingleSurface(nvDec.DecodeSingleSurface(), frameBGR)
    # print(type(nvDwn_planar)) #<class 'PyNvCodec.PySurfaceDownloader'>
    # cv2.imwrite('/dev/shm/xx.jpg', frameBGR.reshape((height,width, 3)))
    cv2.imwrite('/dev/shm/xx.jpg', frameBGR.reshape((height,width, 3)))

    # width, height = nvDec.Width(), nvDec.Height()
    # hwidth, hheight = int(width / 2), int(height / 2)
    # print(f'nvdec format is: {nvDec.Format()}, w is : {width}, h is : {height}')
    # nvCvt = nvc.PySurfaceConverter(width, height, nvDec.Format(), nvc.PixelFormat.YUV420, gpuID)
    # # nvCvt = nvc.PySurfaceConverter(width, height, nvDec.Format(), nvc.PixelFormat.BGR, gpuID)
    # nvRes = nvc.PySurfaceResizer(hwidth, hheight, nvCvt.Format(), gpuID)
    # nvDwn = nvc.PySurfaceDownloader(hwidth, hheight, nvRes.Format(), gpuID)
    # # nvDwn = nvc.PySurfaceDownloader(hwidth, hheight, nvDec.Format(), gpuID)
    # to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
    #
    # num_frame = 0
    #
    # try:
    #     while True:
    #         try:
    #             rawSurface = nvDec.DecodeSingleSurface()
    #             if (rawSurface.Empty()):
    #                 print('No more video frames')
    #                 break
    #         except nvc.HwResetException:
    #             print('Continue after HW decoder was reset')
    #             continue
    #
    #         # cvtSurface = nvCvt.Execute(rawSurface)
    #         # if (cvtSurface.Empty()):
    #         #     print('Failed to do color conversion')
    #         #     break
    #         #
    #         # resSurface = nvRes.Execute(cvtSurface)
    #         # if (resSurface.Empty()):
    #         #     print('Failed to resize surface')
    #         #     break
    #         rgb_byte = to_rgb.Execute(rawSurface)
    #         # rawFrame = np.ndarray(shape=(resSurface.HostSize()), dtype=np.uint8)
    #         rawFrame = np.ndarray(shape=(1280,720), dtype=np.uint8)
    #         # success = nvDwn.DownloadSingleSurface(resSurface, rawFrame)
    #         # success = nvDwn.DownloadSingleSurface(rawSurface, rawFrame)
    #         success = nvDwn.DownloadSingleSurface(rgb_byte, rawFrame)
    #         if not (success):
    #             print('Failed to download surface')
    #             break
    #
    #         num_frame += 1
    #         if (0 == num_frame % nvDec.Framerate()):
    #             print(num_frame)
    #
    # except Exception as e:
    #     print(getattr(e, 'message', str(e)))
    #     # decFile.close()

if __name__=="__main__":
    # python main.py 0 rtsp://admin:abcd1234@192.168.8.222 9 1
    gpu_1 = int(sys.argv[1])
    input_1 = sys.argv[2]
    print(sys.argv[1],sys.argv[2])
    decode_work(gpu_1,input_1)
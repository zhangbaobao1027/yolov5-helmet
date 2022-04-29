import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from alert_msg import send_alert_msg
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    try:
        if webcam:
            if view_img:
                view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
    except Exception as e:
        print('获取图像或者视频流失败,请确认！')
        return -1


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 组织告警信息
        is_alert_flag = False
        is_alert_path = ""
        is_alert_person = 0
        is_alert_camera = opt.name  # 摄像机ip
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # 获取当前时刻秒数
            times = time.time()
            str_times = str(times)

            p = Path(p)  # to Path
            # 对name进行截取
            # sub_names = p.name.split('_')
            if '.flv' in p.name:
                save_path = str(save_dir / str(str_times))  # img.jpg
                is_alert_path = str(str_times)
                # label名称
                txt_path = str(save_dir / 'labels' / is_alert_path) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            else:
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            # 日志中增加相机ip
            s += '{}: '.format(is_alert_camera)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                person_num = 0
                for *xyxy, conf, cls in reversed(det):

                    # if int(cls) == 0 or int(cls) == 1 or int(cls) == 2:  # 0-人 1-头 2-安全帽
                    # 密闭空间要求有人没帽 则报警
                    # if int(cls) == 1:  # 0-人 1-头 2-安全帽
                    person_num += 1
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=opt.line_thickness)

                    # 将告警标志置为True
                    is_alert_flag = True
                # 告警人数
                is_alert_person = person_num

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)
            # modify 20220408 流也保存图片
            if save_img and is_alert_flag:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                elif dataset.mode == 'stream':
                    # fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.png'
                    is_alert_path += '.png'
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            # 告警消息发送
            if is_alert_flag:
                send_alert_msg(is_alert_path, is_alert_camera, is_alert_person)

            is_alert_flag = False
            # 受限空间http流需要跳出
            # break
        cv2.waitKey(2000)  # 1 millisecond
        # time.sleep(2)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def run_yolo_iamge_serve(fvl, camera_ip, smp='runs/detect'):
    """
        --weights:权重的路径地址
        --source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
        --output:网络预测之后的图片/视频的保存路径
        --img-size:网络输入图片大小
        --conf-thres:置信度阈值
        --iou-thres:做nms的iou阈值
        --device:是用GPU还是CPU做推理
        --view-img:是否展示预测之后的图片/视频，默认False
        --save-txt:是否将预测的框坐标以txt文件形式保存，默认False
        --classes:设置只保留某一部分类别，形如0或者0 2 3
        --agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
        --augment:推理的时候进行多尺度，翻转等操作(TTA)推理
        --update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
        --project：推理的结果保存在runs/detect目录下
        --name：结果保存的文件夹名称
        """

    # net_source = 'http://192.168.1.200:10000/api/v1/device/channelsnap?serial=34020000001320000001&realtime=true'
    # net_source = 'http://192.168.1.200:10000/sms/34020000002020000001/flv/hls/34020000001320000001_34020000001320000001.flv'
    # net_source = 'data/images'
    net_source = fvl
    save_main_path = smp
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/helmet_head_person.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=net_source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default='1',
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=save_main_path, help='save results to project/name')
    parser.add_argument('--name', default=camera_ip, help='save results to project/name')  #
    parser.add_argument('--exist-ok', default=True, action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 增加自定义告警地址url
    # parser.add_argument('--my-url', default=camera_ip, help='my camera ip address')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        while True:
            rtncode = detect(opt=opt)
            if -1 == rtncode:
                # 等待30秒 再次请求
                print('请等待30秒， 再次发起请求')
                time.sleep(30)
            else:
                break

if __name__ == '__main__':

    """
    --weights:权重的路径地址
    --source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    --output:网络预测之后的图片/视频的保存路径
    --img-size:网络输入图片大小
    --conf-thres:置信度阈值
    --iou-thres:做nms的iou阈值
    --device:是用GPU还是CPU做推理
    --view-img:是否展示预测之后的图片/视频，默认False
    --save-txt:是否将预测的框坐标以txt文件形式保存，默认False
    --classes:设置只保留某一部分类别，形如0或者0 2 3
    --agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
    --augment:推理的时候进行多尺度，翻转等操作(TTA)推理
    --update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    --project：推理的结果保存在runs/detect目录下
    --name：结果保存的文件夹名称
    """

    # net_source = 'http://192.168.1.200:10000/api/v1/device/channelsnap?serial=34020000001320000001&realtime=true'
    net_source = 'http://192.168.1.200:10000/sms/34020000002020000001/flv/hls/34020000001320000001_34020000001320000001.flv'
    # net_source = 'data/images'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/helmet_head_person.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=net_source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default='1', help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='192.168.13.193', help='save results to project/name')   #
    parser.add_argument('--exist-ok', default=True, action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print()
    check_requirements(exclude=('pycocotools', 'thop'))
    #
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            while True:
                detect(opt)


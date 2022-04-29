import multiprocessing
import time
import os

import cv2
import yaml

from alert_msg import check_online_camera, get_camera_live_flv
from detect import run_yolo_iamge_serve

if __name__ == '__main__':
    # 获取配置文件信息
    yaml_file = 'config/helmet_detect.yaml'
    with open(yaml_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    cfg_url = cfg['helmet']['api_url']
    cfg_wait = cfg['helmet']['check_cycle']
    cfg_smp = cfg['helmet']['image_main_path']
    while True:
        camera_list = check_online_camera(api_url=cfg_url)
        flv_list = []
        if len(camera_list) > 0:
            # 通过ip获取live id
            for str_ip in camera_list:
                live_flv = get_camera_live_flv(str_ip, api_url=cfg_url)
                # print(live_flv)
                # if live_flv is not None and len(live_flv) > 1:
                flv_list.append(live_flv)

        # debug
        # _flv_list = 'http://192.168.1.200:10000/sms/34020000002020000001/flv/hls/34020000001320000001_34020000001320000001.flv'
        # _camera_list = '192.168.13.193'
        _work_processes = []
        for i, camera in enumerate(camera_list):
            flv = flv_list[i]
            print('{}-{}'.format(camera, flv))
            if flv is None or len(flv) < 1:
                # 检查数据源
                print('视频流异常，跳过启动摄像机-{}'.format(camera))
                continue
            # run_yolo_iamge_serve(fvl=flv_list[0], camera_ip=camera_list[0])
            sing_process = multiprocessing.Process(target=run_yolo_iamge_serve,
                                    kwargs={'fvl': flv, 'camera_ip': camera, 'smp': cfg_smp})
            sing_process.daemon = True
            # 启动进程
            sing_process.start()
            _work_processes.append(sing_process)

        time.sleep(cfg_wait)
        process_num = 0
        for _process in _work_processes:
            if _process.is_alive():
                process_num += 1
        if process_num < 1:
            print('无摄像头工作子进程，请注意！')
            # break

        # cv2.waitKey(cfg_wait*60*1000)
        time.sleep(cfg_wait*60)
        # 退出所有子进程 重新启动
        for _process in _work_processes:
            if _process.is_alive():
                _process.terminate()
                _process.join()  # 避免僵尸进程

        print('所有摄像机子进程已终止！')
        time.sleep(2)

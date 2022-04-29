#导入请求包
import argparse
import datetime
import http.client
#导入json包
import json
import os
from multiprocessing import Process

# 增加全局变量
# 获取配置文件信息
import yaml

yaml_file = 'config/helmet_detect.yaml'
with open(yaml_file, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg)
g_api_url = cfg['helmet']['api_url']

def call_api(inputdata, url, api_type):
    """
    :param inputdata: 单个样本的输入参数，是json格式的数据
    :return: 单个样本的探真查询变量结果数据
    """
    # 调用接口
    connection = http.client.HTTPConnection(url)
    headers = {'Content-type': 'application/json'}
    json_foo = json.dumps(inputdata)
    # json_foo = 'videoIp=192.168.13.144'
    res_data = dict()
    try:
        connection.request('POST', api_type, json_foo, headers)
        connection.sock.settimeout(5.0)  # 设置超时
        response = connection.getresponse()
        res = json.loads(response.read().decode())
        # 接口有正确的数据才读入，否则为空
        # if res['result'] or res['code'] == 0:
        res_data = res
    except Exception as e:
        print(e)

    return res_data

def send_alert_msg(is_alert_path, camera_ip, person_num = 0):

    info_dict = dict()
    info_dict['alarmType'] = '1'
    info_dict['contents'] = str(person_num)
    info_dict['imageName'] = is_alert_path
    info_dict['videoIp'] = camera_ip
    # 增加摄像机此帧中的nohat人数
    info_dict['personNum'] = str(person_num)
    # 增加当前时间
    # current_time = datetime.datetime.now()
    info_dict['alarmTime'] = None
    res = json.dumps(info_dict)

    print(res)
    # global g_api_url
    url = g_api_url
    api_type = '/ykkj_space/space/v0/terminal/acceptCameraAlarm'

    print('{}---{}'.format(url, api_type))

    res_data = call_api(info_dict, url, api_type)
    print(res_data)
    # if res['result'] or res['code'] == 0:
    if bool(res_data):
        print('alert send successful!')
    else:
        print('alert send failed!')

def check_online_camera(api_url=g_api_url):

    url = api_url
    api_type = '/ykkj_space/space/v0/sVideo/getOnlineVideos'
    info_dict = dict()
    res_data = call_api(info_dict, url, api_type)
    print(res_data)
    rtn_list = []
    if bool(res_data):
        print('alert send successful!')
        if res_data['code'] == 0:
            rcv_data_list = res_data['data']
            for adata in rcv_data_list:
                print(adata['videoIp'])
                rtn_list.append(adata['videoIp'])

    else:
        print('alert send failed!')

    return rtn_list


def get_camera_live_flv(str_ip, api_url=g_api_url):

    # global g_api_url
    # g_api_url = api_url

    # 通过摄像机ip获取多媒体的flv地址
    info_dict = dict()
    # info_dict['videoIp'] = str_ip
    url = api_url

    api_type = '/ykkj_space/space/v0/sVideo/getPicturetReal?videoIp={}'.format(str_ip)
    res_data = call_api(str_ip, url, api_type)
    print(res_data)
    flv_str = ''
    if bool(res_data):
        flv_str = res_data['data']

    return flv_str

def action(name,*add):
    print(name)
    for arc in add:
        print("%s --当前进程%d" % (arc, os.getpid()))

if __name__ == '__main__':

    # send_alert_msg('123456789.png', '192.168.13.193')

    camera_list = check_online_camera()
    flv_list = []
    if len(camera_list) > 0:
        # 通过ip获取live id
        for str_ip in camera_list:
            live_flv = get_camera_live_flv(str_ip)
            print(live_flv)
            flv_list.append(live_flv)

    send_alert_msg('png', 'ip', 1)
    # 通过进程创建图像识别服务
    # my_tuple = ("http://c.biancheng.net/python/", \
    #             "http://c.biancheng.net/shell/", \
    #             "http://c.biancheng.net/java/")
    # # 创建子进程，执行 action() 函数
    # my_process = Process(target=run_yolo_iamge_serve, args=(flv_list[0], camera_list[0]))
    # # 启动子进程
    # my_process.start()

    # get_camera_live_flv('192.168.13.193')

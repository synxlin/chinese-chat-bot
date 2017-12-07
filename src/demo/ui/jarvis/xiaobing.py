#/usr/bin/env python3
#-*- coding: utf-8 -*-

import itchat as wechat
from itchat.content import *
import time
import threading
import queue
import re


class send_xiaobing_msg(threading.Thread):
    def __init__(self, qin, xiaobing_name):
        threading.Thread.__init__(self)
        self.qin = qin
        self.xiaobing_name = xiaobing_name

    def run(self):
        start=time.time()
        while True:
            msg=self.qin.get()
            wechat.send_msg(msg,self.xiaobing_name)


class receive_xiaobing_msg(threading.Thread):
    def __init__(self,qout,xiaobing_name):
        threading.Thread.__init__(self)
        self.qout = qout
        self.xiaobing_name = xiaobing_name

    def run(self):
        @wechat.msg_register(INCOME_MSG, isFriendChat=False, isGroupChat=False, isMpChat=True)
        def get_msg(msg):
            XB=wechat.search_mps(name=u'小冰')[0]['UserName']
            if msg['FromUserName'] == XB:
                if msg['Type'] == TEXT:
                    responce = msg['Content']
                    emoji = re.compile(r'\[.*?\]' )
                    responce = emoji.sub('',responce)
                    print(responce)
                    self.qout.put(responce)
                else:
                    responce = u'唔，再说一次吧'
                    print(responce)
                    self.qout.put(responce)


def xiaobing(qin,qout):
    input_queue=qin
    output_queue=qout

    def fn():
        wechat.auto_login(enableCmdQR=2)

        xiaobing_name = wechat.search_mps(name=u'小冰')[0]['UserName']

        t1 = send_xiaobing_msg(input_queue,xiaobing_name)
        t1.start()

        t2 = receive_xiaobing_msg(output_queue,xiaobing_name)
        t2.start()

        wechat.run()

    threading.Thread(target=fn).start()


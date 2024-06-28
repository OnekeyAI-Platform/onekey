# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/3/21
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.

import logging
import os
import time
from configparser import ConfigParser
from datetime import datetime

from termcolor import colored

logger = logging.root
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s - %(filename)s:%(lineno)4s]\t%(levelname)s\t%(message)s",
                                       '%Y-%m-%d %H:%M:%S'))
logger.handlers = [handler]
PIPE_HINT = colored('PIPE INPUT SUPPORTED!', 'green', attrs=['blink'])
REQUIRED = colored('REQUIRED!', 'red', attrs=['blink'])


class ProcessPrinter(object):
    """
    Print processing in two way.
    """

    def __init__(self, total_num=None):
        self.last_time = time.time()
        self.start_time = time.time()
        self.last_num = 0
        self.total_num = total_num

    def print_log(self, done_num, total_num=None, ratio=0.1, thread_id=None):
        """
        Print log every total_num * ratio number of examples.

        :param done_num: Already parsed number of samples.
        :param total_num: Total number of samples to parse.
        :param ratio: Print log after dealing `ratio` samples.
        :param thread_id: Default None. If not None, thread_id flag is used.
        """
        total_num = total_num if total_num else self.total_num
        assert total_num, "`total_num` must be supplied!"
        prefix = 'Thread id %s\t' % str(thread_id) if thread_id is not None else ''

        if done_num % (int(total_num * ratio) if int(total_num * ratio) else 1) == 0:
            speed = (done_num - self.last_num) / (time.time() - self.last_time + 1e-10)
            self.last_num = done_num
            self.last_time = time.time()
            logger.info('%sParse %.3d%% ====>> %.7d/%.7d\tSpeed: %.6f/s' % (prefix,
                                                                            done_num * 100 // total_num,
                                                                            done_num, total_num, speed))
        if done_num == total_num:
            speed = total_num / (time.time() - self.start_time + 1e-10)
            self.last_num = 0
            self.last_time = time.time()
            self.start_time = time.time()
            logger.info('%sCompleted!\tSpeed: %f/s' % (prefix, speed))


class ColorPrinter(object):
    def __init__(self):
        self._print_fn = None

    def _get_color_print(self):
        try:
            from termcolor import cprint

            def color_print(msg, color='green'):
                cprint(msg, color)

            self._print_fn = color_print
        except Exception as e:
            print("Can't use color print because of e" % e)

            def print_ignore_color(msg, *used, **unused):
                print(msg)

            self._print_fn = print_ignore_color

    def print_with_color(self, info, flag=True):
        color = 'green' if flag else 'red'
        self.cprint(info, color)

    @property
    def cprint(self):
        """ Print text with given color.

        :param msg: message to be colored
        :param color: which color to use, [grey, red, green, yellow, blue, magenta, cyan, white]
        """
        if not self._print_fn:
            self._get_color_print()
        return self._print_fn

    @staticmethod
    def color_text(msg, color='green', attrs=None):
        """ Color text if sys has termcolor.

        Parameters
        ---------------
        :param msg: message to be colored
        :param color: which color to use, [grey, red, green, yellow, blue, magenta, cyan, white]
        :param attrs: [bold, dark, underline, blink, reverse, concealed]
        :return Colored message.
        """
        try:
            if type(attrs) == str:
                attrs = [attrs]
            return colored(msg, color, attrs=attrs)
        except Exception as e:
            print(e)
            return msg


def save_report(prefix, args, save_to):
    """Save report into `save_to`.

    :param prefix: Prefix.
    :param args: Must in dict type.
    :param save_to: Save to specific path with date added.
    """
    cp = ConfigParser.ConfigParser()
    for s in args:
        cp.add_section(s)
        for o in args[s]:
            cp.set(s, o, args[s][o])

    save_to = os.path.join(save_to, datetime.now().strftime("%Y%m%d"))
    os.makedirs(save_to, exist_ok=True)
    with open(os.path.join(save_to, "%s-%s" % (prefix, datetime.now().strftime("%Y%m%d-%H%M%S"))), "w") as f:
        cp.write(f)


def log_long_str(string, logger_type=logger.info, prefix=''):
    """
    Long string to log.

    :param string: string
    :param logger_type: info, warning or error.
    :param prefix: Prefix of each line.
    :return:
    """
    for l in str(string).split('\n'):
        logger_type(f"{prefix}{l}")

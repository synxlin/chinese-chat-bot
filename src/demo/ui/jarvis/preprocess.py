#/usr/bin/env python3
# -*- coding: utf-8 -*-

from pypinyin import lazy_pinyin


def preprocess(ss):
    res = None
    ss_pinyin = lazy_pinyin(ss)
    ss_pinyin = ''.join(ss_pinyin)
    if "tianqi" in ss_pinyin:
        res = "北京今天天气"
    elif "jidian" in ss_pinyin or "shijian" in ss_pinyin:
        res = "北京现在几点"
    elif "xinwen" in ss_pinyin:
        res = "今天新闻"
    else:
        res = ss 
    return res


def test():
    ss = "景天吉点怎么样"
    res = preprocess(ss)
    print(res)


if __name__ == '__main__':
    test()

#!/usr/bin/env python3
# coding: utf-8  

import requests
import time
import logging
import sys
from bs4 import BeautifulSoup
import re
import os
import random

# 台风数据设置
year_start = 1980 #开始年份
year_end = 1981 #结束年份
typhoon_img_type = 1 #图片类型 Visible、Infrared 1、Infrared 2、Infrared 3、Infrared 4分别是0、1、2、3、4
download_dir = 'tys_raw' #下载的文件夹

# 网页URL设置
typhoon_host = 'http://agora.ex.nii.ac.jp'
typhoon_year = typhoon_host + '/digital-typhoon/year/wnp/%d.html.en'
typhoon_img_list = typhoon_host + '/digital-typhoon/summary/wnp/k/%s.html.en'
typhoon_img = typhoon_host + '/digital-typhoon/wnp/by-name/%s/%d/512x512/%s.%s.jpg'

# 下载相关设置
timeout = 31
sleep_html = 0.002
sleep_img = 0.002

# 解析相关配置
html_parer = 'html.parser'

# 文件日志配置
logger = logging.getLogger("AppName")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
file_handler = logging.FileHandler("运行日志.log")
file_handler.setFormatter(formatter) 
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

def getHeader():
    baidu_ua = 'Mozilla/5.0 (compatible; Baiduspider/2.0;+http://www.baidu.com/search/spider.html)'
    google_ua = 'Mozilla/5.0+(compatible;+Googlebot/2.1;++http://www.google.com/bot.html)'
    sogou_ua = 'Sogou+web+spider/4.0(+http://www.sogou.com/docs/help/webmasters.htm#07)'
    bing_ua = 'Mozilla/5.0+(compatible;+bingbot/2.0;++http://www.bing.com/bingbot.htm)'
    user_agents = [baidu_ua,google_ua,sogou_ua,bing_ua]
    headers={
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
                'Connection': 'keep-alive',
                'Cache-Control':'no-cache',
                'Pragma':'no-cache'
            }
    return headers

def download(url):
    if url == None:
        return
    response =None
    for i in [1,2,3]:
        try:
            time.sleep(sleep_html)
            response=requests.get(url=url, headers=getHeader(), timeout=timeout)
            if response.status_code != 200:
                logger.warn('下载网页错误%d：%s'%(response.status_code,url))
                return
            else:
                return response.content.decode("utf-8")
        except :
            if i == 3 :
                logger.exception('下载网页错误：%s'%url)
                return
            else:
                time.sleep(5*(i-1))

def downloadBigFile(url,srcFile):
    f = None
    if os.path.exists(srcFile):
        return srcFile
    try:
        time.sleep(sleep_img)
        logger.info('下载图片%s，来源%s' % (srcFile,url))
        r = requests.get(url, headers=getHeader(), timeout=31, stream=True, allow_redirects=False)
        if r.status_code == 200:
            f = open(srcFile, 'wb')
            for chunk in r.iter_content(chunk_size=512*1024):
                if chunk:
                    f.write(chunk)
            f.close()
            return srcFile
        else:
            logger.warn('下载%d错误%s，来源%s' % (r.status_code,srcFile,url))
            return None
    except:
        logger.exception('下载错误%s，来源%s' % (srcFile,url))
        return None

def parseYear(html):
    soup  = BeautifulSoup(html, html_parer)
    typhoon_nodes = soup.find('table',class_='TABLELIST').find_all('a')
    typhoon_numbers = []
    for node in typhoon_nodes:
        if node.get_text().isnumeric():
            typhoon_numbers.append(node.get_text())
    return typhoon_numbers

def downImgs(typhoon_numbers):
    for number in typhoon_numbers:
        logger.info('%s台风图片列表：%s'% (number,typhoon_img_list % number))
        html = download(typhoon_img_list % number)
        soup  = BeautifulSoup(html, html_parer)
        img_nodes = soup.find('table',class_='TRACKINFO').find_all('a')
        for node in img_nodes:
            img_detail = typhoon_host + node['href']
            prefix = getValue(img_detail,'prefix')
            date = prefix[len(prefix)-6:]
            logger.info('%s-%s台风图片详情：%s'% (number,date,img_detail))
            img_html = download(img_detail)
            img_soup  = BeautifulSoup(img_html, html_parer)
            infos = img_soup.find_all('td',class_='META')
            hPa = '1000'
            kt = '0'
            for info in infos:
                if ' hPa' in info.get_text():
                    hPa = info.get_text().replace('hPa','').replace(' ','').replace('\n','')
                elif ' kt' in info.get_text():
                    kt = info.get_text().replace('kt','').replace(' ','').replace('\n','')
            downItem = {
                'file': '%s/%s.jpg' % (download_dir,'_'.join([number,date,kt,hPa])),
                'url': typhoon_img % (number,typhoon_img_type,prefix,number)
            }
            downloadBigFile(downItem['url'],downItem['file'])
        logger.info('%s台风图片共%d个下载完成' % (number,len(img_nodes)))


def getValue(url,key):
    query = url.split('?')[1]
    kvs = query.split('&')
    for kv in kvs:
        k_v = kv.split('=')
        if k_v[0]==key:
            return k_v[1]
    return None

def main():
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    for year in range(year_start, year_end+1):
        html_year = download(typhoon_year % year)
        typhoon_numbers = parseYear(html_year)
        logger.info('%d年台风列表，共计%d个，来源%s' % (year, len(typhoon_numbers), typhoon_year % year))
        downImgs(typhoon_numbers)
        logger.info('%d年台风图片下载全部完成' % year)
    pass

if __name__ == '__main__':
    main()
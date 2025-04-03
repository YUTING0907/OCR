from PIL import Image
import pytesseract
import cv2
import os
import string
import json
import re
import numpy as np
import heapq
import threading
import queue  # Python 3: Queue -> queue
import psutil
import multiprocessing
import time
import sys

'''
利用tesseract进行ocr身份证识别
'''
os.environ['TESSDATA_PREFIX'] = r"C:\Users\xxx\Anaconda3\envs\pytorch\share\tessdata"  # Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\xxx\Anaconda3\envs\pytorch\Library\bin\tesseract.exe"
# 单个图片识别item
class ImageRecognizerItem:
    def __init__(self, recognizedText, rect):
        self.rect = rect
        self.recognizedText = recognizedText
        self.dealedText = ""


# 身份证信息类
class IDcardInfo:
    def __init__(self):
        self.IDNumber = ""
        self.name = ""
        self.sex = ""
        self.birthDate = ""
        self.address = ""
        self.issueDate = ""
        self.expiryDate = ""
        self.authority = ""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class ThreadRecognize(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            virtualMemoryInfo = psutil.virtual_memory()
            availableMemory = virtualMemoryInfo.available
            if availableMemory > MEMORY_WARNING:
                args = self.queue.get()
                recognizeImage(*args)
                self.queue.task_done()


# 选择最大三个矩形框
def findIDcnt(contours):
    widths = [cv2.boundingRect(cnt)[2] for cnt in contours]
    top_3_widths = heapq.nlargest(3, widths)
    return [contours[widths.index(w)] for w in top_3_widths]


MEMORY_WARNING = 400 * 1024 * 1024
CPU_COUNT = multiprocessing.cpu_count() if multiprocessing.cpu_count() else 1
ENABLE_THREAD = True

IDrect = ()
recognizedItems = []
handledTexts = {}


def recognizeImage(results, cvimage, rect, language, charWhiteList=None):
    global IDrect
    if IDrect == rect:
        return
    config = "--psm 7"
    if charWhiteList:
        config += f" -c tessedit_char_whitelist={charWhiteList}"
    image = Image.fromarray(cvimage)
    result = pytesseract.image_to_string(image, lang=language, config=config)
    filtered_result = re.sub(r"[\W_]+", "", result)
    if language == "eng" and len(result) == 18:
        handledTexts["IDnumber"] = result
        IDrect = rect
    elif filtered_result:
        results.append(ImageRecognizerItem(filtered_result, rect))

def handlePersonalInfo():
    for item in reversed(recognizedItems):
        if item.recognizedText.startswith("姓名"):
            handledTexts["name"] = item.recognizedText[2:]
        elif item.recognizedText.isdigit() and int(item.recognizedText) > 10000000:
            recognizedItems.remove(item)
        elif item.recognizedText.startswith(("19", "20")):
            handledTexts["birthDate"] = item.recognizedText
        elif item.recognizedText.startswith("出生"):
            handledTexts["birthDate"] = item.recognizedText[2:]
        elif item.recognizedText.startswith("性别"):
            handledTexts["gender"] = item.recognizedText[2:]
        elif item.recognizedText.startswith("民族"):
            handledTexts["ethnic"] = item.recognizedText[2:]
        elif item.recognizedText.startswith("公民身份号码"):
            if "IDnumber" not in handledTexts:
                handledTexts["IDnumber"] = item.recognizedText[6:]
        elif item.recognizedText.startswith("住址"):
            handledTexts["address"] = item.recognizedText[2:]
        else:
            handledTexts["address"] += item.recognizedText[2:]


def main(path):
    handledTexts.update({
        "name": "", "birthDate": "", "gender": "", "ethnic": "", "IDnumber": "", "address": ""
    })

    '''
    if len(sys.argv) != 2:
        print(json.dumps({'code': 1001, 'data': '无效参数'}))
        exit(1)
    '''
    filePath = path
    img = cv2.imread(filePath, 0)
    img = cv2.resize(img, (1200, 900))

    # 二值化
    retval, binaryed = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 20))
    eroded = cv2.erode(binaryed, kernel)

    inverted = cv2.bitwise_not(eroded)

    contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    IDcnts = findIDcnt(contours)

    cv2.imshow("Processed Image", inverted)
    cv2.waitKey(0)

    queue_obj = queue.Queue()
    if ENABLE_THREAD:
        for _ in range(CPU_COUNT):
            t = ThreadRecognize(queue_obj)
            t.setDaemon(True)
            t.start()

    for IDcnt in IDcnts:
        x, y, w, h = cv2.boundingRect(IDcnt)
        rect = (x, y, w, h)
        IDimg = img[y:y + h, x:x + w]
        if ENABLE_THREAD:
            queue_obj.put((recognizedItems, IDimg, rect, "eng", "0123456789X"))
        else:
            recognizeImage(recognizedItems, IDimg, rect, "eng", "0123456789X")

    queue_obj.join()
    handlePersonalInfo()

    result = json.dumps(handledTexts, indent=4)
    print(json.dumps({'code': 1000, 'data': json.loads(result)}))
    cv2.destroyAllWindows()

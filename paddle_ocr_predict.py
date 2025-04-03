from paddlex import create_pipeline
import os
import re
import cv2
import numpy as np

def predict():
    directory = "D:/0000_Work/ai/identification_img/"
    pipeline = create_pipeline(pipeline='OCR')

    file_names = get_file_names(directory)
    id_regex = r'(\d{18,18}|\d{15,15}|\d{17,17}\d|X|x)'
    output = pipeline.predict(directory)
    for res in output:
        # res.print()
        input_path = res.get('input_path')
        file_name = input_path.split("/")[-1]
        rec_text = res.get("rec_text")
        combined_text = " ".join(str(item) for item in rec_text)  # 确保每个元素都是字符串
        matches = re.findall(id_regex, combined_text)
        #print(input_path)
        print(rec_text)
        # 输出匹配结果
        if matches:
            print(f"文件: {file_name}, 匹配到的身份证号码: {matches}")
        else:
            print(f"文件: {file_name}, 未匹配到身份证号码")

###############################
# 현재 1개의 레시피 추천까지 진행
##############################
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import threading
import numpy as np
import picamera

from tkinter import *
from tkinter import messagebox

from PIL import Image, ImageTk ###import 추가->mj
from tflite_runtime.interpreter import Interpreter
import tflite ###import 추가->mj
import pandas as pd

class Camera(threading.Thread):
  m_bPressed = False
  m_rFood ={} ######################카메라 고장으로 잠시 주석하고 테스트했음
  #m_rFood={'onion':1,'tomato':1} ##대신 테스트했던 코드
  m_bExit = False
  
  def __init__(self):
    super().__init__()

  def Pressed(self, bEnable) :
      self.m_bPressed = bEnable

  def load_labels(self, path):
    with open(path, 'r') as f:
      return {i: line.strip() for i, line in enumerate(f.readlines())}


  def set_input_tensor(self, interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def result(self):
    s = self.m_rFood.keys()

    return s

  def classify_image(self, interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    self.set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
      scale, zero_point = output_details['quantization']
      output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

  def exit(self):
    self.m_bExit = True

  def run(self):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    args = parser.parse_args()

    #labels = self.load_labels('labels.txt')
    #interpreter = Interpreter('model_unquant.tflite')
    labels = self.load_labels(args.labels)
    interpreter = Interpreter(args.model)

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    with picamera.PiCamera(resolution=(640, 480), framerate=3) as camera:
      camera.start_preview(fullscreen=False, window=(100,100,640,480) )
      try:
        stream = io.BytesIO()
        for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
      
          stream.seek(0)
          image = Image.open(stream).convert('RGB').resize((width, height),Image.ANTIALIAS)
          start_time = time.time()
          results = self.classify_image(interpreter, image)

          elapsed_ms = (time.time() - start_time) * 1000
          label_id, prob = results[0]

          stream.seek(0)
          stream.truncate()
          camera.annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob,
                                                        elapsed_ms)

          if self.m_bPressed :
            self.m_rFood[labels[label_id]]=label_id
            #self.m_rFood.append(labels[label_id])
            print ( labels[label_id] )
            self.m_bPressed = False
            camera.annotate_text = '%s added.' % (labels[label_id] )

          if self.m_bExit:
            break
              
      finally:
        camera.stop_preview()

cam = Camera()

def pressed_button() :
    cam.Pressed(True)
    print('button pressed')

def show_result():
    msg = messagebox.showinfo('result',cam.result())
    print('show')
def pressed_recipe(text):
    print(text)

def show_recipe():
    source = cam.result()
    data= pd.read_csv('food_Ingredients.csv')
    print(data)

    recipe_ingredient=data['food ingredients']
    ingredients=[]
    count=[0 for _ in range(len(data))]
    print("data 갯수 %d" %len(data))
    recommend=[]
    for num,i in enumerate(recipe_ingredient):
        ingredients.append(i.split(','))
    for num,a in enumerate(ingredients):
        print(a)
        for s in source:
            if s in a:
                count[num]+=1 # 나중에 딕셔너리형으로 바꾸는거 고려
    print(max(count))
    for num,cnt in enumerate(count):
        if max(count)==0:
            print("해당재료로 만들 수 있는 레시피 없음.")
        if cnt==max(count) :
            recommend.append(data['cook'][num])
    print(recommend)
    
    recipe = Tk()
    recipe.configure(bg='white')
    recipe.geometry('750x600+100+100')
    recipe.title(recommend[0])
    l = Label(recipe, bg="white", text='레시피 추천', font=13) #요리제목
    l.place(x=30, y=10)
    
    image1 = Image.open("image/1_%s_2.jpg"%recommend[0])
    image2 = Image.open("image/1_%s_3.jpg"%recommend[0])

    image1 = image1.resize((350,500),Image.ANTIALIAS)
    image2 = image2.resize((350,500),Image.ANTIALIAS)

    img1 = ImageTk.PhotoImage(image1,master=recipe)  
    img2 = ImageTk.PhotoImage(image2,master=recipe)

    label1 = Label(recipe, image= img1)
    label2 = Label(recipe, image= img2)
    
    # # Position image
    label1.place(x=20, y=60)
    label2.place(x=370,y=60)

    btn={}
    for num,a in enumerate(recommend):
          btn[a] = Button(recipe, width=20, height=1, text=a, bg="yellow", command=lambda: pressed_recipe(a))
          btn[a].place(x=130+150*num, y=10)
    print(btn)
    recipe.mainloop()

    return recommend


def main():
    try:
        cam.start()
        window = Tk()
        window.geometry('250x250+750+300')
        window.title('AI CHEF')

        l = Label(window, bg="white", text='Capturing the Image', font=13)
        l.place(x=30, y=10)
  
        b1 = Button(window, width=10, height=2, text='CAPTURE', bg="magenta", command=pressed_button)
        b1.place(x=65, y=60)

        b2 = Button(window, width=10, height=2, text='RESULT', bg="pink", command=show_result)
        b2.place(x=65, y=120)

        b3 = Button(window, width=10, height=2, text='RECIPE', bg="blue", command=show_recipe)
        b3.place(x=65, y=180)
   
        window.mainloop()

    finally:
        cam.exit()
        cam.join()
        quit()
    return

if __name__ == '__main__':
    main()

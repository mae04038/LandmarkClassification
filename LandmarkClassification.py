import numpy as np
import tensorflow as tf
import googlemaps
import folium
import webbrowser
import cv2
import time
from selenium import webdriver
import urllib
from urllib.parse import quote_plus
from urllib.request import urlopen
from googletrans import Translator

imagePath = 'c:/tmp/imageForTest.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath = 'c:/tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = 'c:/tmp/output_labels.txt'                                   # 읽어들일 labels 파일 경로

#Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow('Show a Image ',img)
    k = cv2.waitKey(125)
    # Specify the countdown
    j = 50
    # set the key for the countdown to begin
    if k == ord('q'):
        while j>=10:
            ret, img = cap.read()
            # Display the countdown after 10 frames so that it is easily visible otherwise,
            # it will be fast. You can set it to anything or remove this condition and put 
            # countdown on each frame
            if j%10 == 0:
                # specify the font and draw the countdown using puttext
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,str(j//10),(250,250), font, 7,(255,255,255),10,cv2.LINE_AA)
            cv2.imshow('Show a Image',img)
            cv2.waitKey(125)
            j = j-1
        else:
            ret, img = cap.read()
            # Display the clicked frame for 1 sec.
            # You can increase time in waitKey also
            cv2.imshow('Show a Image',img)
            cv2.waitKey(1000)
            # Save the frame
            cv2.imwrite('C:/tmp/imageForTest.jpg',img)
    # Press Esc to exit
    elif k == 27:
        break
cap.release()
cv2.destroyAllWindows()


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        # 추론할 이미지를 인풋으로 넣고 추론 결과인 소프트 맥스 행렬을 가져옵니다. 
        predictions = sess.run(softmax_tensor, feed_dict={'DecodeJpeg/contents:0': image_data})
        # 불필요한 차원을 제거합니다.
        predictions = np.squeeze(predictions)

        # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)들의 인덱스를 가져옵니다.
        # e.g. [0 3 2 4 1]]
        top_k = predictions.argsort()[-5:][::-1]
 
        # output_labels.txt 파일로부터 정답 레이블들을 list 형태로 가져옵니다.
        f = open(labelsFullPath, 'r')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        # 가장 높은 확률을 가진 인덱스들부터 추론 결과(Top-5)를 출력합니다.
        print("Top-5 추론 결과:")
        for node_id in top_k:
            label_name = labels[node_id]
            probability = predictions[node_id]
            print('%s (확률 = %.5f)' % (label_name, probability))

        # 가장 높은 확류을 가진 Top-1 추론 결과를 출력합니다.
        print("\nTop-1 추론 결과:")
        answer = labels[top_k[0]]
        probability = predictions[top_k[0]]
        print('%s (확률 = %.5f)' % (answer, probability))

        create_map(answer)

        
def create_map(landmark):
    
    with tf.Session() as sess:
        
        tmp = landmark
        name = tmp.replace("'", "")

        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")
       
        driver = webdriver.Chrome('C:/Users/Heeseung/Downloads/chromedriver_win32/chromedriver.exe', chrome_options=options)
        tr = Translator()
        tr_result = tr.translate(name, src='en', dest = 'ko')
        kr_result = tr_result.text
        korean = kr_result.replace(" ", "")

        wikiURL = 'https://ko.wikipedia.org/wiki/'
        searchURL = wikiURL + urllib.parse.quote(korean)

        driver.get(searchURL)
        search = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div/p[1]')
        data = search.text
        print(data)
        
        mykey = 'AIzaSyAK3hHRxscu-9FwtacVTeyKMu-umMXuflo'
        gmaps = googlemaps.Client(key = mykey)
        
        geo = gmaps.geocode(landmark)
        lat_long = [geo[0]['geometry']['location']['lat'],geo[0]['geometry']['location']['lng']]        

        landmark = folium.Map(location=lat_long, zoom_start = 50)
        html = """
            <a href="https://www.klook.com/ko/" target="_blank"> Details.</a>
            """
        iframe = folium.IFrame(html = data + html, width=400, height = 150)
        popup = folium.Popup(iframe, max_height = 650)
        
        folium.Marker(lat_long, popup = popup, icon = folium.Icon(icon='cloud')).add_to(landmark)
        
        svFilename = 'c:/tmp/landmark.html'
        landmark.save(svFilename)
        webbrowser.open(svFilename)
        
if __name__ == '__main__':
    run_inference_on_image()

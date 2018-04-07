import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def predict(image_data):
    global consonant
    consonant = {'giyeok':1, 'nieun': 2, 'diguek':3, 'rieul': 4, 'mieum':5, 
                'bieup':6, 'sieuh':7,'o': 8, 'jieuh': 9, 'chieuh':10, 'kieuk':11,
                'tiguek':12, 'pieup':13, 'hieuh':14}
    global vowel
    vowel = {'ah': 16, 'ya':17, 'uh':18, 'yeo': 19, 'oh':20, 'yo':21,
            'woo':22, 'you':23, 'euh':24, 'yee':25}
    global korean_dict
    korean_dict = {'안': [8, 16, 2], '녕':[2, 19, 8], '년':[2, 19, 2], '난':[2,16,2]}

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("C:/tmp/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("C:/tmp/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0

    cap = cv2.VideoCapture(1)

    res, score = '', 0.0
    i = 0
    key = list()
    cnt = 0
    mem = ''
    consecutive = 0
    sequence = ''
    tmp = ''
    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            x1, y1, x2, y2 = 100, 100, 300, 300
            img_cropped = img[y1:y2, x1:x2]

            c += 1
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            a = cv2.waitKey(33)
            if i == 4:
                res_tmp, score = predict(image_data)
                res = res_tmp
                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'space':
                        sequence += ' '
                    elif res == 'del':
                        sequence = sequence[:-1]
                    else:
                        sequence += res + " "
                        cnt += 1
                    if cnt == 3:
                        cnt = 0
                        sequence = sequence.split()
                        for j in range(len(sequence)):
                            try:
                                if j is 0:   
                                    key.append(consonant[sequence[j]])
                                elif j is 1:
                                    key.append(vowel[sequence[j]])
                                else:
                                    key.append(consonant[sequence[j]])
                            except:
                                #cv2.VideoCapture(0).release()
                                break
                        check = 0

                        for write, syllable in korean_dict.items():
                            try:
                                if key[0] == syllable[0]:
                                    check += 1
                                if key[1] == syllable[1]:
                                    check += 1 
                                if key[2] == syllable[2]:
                                    check += 1
                                if check == 3:
                                    print(write),
                                    break
                                else:
                                    check = 0
                            except:
                                break                                         
                        key.clear()
                        sequence = ''
                    consecutive = 0
                elif a == 27:
                    cv2.VideoCapture(0).release()
            ' '.join(sequence)
            i += 1
            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("img", img)
            img_sequence = np.zeros((200,1200,3), np.uint8)
            cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('sequence', img_sequence)
        else:
            break
"""
 Face Recognition from Video (라이브러리를 이용하여 동영상에서 얼굴인식)

 facenet-pytorch,
 mtcnn,
 
 활용한 비디오로부터 opencv를 이용한 얼굴인식 추출

    from. http://machinelearningkorea.com/2020/01/27/face-recognition-from-video-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%B9%84%EB%94%94%EC%98%A4%EB%A1%9C%EB%B6%80%ED%84%B0-%EC%96%BC%EA%B5%B4%EC%9D%B8/
"""
import cv2
import numpy as np
from tqdm import tqdm_notebook as tqdm # python 로딩-bar 만들기
from matplotlib import pyplot as plt
from PIL import Image
import torch
import time  # 시간 측정용

'''
 random mp4 video convert to images.
 고정된 간격의 이미지를 추출하고 numpy 행렬로 변경
'''
video_path = '../sample_video/'
reader = cv2.VideoCapture(video_path + 'sample.mp4')

image_list = []
for i in tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))):  # tqdm : 진행률 표시
    _, image = reader.read()                                      # 사용법은 기존 for문의 range범위를 () 감싸기만 하면됨.
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)           
    image_list.append(image)

reader.release()    # 메모리 비우기
images_np = np.stack(image_list)
print(images_np.shape)


'''
 ploating faces (플롯 그리기) .ipydb용 matplot
'''
# def plot_faces(images, figsize=(10.8/2, 19.2/2)):
#     shape = images[0].shape
#     images = images[np.linspace(0, len(images)-1, 16).astype(int)] # np.linspace() 1차원 배열 생성, 16개 만큼

#     im_plot = []
#     for i in range(0, 16, 4):
#         im_plot.append(np.concatenate(images[i:i+4], axis=0)) # 이미지 4개씩 붙이기
    
#     im_plot = np.concatenate(im_plot, axis=1)


'''
 timer for calculating detector speed 
 (각각의 라이브러리 속도 측정을 위한 타이머 설정)
'''
def timer(detector, detect_F, images, *args):
    start = time.time()
    faces = detect_F(detector, images, *args)  # detect-Function 불러오기!
    elapsed = time.time() - start
    print(f' , {elapsed:.3f} seconds')
    return faces, elapsed

'''
 gpu setting
'''
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


'''
 pytorch-facenet 이용한 detect faces

 batch processing 이기 때문에, 메모리 사용량이 순간적으로 6G 까지 올라감
 https://pypi.org/project/facenet-pytorch/
'''
from facenet_pytorch import MTCNN
detector = MTCNN(device=device, post_process=False)

def detect_facenet_pytorch(detector, images, batch_size):
    faces = []
    for lb in np.arange(0, len(images), batch_size):
            imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]   # numpy 배열을 Image 객체로 바꿀때는 fromarray() 클래스 이용
            faces.extend(detector(imgs_pil)) # append와 extend 차이점!
    return faces  # 아마 이 faces 가 리스트 형태로 되어있어서 이부분을 enumerate로 저장하면 될꺼야


times_facenet_pytorch = []  # batched
times_facenet_pytorch_nb = []   # not-batched

print('Detecting faces in frames', end='')
faces, elapsed = timer(detector, detect_facenet_pytorch, image_list, 20)
times_facenet_pytorch.append(elapsed)
# plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy()) # torch.permute() 는 pytorch 상에서 tf.transpose 차원 순서 변환!
                                                                 # https://devbruce.github.io/machinelearning/ml-05-np_torch_summary/
del detector
torch.cuda.empty_cache()


'''
 Detecting face with MTCNN library 
 (MTCNN을 활용한 얼굴인식)

 특별한 문제는 없지만, 속도가 매우 느리다
'''
# from mtcnn import MTCNN
# detector_mtcnn = MTCNN()

# def detect_mtcnn(detector_mtcnn, images):
#     faces = []
#     for image in images:
#         boxes = detector_mtcnn.detect_faces(image)
#         box = boxes[0]['box']
#         face = image[box[1]:box[3] + box[0]:box[2] + box[0]]
#         faces.append(face)
#     return faces


    




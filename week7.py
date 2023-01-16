import cv2
import numpy as np
import math

from cv2 import KeyPoint


def my_padding(src, filter):
    (h, w) = src.shape
    if isinstance(filter, tuple):
        (h_pad, w_pad) = filter
    else:
        (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h + h_pad * 2, w + w_pad * 2))
    padding_img[h_pad:h + h_pad, w_pad:w + w_pad] = src

    # repetition padding
    # up
    padding_img[:h_pad, w_pad:w_pad + w] = src[0, :]
    # down
    padding_img[h_pad + h:, w_pad:w_pad + w] = src[h - 1, :]
    # left
    padding_img[:, :w_pad] = padding_img[:, w_pad:w_pad + 1]
    # right
    padding_img[:, w_pad + w:] = padding_img[:, w_pad + w - 1:w_pad + w]

    return padding_img


def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape

    # filter 확인
    # print('<filter>')
    # print(filter)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + f_h, col:col + f_w] * filter)

    return dst


def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    return sobel_x, sobel_y


def calc_derivatives(src):
    # calculate Ix, Iy
    sobel_x, sobel_y = get_my_sobel()
    Ix = my_filtering(src, sobel_x)
    Iy = my_filtering(src, sobel_y)
    return Ix, Iy


def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h + ksize, w + ksize))
    pad_img[ksize // 2:h + ksize // 2, ksize // 2:w + ksize // 2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row: row + ksize, col:col + ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]

    return dst


def SIFT(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print("get keypoint")
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)
    dst[dst < 0.01 * dst.max()] = 0
    dst = find_local_maxima(dst, 21)
    dst = dst / dst.max()

    # harris corner에서 keypoint를 추출
    # np.nonzero : Return the indices of the elements that are non-zero.
    y, x = np.nonzero(dst)

    #########################################################################
    # Keypoint에 대한 정보를 Keypoint class의 객체로서 저장(Matching을 할때 사용하기 위함)
    # Arguments 정보
    # pt_x, pt_y : keypoint의 x,y좌표
    # size : keypoint 직경(과제에서는 사용하지 않음 None 설정)
    # key_angle : keypoint의 orientation(histogram을 사용해서 구함)
    # response : 위의 주어진 keypoint 좌표에서의 값이 keypoint(본 과제에서는 harris response를 사용)
    # octave : scale의 변화를 확인하기 위한 값(과제에서는 사용하지 않음, 즉 scale을 고려하지 않음)
    # class id : object id (-1로 설정)
    #########################################################################
    keypoints = []
    for i in range(len(x)):
        # x, y, size, angle, response, octave, class_id
        pt_x = int(x[i])  # point x
        pt_y = int(y[i])  # point y
        size = None
        key_angle = -1.
        response = dst[y[i], x[i]]  # keypoint에서 harris corner의 측정값
        octave = 0  # octave는 scale 변화를 확인하기 위한 값 (현재 과제에서는 사용안함)
        class_id = -1
        keypoints.append(KeyPoint(pt_x, pt_y, size, key_angle, response, octave, class_id))

        #################################################
        # 참고 : 주석 풀어서 사용 할 것 (디버깅을 통해서 하면 더 간편)
        # Keypoint 객체 속성 값 확인
        # Keypoints[0].pt
        # Keypoints[0].size
        # Keypoints[0].angle
        # Keypoints[0].response
        # Keypoints[0].octave 등등
        # print(keypoints[0].pt)
        # print(keypoints[0].angle)
        # print(keypoints[0].response)
        # print(keypoints[0].octave)
        #################################################

    print('keypoints counts :  {}'.format(len(keypoints)))
    print('get Ix and Iy...')
    Ix, Iy = calc_derivatives(gray)

    print('calculate angle and magnitude')

    ##########################################
    # Todo
    # magnitude / orientation 계산
    ##########################################
    magnitude = np.sqrt((Ix ** 2) + (Iy ** 2))
    angle = np.arctan2(Iy, Ix)  # radian 값
    angle = np.rad2deg(angle)  # radian 값을 degree로 변경 > -180 ~ 180도로 표현
    angle = (angle + 360) % 360  # -180 ~ 180을 0 ~ 360의 표현으로 변경

    # keypoint 방향
    print('calculate orientation assignment')

    num = 0  # 추가된 keypoint 개수
    for i in range(len(keypoints)):
        x, y = keypoints[i].pt
        orient_hist = np.zeros(36, )
        for row in range(-8, 8):
            for col in range(-8, 8):
                p_y = int(y + row)
                p_x = int(x + col)
                if p_y < 0 or p_y > src.shape[0] - 1 or p_x < 0 or p_x > src.shape[1] - 1:
                    continue  # 이미지를 벗어나는 부분에 대한 처리
                gaussian_weight = np.exp((-1 / 16) * (row ** 2 + col ** 2))
                orient_hist[int(angle[p_y, p_x] // 10)] += magnitude[p_y, p_x] * gaussian_weight

        ###################################################################
        ## ToDo
        ## orient_hist에서 가중치가 가장 큰 값을 추출하여 keypoint의 angle으로 설정
        ## orient_hist에서 가장 큰 가중치의 0.8배보다 큰 가중치의 값도 keypoint의 angle로 설정
        ## 즉 같은 keypoint에 대한 angle정보가 2개이상 있을 수 있음
        ## 이러한 정보 또한 KeyPoint 클래스를 사용하여 저장
        ## angle은 0 ~ 360도의 표현으로 저장해야 함
        ## np.max, np.argmax를 활용하면 쉽게 구할 수 있음
        ## keypoints[i].angle = ???
        ###################################################################

        # 가장 큰 가중치값의 angle 저장
        ##------------------------------------------------------
        ##------------------------------------------------------
        keypoints[i].angle = float(np.argmax(orient_hist) * 10)

        for j in range(36): #방향
            if (j != np.argmax(orient_hist)) and (orient_hist[j] > orient_hist[int(np.argmax(orient_hist))] * 0.8):
                keypoints.append(KeyPoint(x, y, size, j * 10, response, octave, class_id))


        # orient_hist에서 가장 큰 가중치의 0.8배보다 큰 가중치의 값도 keypoint의 angle로 설정하여
        # KeyPoint의 클래스의 객체로 저장
        # 예시
        # keypoints.append(KeyPoint(pt_x, pt_y, size, key_angle, response, octave, class_id)

        # 위의 과정이 올바르게 구현 되었는지 확인
        # 160개
    print("총 key point 개수 : {}".format(len(keypoints)))

    print('calculate descriptor')

        # 8 orientation * 4 * 4 = 128 dimensions
    descriptors = np.zeros((len(keypoints), 128))
    descriptors_angle = np.zeros((angle.shape[0], angle.shape[1]))  # 회전한 각도를 넣어주기 위한 2차원 np array

    for i in range(len(keypoints)):
        x, y = keypoints[i].pt
        theta = np.deg2rad(keypoints[i].angle)  # 호도법 -> radian
        # 키포인트 각도 조정을 위한 cos, sin값
        cos_angle = np.cos(theta)
        sin_angle = np.sin(theta)

        other_orient_hist = np.zeros(128, )  # point의 방향을 저장

        #######################################
        # Keypoint을 기준으로 16 x 16 patch를 추출
        # 2중 for문을 돌면서 16 x 16 window내의 모든 점을 순회 ->  [-8, 8)
        for row in range(-8, 8):
            for col in range(-8, 8):
                # 회전을 고려한 point값을 얻어냄
                row_rot = np.round((cos_angle * col) + (sin_angle * row))
                col_rot = np.round((cos_angle * col) - (sin_angle * row))

                p_y = int(y + row_rot)
                p_x = int(x + col_rot)
                if p_y < 0 or p_y > (src.shape[0] - 1) \
                        or p_x < 0 or p_x > (src.shape[1] - 1):
                    continue
                ###################################################################
                ## ToDo
                ## descriptor을 완성
                ## 회전 변환된 윈도우 좌표에 대해서 angle histogram을 구함
                ## descriptor angle : angle[p_y, p_x] - keypoints[i].angle
                ## descriptor angle을 8개의 orientation( 0 ~ 7 labeling)으로 표현
                ## 4×4의 window에서 8개의 orientation histogram으로 표현
                ## 최종적으로 128개 (8개의 orientation * 4 * 4)의 descriptor를 가짐
                ## gaussian_weight = np.exp((-1 / 16) * (row_rot ** 2 + col_rot ** 2))
                ###################################################################

                '''
                
                '''
                # descriptor angle 값
                descriptors_angle[p_y, p_x] = angle[p_y, p_x] - keypoints[i].angle

                # 조정된 angle 값 : 0 ~ 7 범위의 값 가짐

                # 128-dimension에서 어떤 index에 넣어줄 것인지를 결정
                # row                      col          총 8개
                hist_index = ((int((row + 8) // 4) * 4) + int((col + 8) // 4)) * 8

                gaussian_weight = np.exp((-1 / 16) * (row_rot ** 2 + col_rot ** 2))

                # 추가된 keypoint 계산과 비슷하게 진행하지만 hist_index 를 추가한다
                other_orient_hist[hist_index + int(descriptors_angle[p_y, p_x] // 45)] += magnitude[p_y, p_x] * gaussian_weight

                # descriptors에 값 저장
                descriptors[i] = other_orient_hist

    return keypoints, descriptors

def main():
    src = cv2.imread("zebra.png")
    src_rotation = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)

    # 회전된 이미지 확인하고 싶으면 주석 풀어서 확인해 볼 것
    #cv2.imshow('rotation image', src_rotation)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    kp1, des1 = SIFT(src)
    kp2, des2 = SIFT(src_rotation)

    ## Matching 부분 ##
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    des1 = des1.astype(np.uint8)
    des2 = des2.astype(np.uint8)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(src, kp1, src_rotation, kp2, matches[:20], outImg=None, flags=2)

    # 결과의 학번 작성하기!
    cv2.imshow('my_sift_201702086', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
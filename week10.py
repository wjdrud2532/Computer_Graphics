import numpy as np
import cv2
import random
from cv2 import DMatch
import math
from math import floor

################################################################################
#########################       9주차 코드 그대로
################################################################################
def fit_coordinates(src, M):
    h, w, _ = src.shape
    cor_transform = []
    for row in range(h + 1):
        for col in range(w + 1):
            P = np.array([[col],[row],[1]])
            P_dst = np.dot(M, P)  # (x,y,1) vector와 Translation matrix를 곱함
            dst_col = P_dst[0][0]  # x
            dst_row = P_dst[1][0]  # y
            cor_transform.append((dst_row, dst_col))
    cor_transform = list(set(cor_transform))  # 중복제거
    cor_transform = np.array(cor_transform)
    row_max = np.max(cor_transform[:, 0])
    row_min = np.min(cor_transform[:, 0])
    col_max = np.max(cor_transform[:, 1])
    col_min = np.min(cor_transform[:, 1])
    return row_max, row_min, col_max, col_min

def backward(src, M):
    h, w, c = src.shape
    M_inv = np.linalg.inv(M)
    row_max, row_min, col_max, col_min = fit_coordinates(src, M)
    h_ = round(row_max - row_min)
    w_ = round(col_max - col_min)
    dst = np.zeros((h_, w_, c))
    for row in range(h_):
        for col in range(w_):
            P_dst = np.array([[col + col_min],[row + row_min],[1]])
            P = np.dot(M_inv, P_dst)
            src_col = P[0, 0]
            src_row = P[1, 0]
            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)
            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)
            if src_col_right >= w or src_row_bottom >= h or src_col_left < 0 or src_row_top < 0:
                continue
            s = src_col - src_col_left
            t = src_row - src_row_top
            intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left, :] \
                        + s * (1 - t) * src[src_row_top, src_col_right, :] \
                        + (1 - s) * t * src[src_row_bottom, src_col_left, :] \
                        + s * t * src[src_row_bottom, src_col_right, :]
            dst[row, col, :] = intensity
        dst = dst.astype(np.uint8)
    return dst

def my_ls(matches, kp1, kp2):
    A = []
    b = []
    for idx, match in enumerate(matches):
        trainInd = match.trainIdx
        queryInd = match.queryIdx
        x, y = kp1[queryInd].pt
        x_prime, y_prime = kp2[trainInd].pt
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        b.append([x_prime])
        b.append([y_prime])
    A = np.array(A)
    b = np.array(b)
    try:
        X = np.dot(np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T, b) )
    except:
        print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
        X = None
    return X

def L2_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1-vector2)**2))

def feature_matching_RANSAC(img1, img2, keypoint_num=None, iter_num=500, threshold_distance=5):
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)
    matches_shuffle = matches.copy()
    inliers = []
    M_list = []
    temp_inlier_cnt = -1
    for i in range(iter_num):
        print('\rcalculate RANSAC ... %d ' % (int((i + 1) / iter_num * 100)) + '%', end='\t')
        random.shuffle(matches_shuffle)
        three_points = matches_shuffle[:3]
        X = my_ls(three_points, kp1, kp2)
        if X is None:
            continue
        M = np.array([[X[0][0], X[1][0], X[2][0]],[X[3][0], X[4][0], X[5][0]],[0, 0, 1]])
        M_list.append(M)
        count_inliers = 0
        for idx, match in enumerate(matches):
            trainInd = match.trainIdx
            queryInd = match.queryIdx
            x, y = kp1[queryInd].pt
            x_prime, y_prime = M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]
            x_hat, y_hat = kp2[trainInd].pt
            dist = L2_distance(vector1=np.array([x_prime, y_prime]), vector2=np.array([x_hat, y_hat]))
            if dist < threshold_distance:
                count_inliers += 1
        inliers.append(count_inliers)
        if (count_inliers > temp_inlier_cnt):
            temp_inlier_cnt = count_inliers
    best_M = M_list[inliers.index(temp_inlier_cnt)]
    print(best_M)
    result = backward(img1, best_M)
    return result.astype(np.uint8)

################################################################################
#########################       9주차 코드 그대로
################################################################################

def get_matching_keypoints(img1, img2, keypoint_num):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: 추출한 keypoint의 수
    :return: img1의 특징점인 kp1, img2의 특징점인 kp2, 두 특징점의 매칭 결과
    '''
    sift = cv2.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    ##########################################
    # TODO Brute-Force Feature Matching 구현
    ##########################################

    my_matches = my_feature_matching(des1, des2)

    ############################################################
    # TODO TEST 내장 함수를 사용한 것과 직접 구현 것의 결과 비교
    # 다음 3가지 중 하나라도 통과하지 못하면 잘못 구현한것으로 판단하여 프로그램 종료
    # 오류가 없다면 "매칭 오류 없음" 출력
    bf = cv2.BFMatcher_create(cv2.NORM_L2)
    matches = bf.match(des1, des2)
    # 1. 매칭 개수 비교
    assert len(matches) == len(my_matches)
    # 2. 매칭 점 비교
    for i in range(len(matches)):
        if (matches[i].trainIdx != my_matches[i].trainIdx) \
                or (matches[i].queryIdx !=
                                        my_matches[i].queryIdx):
            print("matching error")
            return

    # 3. distance 값 비교
    for i in range(len(matches)):
        if int(matches[i].distance) != int(my_matches[i].distance):
            print("distance calculation error")
            return

    print("매칭 오류 없음")
    ##########################################################

    # DMatch 객체에서 distance 속성 값만 가져와서 정렬
    my_matches = sorted(my_matches, key=lambda x: x.distance)
    # 매칭된 점들 중 20개만 가져와서 표시
    result = cv2.drawMatches(img1, kp1, img2, kp2, my_matches[:20], outImg=None, flags=2)

    #cv2.imshow('My BF matching result', result)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    return kp1, kp2, my_matches

def my_feature_matching(des1, des2):
    ##########################################
    # TODO Brute-Force Feature Matching 구현
    # matches: cv2.DMatch의 객체를 저장하는 리스트
    # cv2.DMatch의 배열로 구성
    # cv2.DMatch:
    # trainIdx: img1의 kp1, des1에 매칭되는 index
    # queryIdx: img2의 kp2, des2에 매칭되는 index
    # kp1[queryIdx]와 kp2[trainIdx]는 매칭된 점
    # return 값 : matches
    ##########################################

    #cv.Dmatch - 4개의 속성을 갖는 객체, 매칭 결과를 표현한다
    #queryIdx와 trainIdx로 두 이미지의 어느 지점이 서로 매칭 되었는지
    #distnace로 얼마나 가까운 거리 인지도 알 수 있다

    matches = list()

    #블루트 포스 시작
    for i in range(len(des1)):

        #매칭을 통과하는 가장 작은 값, 그냥 해본거
        threshold = 435

        for j in range(len(des2)):
            #des1, des2에 대해 Brute-Force 탐색

            #각 좌표에 대한 거리를 구하고
            dist = L2_distance(des1[i], des2[j])

            #설정한 threshold보다 값이 작을 때
            #queryIdx trainIdx 를 현재 좌표로 설정
            if threshold > dist:
                threshold = dist
                queryIdx = i
                trainIdx = j
        # Dmatch list에 값을 추가              이미지 인덱스에는 아무거나 상관없음
        #i, j좌표에서 가장
        matches.append(DMatch(queryIdx, trainIdx, 122, threshold))

    return matches

def feature_matching_gaussian(img1, img2, keypoint_num=None, iter_num=500, threshold_distance=5):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: sift에서 추출할 keypoint의 수
    :param iter_num: RANSAC 반복횟수
    :param threshold_distance: RANSAC에서 inlier을 정할때의 거리 값
    :return: RANSAC을 이용하여 변환 된 결과
    '''
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)
    matches_shuffle = matches.copy()
    inliers = []
    M_list = []
    temp_inlier_cnt = -1

    for i in range(iter_num):
        print('\rcalculate gaussian ... %d ' % (int((i + 1) / iter_num * 100)) + '%', end='\t')
        random.shuffle(matches_shuffle)
        three_points = matches_shuffle[:3]
        X = my_ls(three_points, kp1, kp2)
        if X is None:
            continue
        M = np.array([[X[0][0], X[1][0], X[2][0]],
                      [X[3][0], X[4][0], X[5][0]],
                      [0, 0, 1]])
        M_list.append(M)
        count_inliers = 0
        for idx, match in enumerate(matches):
            trainInd = match.trainIdx
            queryInd = match.queryIdx

            kp1_x, kp1_y = kp1[queryInd].pt
            kp2_x, kp2_y = kp2[trainInd].pt

            x, y = kp1[queryInd].pt
            x_prime, y_prime = M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]
            x_hat, y_hat = kp2[trainInd].pt

            dist = L2_distance(vector1=np.array([x_prime, y_prime]), vector2=np.array([x_hat, y_hat]))

            if dist < threshold_distance:
                count_inliers += 1

        #if (count_inliers > temp_inlier_cnt):
        #    temp_inlier_cnt = count_inliers

        inliers.append(count_inliers)
        ######################
        if (count_inliers > temp_inlier_cnt):
            temp_inlier_cnt = count_inliers
        #M_list.append(M)

    best_M = M_list[inliers.index(temp_inlier_cnt)]
    #best_M = M_list[np.argmax(inliers)]

    print(best_M)

    '''
    잘 나올 때
[[  1.52885076  -0.66353417 169.43142044]
 [  1.34250396   1.90677187   3.07800667]
 [  0.           0.           1.        ]]

[[  1.53181988  -0.67183684 170.07652793]
 [  1.34916507   1.92774226  -1.70565929]
 [  0.           0.           1.        ]]

[[  1.53171937  -0.65999007 169.52222532]
 [  1.34680531   1.92341496   0.50020531]
 [  0.           0.           1.        ]]
    '''

    result = backward_gaussian(img1, img2, best_M)
    return result.astype(np.uint8)

def backward_gaussian(img1, img2, M):
    h, w, c = img2.shape
    h1, w1, c1 = img1.shape

    #결과값을 위한 zeros 공간 생성
    dst = np.zeros((h, w, c))

    #가우시안 적용을 위해 마스크 생성
    Gaussian_mask = my_get_Gaussian_mask((3, 3), 3)

    #어차피 공간 확보만 하면 되기 때문에 padding 종류 상관없이 둘 다 된다
    #zero가 더 빨라서 그냥 사용
    pad_img = my_padding(img1, Gaussian_mask, 'zero')
    #pad_img = my_padding(img1, gaus_mask, 'repetition')

    for row in range(h):
        for col in range(w):
            #img2의 모든 좌표를 돈다

            #현재 좌표에 대한 정보를 가져온다
            #곱셈 크기를 맞추기 위해 역행렬로 바꾼다
            xy_T = np.array([[col, row, 1]]).T

            #M의 역행렬과 xy_T를 곱함
            xy = (np.linalg.inv(M)).dot(xy_T)


            x_ = xy[0, 0]
            y_ = xy[1, 0]
            #xy[2, 0]은 항상 1, 크기를 위한 쓰래기값

            #img범위 안에 존재하는 경우
            if x_ > 0 and y_ > 0 and (x_ + 1) < w1 and (y_ + 1) < h1:
                temp_dst = 0

                # 실습 ppt 11페이지 공식 그대로
                # 필터 크기만큼 돌면서 반복
                for i in range(3):
                    for j in range(3):
                        #쌍선형 보간법 적용
                        temp_dst += bilinear_interpolation(pad_img, x_ + i, y_ + j) * Gaussian_mask[i, j]

                #추가
                dst[row, col, ] = temp_dst

    return dst

#영상처리 가우시간 2D
''' 
def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[-int(msize/2) : int(msize/2) + 1, -int(msize/2) : int(msize/2) + 1]
        #−𝑛 ~ 𝑛 범위의 mask에서의 x좌표
    

    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]


    # 2차 gaussian mask 생성
    gaus2D = (np.exp ( - (x**2 + y**2) / (2 * sigma**2))) / (2*np.pi*sigma**2)
            # e의 지수함수로 마이너스 2*시그마 제곱 분에 (x제곱 * y제곱)

    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    #총 합을 1로 만든다.
    return gaus2D
'''

def my_get_Gaussian_mask(fshape, sigma=1):

    (f_h, f_w) = fshape

    y, x = np.mgrid[-(f_h // 2) : (f_h // 2) + 1, -(f_w // 2) : (f_w // 2) + 1]
    #가우시안 공식 그대로 적용    스칼라                                익스포넨셜
    mask = (1 / (2 * np.pi * sigma ** 2)) * (np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2))))
    Gaussian_filter = mask / np.sum(mask)
    return Gaussian_filter

#영상처리 강의 my_padding 사용
def my_padding(src, filter, pad_type='zero'):
    (h, w, c) = src.shape
    #일반적인 2D padding과 같고 차원 c만 추가함

    (p_h, p_w) = filter.shape

    #print(p_h)
    #print(p_w)

    pad_img = np.zeros((h+2*p_h, w+2*p_w, c))
    pad_img[p_h:p_h+h, p_w:p_w+w, :] = src

    if pad_type == 'repetition':
        print('repetition padding')
        for i in range(p_h):
            for j in range(w):
                pad_img[i][j + p_w] = src[0][j]
        for i in range(h, h + p_h):
            for j in range(w):
                pad_img[i + p_h][j + p_w] = src[h-1][j]
        for i in range(h + 2*p_h):
            for j in range(p_w):
                pad_img[i][j] = pad_img[i][p_w]
        for i in range(h + 2*p_h):
            for j in range(w, w + p_w):
                pad_img[i][j + p_w] = pad_img[i][w + p_w - 1]

    else:
        print('zero padding')

    return pad_img

#쌍선형 보간법
def bilinear_interpolation(img, q_, p_):

    #import math에 floor 해야함
    p = floor(p_)
    q = floor(q_)
    err_col = q_ - q
    err_row = p_ - p
    value = (1-err_row)*(img[p][q]*(1-err_col)
                         + img[p+1][q]*err_col) \
                        + err_row*(img[p][q+1]*(1-err_col)
                       + img[p+1][q+1]*err_col)
    return value

def main():
    src = cv2.imread('Lena.png')
    src = cv2.resize(src, None, fx=0.5, fy=0.5)
    src2 = cv2.imread('Lena_transforms.png')

    result = feature_matching_RANSAC(src, src2)
    gussian_result = feature_matching_gaussian(src, src2)

    cv2.imshow("input", src)
    cv2.imshow("goal", src2)
    cv2.imshow('result', result)
    cv2.imshow('gaussian result', gussian_result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

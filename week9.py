import numpy as np
import cv2
import random

def fit_coordinates(src, M):

    h, w, _ = src.shape
    cor_transform = []

    # 원본 이미지의 좌표 값을 행렬 M에 의해 변환시켰을 떄
    # 변환된 좌표의 최대 최소 범위 파악
    for row in range(h + 1):
        for col in range(w + 1):
            P = np.array([
                [col],  # x
                [row],  # y
                [1]
            ])

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

    #print("row max : {} row min : {}".format(row_max, row_min))
    #print("col max : {} col min : {}".format(col_max, col_min))

    return row_max, row_min, col_max, col_min

def backward(src, M):
    #실습 코드 그대로
    ##############################
    # TODO Backward 방식 구현
    # 실습 참고
    ##############################

    h, w, c = src.shape
    # M 역행렬 구하기
    M_inv = np.linalg.inv(M)

    # dst shape 구하기
    row_max, row_min, col_max, col_min = fit_coordinates(src, M)
    # dst 크기
    h_ = round(row_max - row_min)
    w_ = round(col_max - col_min)
    dst = np.zeros((h_, w_, c))

    for row in range(h_):
        for col in range(w_):
            P_dst = np.array([
                [col + col_min],
                [row + row_min],
                [1]
            ])
            # original 좌표로 매핑
            P = np.dot(M_inv, P_dst)
            src_col = P[0, 0]
            src_row = P[1, 0]
            # bilinear interpolation

            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)

            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)

            # index를 초과하는 부분에 대해서는 값을 채우지 않음
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
    #최소자승법 공식
    '''
    :param matches: keypoint matching 정보
    :param kp1: keypoint 정보.
    :param kp2: keypoint 정보2.
    :return: X : 위의 정보를 바탕으로 Least square 방식으로 구해진 Affine 변환 matrix의 요소 [a, b, c, d, e, f].T
    '''

    ##############################
    # TODO Least square 구현
    # A : 원본 이미지 좌표 행렬
    # b : 변환된 좌표 벡터
    # X : 구하고자 하는 Unknown transformation
    ##############################
    A = []
    b = []
    for idx, match in enumerate(matches):
        trainInd = match.trainIdx
        queryInd = match.queryIdx

        x, y = kp1[queryInd].pt
        x_prime, y_prime = kp2[trainInd].pt

        # ---------------------------------------------------
        # ---------------------------------------------------
        #실습 파일의 최소 자승법 공식 그대로 사용
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        b.append([x_prime])
        b.append([y_prime])
        # ---------------------------------------------------
        # ---------------------------------------------------
    #리스트에 넣고
    A = np.array(A)
    b = np.array(b)

    try:
        # ---------------------------------------------------
        # ---------------------------------------------------
        #실습 ppt 22페이지 공식 그대로 적용
        X = np.dot(np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T, b) )
        # ---------------------------------------------------
        # ---------------------------------------------------
    except:
        print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
        X = None
    return X


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

    bf = cv2.BFMatcher(cv2.DIST_L2)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    """
    SIFT에서 특징점들에 대한 매칭 결과 확인하고 싶으면 주석 풀어서 확인
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], outImg=None, flags=2)

    cv2.imshow('matching result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    """
    matches: List[cv2.DMatch]
    cv2.DMatch의 배열로 구성

    matches[i]는 distance, imgIdx, queryIdx, trainIdx로 구성됨
    trainIdx: 매칭된 img1에 해당하는 index
    queryIdx: 매칭된 img2에 해당하는 index

    kp1[queryIdx]와 kp2[trainIdx]는 매칭된 점
    """
    return kp1, kp2, matches

def feature_matching(img1, img2, keypoint_num=None):

    #이전 주체 과제에서 했던 get_matching_keypoints로 keypoint 값을 가져온다
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    #최소 자승법(Least Square)을 통해 X를 구한다
    X = my_ls(matches, kp1, kp2)


    ##########################################
    # TODO Unknown transformation Matrix 구하기
    # M : [ [ a b c]     3 x 3 행렬
    #       [ d e f]
    #        [0 0 1 ]]
    ##########################################
    # ---------------------------------------------------
    # ---------------------------------------------------
    #최소 자승법으로 오차를 줄인 값을 M에 할당
    M = np.array([[X[0][0], X[1][0], X[2][0]],
                  [X[3][0], X[4][0], X[5][0]],
                  [0, 0, 1]])
    # ---------------------------------------------------
    # ---------------------------------------------------

    #이미지 fir을 위해 True를 추가함
    result = backward(img1, M)
    return result.astype(np.uint8)

def feature_matching_RANSAC(img1, img2, keypoint_num=None, iter_num=200, threshold_distance=5):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: sift에서 추출할 keypoint의 수
    :param iter_num: RANSAC 반복횟수, 디폴트 500 이였음
    :param threshold_distance: RANSAC에서 inlier을 정할때의 거리 값
    :return: RANSAC을 이용하여 변환 된 결과
    '''
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    matches_shuffle = matches.copy()


    #########################################################
    # TODO RANSAC 구현하기
    # inliers : inliers의 개수를 저장
    # M_list : 랜덤하게 뽑은 keypoint 3개의 좌표로 구한 변환 행렬
    # 절차
    # 1. 랜덤하게 3개의 matches point를 뽑아냄
    # 2. 1에서 뽑은 matches를 가지고 Least square를 사용한 affine matrix M을 구함
    # 3. 2에서 구한 M을 가지고 모든 matches point와 연산하여 inlier의 개수를 파악
    # 4. M을 사용하여 변환된 좌표와 SIFT를 통해 얻은 변환된 좌표와의 L2 Distance를 구함
    # 5. 거리 값이 threshold_distance보다 작으면 inlier로 판단
    # 6. iter_num만큼 반복하여 가장 많은 inlier를 보유한 M을 Best_M으로 설정
    ##########################################################
    inliers = []    #inlier 개수 저장
    M_list = [] #랜덤으로 고른 데이터들의 affine 행렬
    temp_inlier_cnt = -1 #최대값 찾기 위해

    for i in range(iter_num):
        print('\rcalculate RANSAC ... %d ' % (int((i + 1) / iter_num * 100)) + '%', end='\t')
        random.shuffle(matches_shuffle)

        #1. 랜덤으로 3개
        three_points = matches_shuffle[:3]

        # ---------------------------------------------------
        # ---------------------------------------------------
        # X = my_ls(???)
        # if X is None:
        #     continue
        #
        # M = ???

        #2번 ---
        X = my_ls(three_points, kp1, kp2)

        #np.linalg.inv((np.dot(A.T, A)) 을 계산할 수 없는 경우 넘어감
        #Lena 파일로 했을 때는 안나옴
        if X is None:
            continue

        #마찬가지로 오차를 줄인 값을 M에 할당
        M = np.array([[X[0][0], X[1][0], X[2][0]],
                      [X[3][0], X[4][0], X[5][0]],
                      [0, 0, 1]])
        # ---------------------------------------------------
        # ---------------------------------------------------
        M_list.append(M)

        count_inliers = 0

        #모든 match를 돌며 inlier 개수 파악 시작
        for idx, match in enumerate(matches):
            trainInd = match.trainIdx
            queryInd = match.queryIdx

            kp1_x, kp1_y = kp1[queryInd].pt
            kp2_x, kp2_y = kp2[trainInd].pt

            # ---------------------------------------------------
            # ---------------------------------------------------
            # L2 거리(유클리디안)를 구하기 위해

            #print('ke1_x = ', kp1_x)

            #가로에서 세로로
            keyPointarr = np.array([[kp1_x, kp1_y, 1]]).T
            #print('ketPointarr = ', keyPointarr)

            #M과 kp1, 2의 행렬곱으로 L2 거리를 구하기 위해 행렬을 같게 만든다
            keyPointDot = np.dot(M, keyPointarr)
            #print('M = ', M)
            #print('keyPointDot = ', keyPointDot)

            temp_vector = [kp2_x, kp2_y]
            #vector1, vector2에 맞게 변경

            #4번 L2 Dis 를 구함
            dist = L2_distance(temp_vector, keyPointDot)

            #5.  설정한 쓰레드 값 이상일 경우 가장 좋은 값을 찾기 위해 cnt ++
            if dist < threshold_distance:
                count_inliers += 1
            # ---------------------------------------------------
            # ---------------------------------------------------

        inliers.append(count_inliers)

    # ---------------------------------------------------
    # ---------------------------------------------------
        if(count_inliers > temp_inlier_cnt):
            temp_inlier_cnt = count_inliers

    #6. affine 값 중 가장 높은 inliers의 cut 가 장 높은 것을 best_M으로
    best_M = M_list[inliers.index(temp_inlier_cnt)]
    print(best_M)
    # ---------------------------------------------------
    # ---------------------------------------------------
    result = backward(img1, best_M)

    return result

def L2_distance(vector1, vector2):
    ##########################################
    # TODO L2 Distance 구하기
    ##########################################

    # ---------------------------------------------------
    # ---------------------------------------------------
    #유클리디안 거리 구하기 공식 그대로 적용
    return np.sqrt((vector1[0] - vector2[0][0]) ** 2 + (vector1[1] - vector2[1][0]) ** 2 )
    # ---------------------------------------------------
    # ---------------------------------------------------
def main():
    src = cv2.imread('./Lena.png')
    src2 = cv2.imread('./Lena_transforms.png')

    result_RANSAC = feature_matching_RANSAC(src, src2)
    result_LS = feature_matching(src, src2)
    cv2.imshow('input', src)
    cv2.imshow('result RANSAC 201702086', result_RANSAC)
    cv2.imshow('result LS 201702086', result_LS)
    cv2.imshow('goal', src2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

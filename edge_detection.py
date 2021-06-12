import math
import numpy as np, cv2
from Common.hough import accumulate, masking, select_lines


def houghLines(src, rho, theta, thresh):
    acc_mat = accumulate(src, rho, theta)  # 허프 누적 행렬 계산
    acc_dst = masking(acc_mat, 7, 3, thresh)  # 마스킹 처리 7행,3열
    lines = select_lines(acc_dst, rho, theta, thresh)  # 직선 가져오기
    return lines


def draw_houghLines(src, lines, nline):
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)  # 컬러 영상 변환
    print(lines, nline)
    if lines is None:
        return None
    min_length = min(len(lines), nline)

    for i in range(min_length):
        rho, radian = lines[i, 0, 0:2]  # 수직거리 , 각도 - 3차원 행렬임
        a, b = math.cos(radian), math.sin(radian)
        pt = (a * rho, b * rho)  # 검출 직선상의 한 좌표 계산
        delta = (-1000 * b, 1000 * a)  # 직선상의 이동 위치
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt, delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2), (0, 255, 0), 2, cv2.LINE_AA)

    return dst


image = cv2.imread("12.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
if image is None:
    raise Exception("영상 파일 읽기 오류")

rho, theta = 1, np.pi / 180

canny2 = cv2.Canny(image, 100, 200)  # OpenCV 캐니 에지

lines2 = cv2.HoughLines(canny2, rho, theta, 80)

if lines2 is None:
    print('None Line')
else:
    dst2 = draw_houghLines(canny2, lines2, 7)
    cv2.imshow("HoughLine", dst2)  # OpenCV 캐니 에지

cv2.imshow("Original Image", image)

cv2.imshow("Canny Edge", canny2)  # OpenCV 캐니 에지

cv2.waitKey(0)

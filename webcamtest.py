import cv2

def webcam_test():
    # 0번 카메라 열기 (기본 웹캠)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 장치가 연결되어 있거나 사용 가능한지 확인하세요.")
        return

    print("웹캠 연결 성공. 'q'를 눌러 종료합니다.")

    # 웹캠에서 실시간 프레임 읽기
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 프레임을 창에 표시
        cv2.imshow('Webcam Test', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_test()

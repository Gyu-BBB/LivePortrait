# coding: utf-8

import tyro
import cv2  # OpenCV 추가
import os  # 파일 처리를 위한 os 모듈 추가
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def process_webcam_stream(live_portrait_pipeline, args):  # args 인자를 추가
    # 웹캠에서 실시간으로 프레임 가져오기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 사용할 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 프레임을 모델에 입력할 이미지로 저장 (또는 바로 처리 가능)
        img_name = "webcam_frame.jpg"
        cv2.imwrite(img_name, frame)

        # 소스 이미지를 웹캠 프레임으로 설정
        args.source_image = img_name

        # 추론 실행
        live_portrait_pipeline.execute(args)

        # 결과를 표시 (처리된 비디오를 다시 출력)
        result_frame = cv2.imread("result_image.jpg")  # 예시로 처리 결과 이미지를 불러옴
        if result_frame is not None:
            cv2.imshow("Live Portrait Result", result_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
    crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # 웹캠에서 실시간 비디오 스트림 처리
    if args.source_image == "webcam":
        process_webcam_stream(live_portrait_pipeline, args)  # args를 인자로 전달
    else:
        # 일반 추론 실행
        live_portrait_pipeline.execute(args)


if __name__ == '__main__':
    main()

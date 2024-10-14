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


def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠을 사용할 수 없습니다.")
            break

        cv2.imshow('Press Space to Capture', frame)

        # 스페이스바를 누르면 사진을 저장
        if cv2.waitKey(1) & 0xFF == ord(' '):
            img_name = "webcam_capture.jpg"
            cv2.imwrite(img_name, frame)
            print(f"이미지 저장됨: {img_name}")
            break

    cap.release()
    cv2.destroyAllWindows()

    return img_name


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # 웹캠에서 사진 캡처
    if args.source_image == "webcam":  # source_image로 수정
        print("웹캠에서 이미지를 캡처합니다...")
        args.source_image = capture_image_from_webcam()  # 소스 이미지를 웹캠에서 받은 이미지로 설정

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
    crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    live_portrait_pipeline.execute(args)


if __name__ == '__main__':
    main()

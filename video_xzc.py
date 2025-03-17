import os
import cv2
import torch
import numpy as np
import argparse
import subprocess
from model.RIFE import Model
from tqdm import *

def get_padding(size):
    pad = (16 - size % 16) % 16
    return (pad // 2, pad - pad // 2)  # 对称填充

def process_video(video_path, output_video_path, model, fps_exp=1, save_frames=False, scale_factor = 1):
    # 加载视频并获取相关信息
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    video_folder = os.path.dirname(video_path)
    if save_frames == True:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{video_name}_{fps}fps"
        output_folder = os.path.join(video_folder, output_name)  # 同名文件夹 带 fps
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # 初始化视频写入器
    ret, first_frame = video.read()
    if not ret:
        print("无法读取视频！")
        return

    height, width, _ = first_frame.shape
    pad_top, pad_bottom = get_padding(height)
    pad_left, pad_right = get_padding(width)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps * (2**fps_exp), (width * scale_factor, height * scale_factor))

    last_image = None
    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用 tqdm 显示进度条
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        frame_count = 0
        while frame_count < total_frames:
            # 显式地设置帧位置，确保我们读取的是正确的帧
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = video.read()
            if not ret:
                print(f"Error reading frame {frame_count}")
                break
            frame_count += 1
            # 处理帧，填充和裁剪
            frame_pad = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            if save_frames:
                frame_filename = f"{output_folder}/{frame_count:08d}.jpg"
                cv2.imwrite(frame_filename, frame)  # 保存帧到文件夹
            
            if last_image is not None:
                I0 = torch.from_numpy(np.transpose(last_image, (2, 0, 1))).to(device).unsqueeze(0).float() / 255.
                I1 = torch.from_numpy(np.transpose(frame_pad, (2, 0, 1))).to(device).unsqueeze(0).float() / 255.
                with torch.no_grad():
                    I_mid = model.inference(I0, I1)
                I_mid = (I_mid.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                frame_to_write = I_mid[0, pad_top:height+pad_top, pad_left:width+pad_left, :]
                if scale_factor == 1:
                    video_writer.write(frame_to_write)
                else:
                    video_writer.write(cv2.resize(frame_to_write, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC) )
                    pass
            # 写入当前帧
            if scale_factor == 1:
                video_writer.write(frame)
            else:
                video_writer.write(cv2.resize(frame, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC) )
            last_image = frame_pad  # 更新 last_image
            # 更新进度条
            pbar.update(1)
            
    # 释放资源
    video.release()
    video_writer.release()
    
    print(f"视频处理完成，总帧数：{frame_count}")
    # 获取视频的音频流
    # 步骤 1：提取第一个视频的音频
    temp_audio = "temp_audio.aac"
    temp_video = "temp_video.mp4"
    subprocess.run([
        "ffmpeg", "-i", video_path,  # 输入第一个视频
        "-q:a", "0",              # 设置音频质量为最高
        "-map", "a",              # 只提取音频流
        temp_audio                # 输出临时音频文件
    ], check=True)
    # 步骤 2：将提取的音频替换到第二个视频中
    subprocess.run([
        "ffmpeg",
        "-i", output_video_path,            # 输入第二个视频（提供视频流）
        "-i", temp_audio,         # 输入提取的音频
        "-c:v", "copy",           # 直接复制视频流，不重新编码
        "-c:a", "aac",            # 编码音频为 AAC 格式
        "-map", "0:v:0",          # 选择第一个输入文件的视频流
        "-map", "1:a:0",          # 选择第二个输入文件的音频流
        "-shortest",              # 以较短的输入流（视频或音频）为输出时长
        temp_video                # 输出最终文件
    ], check=True)

    # 步骤 3：清理临时音频文件
    os.replace(temp_video, output_video_path)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)  # 删除临时音频文件
    if os.path.exists(temp_video):
        os.remove(temp_video)  # 删除临时视频文件
    print(f"音频替换完成，输出文件为 {output_video_path}")

device = torch.device("cuda")

if __name__ == "__main__":
    video_path = 'C:\\Users\\Administrator\\Desktop\\xzc-RIFE\\video\\720P.mp4'  # 请替换为你的实际视频路径
    output_video_name = '2K_60fps_audio.mp4'
    model_dir = 'C:\\Users\\Administrator\\Desktop\\xzc-RIFE\\train_log'
    model = Model()
    model.eval()
    assert model_dir != None, "model_dir is None"
    model.load_model(model_dir)
    torch.backends.cudnn.benchmark = True

    # 控制参数：是否保存帧数图像，帧率倍增因子
    save_frames = False  # 设置为 True 时会保存帧
    scale_factor = 2
    process_video(video_path, os.path.join(os.path.dirname(video_path), output_video_name), model, fps_exp=1, save_frames=save_frames, scale_factor=scale_factor)
 
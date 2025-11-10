import subprocess
import shlex
import argparse

def invert_video(input_binary_mask, output_video_path):
    ffmpeg_cmd = f'''
    ffmpeg -y -i "{input_binary_mask}" \
        -vf "format=gray,lut='if(gt(val\,0)\,255\,0)'" \
        -c:v libx264 -pix_fmt yuv420p "{output_video_path}"
    '''
    process = subprocess.run(
        shlex.split(ffmpeg_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert a binary mask video.")
    parser.add_argument("input_video", help="Path to input binary mask video")
    parser.add_argument("output_video", help="Path to output inverted binary mask video")

    args = parser.parse_args()

    invert_video(args.input_video, args.output_video)
import cv2
import argparse
import os

def generate_edges(in_path, out_path, bright=50, contrast=1.0):
    cap = cv2.VideoCapture(in_path)
    assert cap.isOpened(), "Could not open input video."
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "avc1"
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)
        edges = cv2.Canny(blurred, 10, 50)
        out.write(edges)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edge video from input.")
    
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_video", help="Path to save generated edge video")

    parser.add_argument(
        "--bright",
        type=float,
        default=50,
        help="Brightness offset applied before edge detection (default: 50)"
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Contrast multiplier applied before edge detection (default: 1.0)"
    )

    args = parser.parse_args()

    generate_edges(
        args.input_video,
        args.output_video,
        bright=args.bright,
        contrast=args.contrast
    )
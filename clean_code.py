import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import random
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from keras.layers import TFSMLayer
from keras.models import Sequential

# Path to your SavedModel directory
saved_model_path = 'C:\\Users\\Malavika\\yolov7\\_my_model.h51_'
saved_modely_path = 'C:\\Users\\Malavika\\yolov7\\123\\_my_model.h51_'

# Load the model as an inference-only layer
model_layer = TFSMLayer(saved_model_path, call_endpoint='serving_default')
modely_layer = TFSMLayer(saved_model_path, call_endpoint='serving_default')

# Create a Sequential model and add the TFSMLayer
loaded_model = Sequential([model_layer])
loaded_modely = Sequential([modely_layer])

camara = 1

def detect(opt, source, save_img=False):
    weights, save_txt, view_img, imgsz, trace = opt.weights, opt.save_txt, opt.view_img, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # Load 3d model
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    old_x0, old_y0, old_y01, old_x01, old_x1, old_x2, old_x3 = [[0], [0], [0]], [[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]
    old0, intoldy0, old1, old2, old3, times = 0, 0, 0, 0, 0, 0
    new_x2, new_x1, new_y1, new_y2, new_z1 = [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9
    xx0, xx1, xx2, xx3 = [0] * 3, [0] * 3, [0] * 3, [0] * 3

    # Initialize vid_path and vid_writer
    vid_path, vid_writer = None, None

    for path, img, im0s, vid_cap in dataset:
        times += 1
        plt.cla()
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 480)

        # Correcting the conversion of img to a PyTorch tensor
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() / 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b, old_img_h, old_img_w = img.shape[0], img.shape[2], img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = "C:/Users/Malavika/yolov7/text"
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    return

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w, h = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

            # Add your own functionality here based on the results of detections, e.g., updating the 3D plot
            for det in pred:
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        x, y, z = xyxy[0].item(), xyxy[1].item(), conf.item()
                        ax.scatter(x, y, z, c='r', marker='o')
            
            plt.pause(0.001)

    print(f'Done. ({time.time() - t0:.3f}s)')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--source1', type=str, default='0', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc codec for video writer')
    opt = parser.parse_args()
    return opt

def main(opt):
    if opt.update:
        for opt.weights in ['best.pt']:
            detect(opt, opt.source, save_img=False)
            strip_optimizer(opt.weights)
    else:
        detect(opt, opt.source)
        detect(opt, opt.source1)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


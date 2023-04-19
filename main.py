import argparse
import json
import os
import numpy as np
from tracker import Tracker
import pandas as pd
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='el path al json')
parser.add_argument('--output', type=str, help='path de salida')
parser.add_argument('--video_path', type=str, help='path al video, por si queremos guardar imágenes', default=None)
parser.add_argument('--memory_length', type=int, help='cantidad de frames que espero que re aparezca una detección'+ \
                                            'si reaparece la interpolo con icpen los frames faltantes ', default=None)

# parser.add_argument('--draw_detections', help='dibuja detecciones', default=False, action='store_true')
parser.add_argument('--draw_tracking', help='dibuja trackeo', default=False, action='store_true')
parser.add_argument('--draw_circles', help='dibuja trackeo', default=False, action='store_true')
parser.add_argument('--radius', type=int, help='radio de matcheo en píxeles', default=10)
args = parser.parse_args()
COLUMNS = ['image_name','x','y','r','detection','track_id','label']


def main(args):
    print(args.input[-28:-5])
    detections_file = open(args.input, 'r')
    detections = json.load(detections_file)
    detections_file.close()
    json_path, json_name = os.path.split(args.input)
    video_name, _ = os.path.splitext(json_name)
    tracker = Tracker(COLUMNS, video_name, args)

    # for f in detections.values():
    #     for d in f.values():
    #         if d[2] <15:
    #             print(d[2])

    tracker.start_tracking(detections)


    tracker.write_results(args.output)

if __name__ == '__main__':
    main(args)
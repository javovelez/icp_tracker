import argparse
import json
import os
import numpy as np
import open3d as o3d
from tracker import Tracker
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='el path al json')
parser.add_argument('--output', type=str, help='path de salida')
parser.add_argument('--video_path', type=str, help='path al video, por si queremos guardar imágenes', default=None)

# parser.add_argument('--draw_detections', help='dibuja detecciones', default=False, action='store_true')
parser.add_argument('--draw_tracking', help='dibuja trackeo', default=False, action='store_true')
parser.add_argument('--draw_circles', help='dibuja trackeo', default=False, action='store_true')
parser.add_argument('--radius', type=int, help='radio de matcheo en píxeles', default=10)
args = parser.parse_args()
COLUMNS = ['image_name','x','y','r','detection','track_id','label']

def conform_point_cloud(points):
    """
    create a PointCloud object from a matrix
    inputs:
        points: a mumpy matrix with shape (n, 3) (n arbitrary points and x, y, z coordinates)
    return:
        PointCloud object (open3d)
    """
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def point_cloud_viewer(pcs):
    clouds_list = []
    for i, pc in enumerate(pcs):
        clouds_list.append({
            "name": f"{i}",
            "geometry": pc
        })
    o3d.visualization.draw(clouds_list, show_ui=True, point_size=7)


def to_cloud(frame_detections_dict):
    # print(frame_detections_dict)
    vector = [[det[0], det[1],  0.] for det in frame_detections_dict.values()]
    vector = np.array(vector)
    return conform_point_cloud(vector)


def main(args):
    # print(args.input)
    radius=args.radius
    detections_file = open(args.input, 'r')
    detections = json.load(detections_file)
    detections_file.close()
    json_path, json_name = os.path.split(args.input)
    video_name, _ = os.path.splitext(json_name)
    tracker = Tracker(COLUMNS, video_name, args)
    for i in range(len(detections)-1):
        if i==0:
            previous = detections[str(i)]
            tracker.init_ids(previous)
            continue

        tracker.frame += 1
        current = detections[str(i)]
        previous = detections[str(i-1)]
        current_cloud = to_cloud(current)
        previous_cloud = to_cloud(previous)
        icp = o3d.pipelines.registration.registration_icp(previous_cloud, current_cloud, radius)
        correspondence_set = np.asarray(icp.correspondence_set)
        tracker.update_ids(correspondence_set, current, previous)

        # colors = np.zeros((len(previous_cloud.points), 3))
        # colors[:] = [0, 1, 0]
        # colors[matched_source_idx, :] = [1, 0, 0]
        # previous_cloud.colors = o3d.utility.Vector3dVector(colors)
        # colors[:] = [1, 0, 1]
        # colors[matched_target_idx, :] = [1, 1, 1]
        # current_cloud.colors = o3d.utility.Vector3dVector(colors)
        # point_cloud_viewer([current_cloud, previous_cloud])
    tracker.write_results(args.output)

if __name__ == '__main__':
    main(args)
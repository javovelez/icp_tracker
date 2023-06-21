import pandas as pd
import copy
import cv2
import os
import random
import open3d as o3d
import numpy as np


class Tracker:
    def __init__(self, columns, video_name, args, frame = 0):
        self.previous = o3d.geometry.PointCloud(o3d.utility.Vector3dVector())
        self.previous_cloud_extended = o3d.geometry.PointCloud(o3d.utility.Vector3dVector())
        df = pd.DataFrame(columns=columns)
        self.video_name = video_name
        self.df = df
        self.frame = frame
        self.previous_ids_dict = {}
        self.current_ids_dict = {}
        self.id_ctr = 0
        self.args = args
        self.radius = args.radius
        self.no_matched_detections = dict()
        if args.video_path is not None:
            self.vs = cv2.VideoCapture(args.video_path)
        if args.memory_length:
            self.memory_dict = dict()
            for m in range(args.memory_length):
                self.memory_dict[m] = None


    def start_tracking(self, detections):
        for i in range(len(detections) - 1):
            if i == 0:
                self.previous = detections[str(i)]
                self.init_ids(self.previous)
                continue

            self.frame += 1
            no_matched_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector())
            current = detections[str(i)]
            previous_len = len(self.previous)
            previous_extended_len = len(np.asarray(self.previous_cloud_extended.points))
            current_len = len(current)
            current_cloud = self.to_cloud(current)
            previous_cloud = self.to_cloud(self.previous)
            prev_cloud_to_process = previous_cloud + self.previous_cloud_extended
            icp = o3d.pipelines.registration.registration_icp(prev_cloud_to_process, current_cloud, self.radius)
            correspondence_set = np.asarray(icp.correspondence_set)

            fitness = icp.fitness
            transformation_changed = False
            if icp.fitness < 0.8:
                if icp.fitness >= 0.8:  # este es por si ya enconté un buen fitness no seguir buscando
                    break
                for px_shift in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:
                    for x, y in [(px_shift, 0), (px_shift, px_shift), (0, px_shift), (-px_shift, 0),
                                 (-px_shift, -px_shift), (0, -px_shift), (px_shift, -px_shift), (-px_shift, px_shift)]:
                        previous_copy = copy.deepcopy(self.previous)
                        previous_copy_cloud_aux = self.to_cloud(previous_copy, x, y) +  self.previous_cloud_extended
                        icp2 = o3d.pipelines.registration.registration_icp(previous_copy_cloud_aux, current_cloud,
                                                                           self.radius)

                        if icp.fitness < icp2.fitness:
                            icp = icp2
                            transformation_changed = True
                            self.x, self.y = x, y
                            previous_copy_cloud = copy.deepcopy(previous_copy_cloud_aux)
                            if icp.fitness >= 0.8:
                                break

                correspondence_set = np.asarray(icp.correspondence_set)

            if icp.fitness >= 0.8: #interpolo solo si el matcheo entre frames fue satisfactorio
                # interpolation of not matched points
                no_matched_idx = self.check_no_matched_idx(previous_len, correspondence_set)

                idx_aux = 0
                for key, value in self.previous.items():
                    if int(key) in no_matched_idx:
                        value.append(self.previous_ids_dict[int(key)])
                        self.no_matched_detections[idx_aux+current_len] = value
                        idx_aux += 1
                # self.no_matched_detections = {idx+current_len: value for idx, (key, value) in enumerate(self.previous.items()) if int(key) in no_matched_ids}
                # if no_matched_detections:
                if not transformation_changed:
                    self.x, self.y = 0, 0
                no_matched_cloud = self.to_cloud(self.no_matched_detections, self.x, self.y)
                no_matched_cloud.transform(icp.transformation)

                    # previous_copy_cloud = self.to_cloud(self.previous, self.x, self.y)
                    # previous_matched_idx = correspondence_set[:, 0]
                    # current_matched_idx = correspondence_set[:, 1]
                    # previous_copy_cloud.transform(icp.transformation)
                    # colors = np.zeros((len(previous_copy_cloud.points), 3))
                    # colors[:] = [0, 1, 0]
                    # colors[previous_matched_idx, :] = [1, 0, 0]
                    # previous_copy_cloud.colors = o3d.utility.Vector3dVector(colors)
                    # colors = np.zeros((len(current_cloud.points), 3))
                    # colors[:] = [1, 0, 1]
                    # colors[current_matched_idx, :] = [1, 1, 1]
                    # current_cloud.colors = o3d.utility.Vector3dVector(colors)
                    # print(icp.inlier_rmse)
                    # self.point_cloud_viewer([current_cloud, previous_copy_cloud, no_matched_cloud])

            if icp.fitness < 0.8:
                print(self.args.input[-28:-5], f' fitness: {icp.fitness} frame {self.frame}:')
                print(f'n_previous: {previous_len}; n_current {current_len}, matcheos: {len(correspondence_set)}')

            self.update_ids(correspondence_set, current)
            self.previous = current
            self.previous_cloud_extended = no_matched_cloud

    def check_no_matched_idx(self, previous_len, correspondence_set):
        prev_idx = list(range(previous_len))
        previous_matched = correspondence_set[:,0]
        not_matched_points = []
        for p_idx in prev_idx:
            if p_idx not in previous_matched:
                    not_matched_points.append(p_idx)
        return not_matched_points

    def conform_point_cloud(self, points):
        """
        create a PointCloud object from a matrix
        inputs:
            points: a mumpy matrix with shape (n, 3) (n arbitrary points and x, y, z coordinates)
        return:
            PointCloud object (open3d)
        """
        if len(points)>0:
            return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        else:
            return o3d.geometry.PointCloud(o3d.utility.Vector3dVector())

    def point_cloud_viewer(self, pcs):
        clouds_list = []
        for i, pc in enumerate(pcs):
            clouds_list.append({
                "name": f"{i}",
                "geometry": pc
            })
        o3d.visualization.draw(clouds_list, show_ui=True, point_size=7)

    def to_cloud(self, frame_detections_dict, x=0, y=0):
        # print(frame_detections_dict)
        vector = [[det[0] + x, det[1] + y, 0.] for det in frame_detections_dict.values()]
        vector = np.array(vector)
        return self.conform_point_cloud(vector)

    def draw_circle(self, img, center, radius, color=(23, 220, 75), thickness=1):
        img = cv2.circle(img, center, radius, color, thickness)

    def draw_circles(self, img):
        data = self.df
        labels = data[data['image_name']==self.video_name+f'_{self.frame}.png']
        # ['image_name', 'x', 'y', 'r', 'detection', 'track_id', 'label']
        for _, label in labels.iterrows():
            image_name, x, y, r, _ , track_id, _ = list(label)
            center = (round(x), round(y),)
            radius = round(r)
            if self.args.draw_circles:
                self.draw_circle(img, center, radius)
            text = f"{track_id}"
            cv2.putText(img, text, (round(x)-5, round(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2 )
            # (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.imwrite(f'{self.args.output}/{self.video_name}_{self.frame}_track.png', img)

    def init_ids(self, vector):
        self.id_ctr = len(vector) - 1 #contador de id
        for init_id, v in enumerate(vector.values()):
            self.previous_ids_dict[init_id] = init_id     # diccionario clave: previus detection index; valor:id
                                             # todas las detecciones anteriores tienen id
            self._add_to_bundle(v[0], v[1], v[2], init_id)
        if self.args.draw_tracking:
            frame = self.vs.read()
            flag, img = frame
            if not flag:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.args.video_path)
            self.draw_circles(img)

    def update_ids(self, correspondence_set, current):
        print(f'Frame: {self.frame}')

        self.current_ids_dict.clear()
        len_previous = len(np.asarray(self.to_cloud(self.previous).points))
        len_previous_extended = len(np.asarray(self.previous_cloud_extended.points))
        previous_range = list(range(len_previous))
        previous_not_matched_range = list(range(len_previous, len_previous))
        previous_matched = correspondence_set[:,0]
        real_prev_matches = []
        real_prev_matches_idx = []
        interpolated_prev_matches = []
        interpolated_prev_matches_idx = []
        # primero busco los índices de corresponce_set que corresponden a matcheos reales
        # y a matcheos con puntos interpolados
        for idx, pm in enumerate(previous_matched):
            if pm in previous_range:
                real_prev_matches_idx.append(idx)
            else:
                interpolated_prev_matches_idx.append(idx)

        # print(real_prev_matches)
        # print( 'ipm: ', interpolated_prev_matches)
        # print(np.asarray(self.previous_cloud_extended.points))
        current_matched = correspondence_set[:,1]
        real_current_matched = correspondence_set[real_prev_matches_idx, 1]
        interpolated_current_matched = correspondence_set[interpolated_prev_matches_idx, 1]
        real_prev_matches = correspondence_set[real_prev_matches_idx,0]
        interpolated_prev_matches = correspondence_set[interpolated_prev_matches_idx, 0]
        match_dict = { cv[0]:cv[1] for cv in correspondence_set}

        for idx, _ in enumerate(self.previous.values()):
            if idx in  real_prev_matches:
                self.current_ids_dict[match_dict[idx]] = self.previous_ids_dict[idx]

        for idx, cv in enumerate(current.values()):
            if self.frame == 3 and idx == 21:
                print("holis")
            if idx not in real_current_matched: #and idx not in interpolated_current_matched
                self.id_ctr += 1
                self.current_ids_dict[idx] = self.id_ctr
                self._add_to_bundle(cv[0], cv[1], cv[2], self.current_ids_dict[idx])
            else:
                self._add_to_bundle(cv[0], cv[1], cv[2], self.current_ids_dict[idx])
        if self.args.draw_tracking:
            frame = self.vs.read()
            flag, img = frame
            if not flag:
                raise Exception()
            self.draw_circles(img)
        self.previous_ids_dict = copy.deepcopy(self.current_ids_dict)



    def _add_to_bundle(self, x, y, grape_radius, track_id):
        #['image_name', 'x', 'y', 'r', 'detection', 'track_id', 'label']
        det = [f'{self.video_name}_{self.frame}.png', x, y, grape_radius,'detecting', track_id, 'baya']
        self.df.loc[len(self.df)] = det

    def write_results(self, output_path, name='detections.csv'):
        o_p = output_path.split('/')
        del o_p[-1]
        separador = '/'
        o_p = separador.join(o_p) + '/'
        if o_p.endswith('/'):
            csv_name = output_path + name
        else:
            csv_name = output_path + '/' + name

        # print(csv_name)
        # self.df.sort_values('track_id', inplace=True)
        self.df.to_csv(csv_name)

import numpy as np
import yaml

class Obstacle():
    """
    Dynamic or static rectangular obstacle. It is assumed that dynamic objects are under constant acceleration.
    E.g. moving vehicle, parked vehicle, wall
    """
    def __init__(self, centroid, dx, dy, angle=0, vel=[1, 0], acc=[0, 0]):
        """
        :param centroid: centroid of the obstacle
        :param dx: length of the vehicle >=0
        :param dy: width of the vegicle >= 0
        :param angle: anti-clockwise rotation from the x-axis
        :param vel: [x-velocity, y-velocity], put [0,0] for static objects
        :param acc: [x-acceleration, y-acceleration], put [0,0] for static objects/constant velocity
        """
        self.centroid = centroid
        self.dx = dx
        self.dy = dy
        self.angle = angle
        self.vel = vel #moving up/right is positive
        self.acc = acc
        self.time = 0 #time is incremented for every self.update() call

    def __get_points(self, centroid):
        """
        :return A line: ((x1,y1,x1',y1'))
                or four line segments: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        dx_cos = self.dx*np.cos(self.angle)
        dx_sin = self.dx*np.sin(self.angle)
        dy_sin = self.dy*np.sin(self.angle)
        dy_cos = self.dy*np.cos(self.angle)

        BR_x = centroid[0] + 0.5*(dx_cos + dy_sin) #BR=Bottom-right
        BR_y = centroid[1] + 0.5*(dx_sin - dy_cos)
        BL_x = centroid[0] - 0.5*(dx_cos - dy_sin)
        BL_y = centroid[1] - 0.5*(dx_sin + dy_cos)
        TL_x = centroid[0] - 0.5*(dx_cos + dy_sin)
        TL_y = centroid[1] - 0.5*(dx_sin - dy_cos)
        TR_x = centroid[0] + 0.5*(dx_cos - dy_sin)
        TR_y = centroid[1] + 0.5*(dx_sin + dy_cos)

        seg_bottom = (BL_x, BL_y, BR_x, BR_y)
        seg_left = (BL_x, BL_y, TL_x, TL_y)

        if self.dy == 0: #if no height
            return (seg_bottom,)
        elif self.dx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (TL_x, TL_y, TR_x, TR_y)
            seg_right = (BR_x, BR_y, TR_x, TR_y)
            return (seg_bottom, seg_top, seg_left, seg_right)

    def __get_points_old(self, centroid):
        """
        :return A line: ((x1,y1,x1',y1'))
                or four line segments: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        seg_bottom = (centroid[0] - self.dx/2, centroid[1] - self.dy/2, centroid[0] + self.dx/2, centroid[1] - self.dy/2)
        seg_left = (centroid[0] - self.dx/2, centroid[1] - self.dy/2, centroid[0] - self.dx/2, centroid[1] + self.dy/2)

        if self.dy == 0: #if no height
            return (seg_bottom,)
        elif self.dx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (centroid[0] - self.dx/2, centroid[1] + self.dy/2, centroid[0] + self.dx/2, centroid[1] + self.dy/2)
            seg_right = (centroid[0] + self.dx/2, centroid[1] - self.dy/2, centroid[0] + self.dx/2, centroid[1] + self.dy/2)
            return (seg_bottom, seg_top, seg_left, seg_right)

    def update(self, pos=None, recycle_pos=True):
        """
        :param pos: manually give a position. If None, update based on time.
        :return: updated centroid
        """
        if pos is None:
            disp_x = self.centroid[0] + self.vel[0]*self.time + 0.5*self.acc[0]*(self.time**2) #s_x = ut + 0.5at^2
            disp_y = self.centroid[1] + self.vel[1]*self.time + 0.5*self.acc[1]*(self.time**2) #s_y = ut + 0.5at^2
        else:
            if recycle_pos is True:
                if self.time >= pos.shape[0]:
                    t = self.time%pos.shape[0]
                else:
                    t = self.time
            else: #stay at where it is when t > t_max
                if self.time > pos.shape[0]:
                    t = pos.shape[0]
                else:
                    t = self.time
            disp_x = pos[t, 0]
            disp_y = pos[t, 1]
        self.time += 1 #time is incremented for every self.update() call
        return self.__get_points(centroid=[disp_x, disp_y])

class EnvModel:
    def __init__(self, env_config_path):
        # set up the environment with obstacles
        self.all_obstacles, area = self.load_obstacles_config(env_config_path='./config/toy1.yaml')
                                                        
    def update_obstacles(self):
        # update obstacles
        all_obstacle_segments = []
        for obs_i in self.all_obstacles:
            all_obstacle_segments += obs_i.update()
        # get the environment for plotting purposes
        connected_components = self.connect_segments(all_obstacle_segments)

        return all_obstacle_segments, connected_components
        
    def load_obstacles_config(self, env_config_path):
        """
        :param environment: name of the yaml config file
        :return: all obstacles, area of the environment
        """
        with open(env_config_path) as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)

            # load environment area parameters
            area = yaml_data['area']
            area = (area['x_min'], area['x_max'], area['y_min'], area['y_max'])

            # load static and dynamic obstacles
            obs = yaml_data['obstacles']
            all_obstacles = []
            for i in range(len(obs)):
                obs_i = Obstacle(centroid=[obs[i]['centroid_x'], obs[i]['centroid_y']], dx=obs[i]['dx'], dy=obs[i]['dy'],
                                angle=obs[i]['orientation']*np.pi/180, vel=[obs[i]['velocity_x'], obs[i]['velocity_y']],
                                acc=[obs[i]['acc_x'], obs[i]['acc_y']])
                all_obstacles.append(obs_i)
        return all_obstacles, area

    def connect_segments(self, segments, resolution = 0.01):
        """
        :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
            step_size : resolution for plotting
        :return: stack of all connected line segments as (X, Y)
        """

        for i, seg_i in enumerate(segments):
            if seg_i[1] == seg_i[3]: #horizontal segment
                x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
                y = seg_i[1]*np.ones(len(x))
            elif seg_i[0] == seg_i[2]: #vertical segment
                y = np.arange(min(seg_i[1],seg_i[3]), max(seg_i[1],seg_i[3]), resolution)
                x = seg_i[0]*np.ones(len(y))
            else: # gradient exists
                m = (seg_i[3] - seg_i[1])/(seg_i[2] - seg_i[0])
                c = seg_i[1] - m*seg_i[0]
                x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
                y = m*x + c

            obs = np.vstack((x, y)).T
            if i == 0:
                connected_segments = obs
            else:
                connected_segments = np.vstack((connected_segments, obs))

        return connected_segments
    
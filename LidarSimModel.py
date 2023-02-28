import numpy as np


class LidarSimmodel:
    def __init__(self):
        # set Lidar parameters
        self.n_reflections = 36
        self.fov = 180 * np.pi/180     # fov(deg. to rad.) 
        self.max_laser_distance = 50 
        self.unoccupied_points_per_meter = 0.5
        self.install_angle = -np.pi/2  # 정면방향 설치

    def lidar_sim(self, pos_x, pos_y, psi, all_obstacle_segments):
        self.scan_data = []
        cur_pos = np.array([pos_x, pos_y, psi])
        # update laser reflections 
        self.dist_theta = self.get_laser_ref(all_obstacle_segments, self.fov, self.n_reflections, self.max_laser_distance, cur_pos)
        # (x,y) of laser reflections
        intall_ang = self.install_angle 
        angles = np.linspace(psi+intall_ang, psi+intall_ang + self.fov, self.n_reflections)
        self.laser_data_xy = np.vstack([self.dist_theta*np.cos(angles), self.dist_theta*np.sin(angles)]).T + cur_pos[:2]
                
    # update laser reflections 
    def get_laser_ref(self, segments, fov=np.pi, n_reflections=180, max_dist=100, xytheta_robot=np.array([0.0, 0.0])):
        """
        :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
            fov: sight of the robot - typically pi or 4/3*pi
            n_reflections: resolution=fov/n_reflections
            max_dist: max distance the robot can see. If no obstacle, laser end point = max_dist
            xy_robot: robot's position in the global coordinate system
        :return: 1xn_reflections array indicating the laser end point
        """
        xy_robot = xytheta_robot[:2] #robot position
        theta_robot = xytheta_robot[2]-np.pi/2 #robot angle in rad

        angles = np.linspace(theta_robot, theta_robot+fov, n_reflections)
        dist_theta = max_dist*np.ones(n_reflections) # set all laser reflections to 100

        for seg_i in segments:
            xy_i_start, xy_i_end = np.array(seg_i[:2]), np.array(seg_i[2:]) #starting and ending points of each segment
            for j, theta in enumerate(angles):
                xy_ij_max = xy_robot + np.array([max_dist*np.cos(theta), max_dist*np.sin(theta)]) # max possible distance
                intersection = self.get_intersection(xy_i_start, xy_i_end, xy_robot, xy_ij_max)

                if intersection is not None: #if the line segments intersect
                    r = np.sqrt(np.sum((intersection-xy_robot)**2)) #radius

                    if r < dist_theta[j]:
                        dist_theta[j] = r
        
        return dist_theta
        

    def get_intersection(self, a1, a2, b1, b2) :
        """
        :param a1: (x1,y1) line segment 1 - starting position
        :param a2: (x1',y1') line segment 1 - ending position
        :param b1: (x2,y2) line segment 2 - starting position
        :param b2: (x2',y2') line segment 2 - ending position
        :return: point of intersection, if intersect; None, if do not intersect
        #adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
        """
        def perp(a) :
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            return b

        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot( dap, dp )

        intersct = np.array((num/denom.astype(float))*db + b1) #TODO: check divide by zero!

        delta = 1e-3
        condx_a = min(a1[0], a2[0])-delta <= intersct[0] and max(a1[0], a2[0])+delta >= intersct[0] #within line segment a1_x-a2_x
        condx_b = min(b1[0], b2[0])-delta <= intersct[0] and max(b1[0], b2[0])+delta >= intersct[0] #within linex segment b1_x-b2_x
        condy_a = min(a1[1], a2[1])-delta <= intersct[1] and max(a1[1], a2[1])+delta >= intersct[1] #within line segment a1_y-b1_y
        condy_b = min(b1[1], b2[1])-delta <= intersct[1] and max(b1[1], b2[1])+delta >= intersct[1] #within line segment a2_y-b2_y
        if not (condx_a and condy_a and condx_b and condy_b):
            intersct = None #line segments do not intercept i.e. interception is away from from the line segments

        return intersct
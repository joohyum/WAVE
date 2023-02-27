import numpy as np
import matplotlib.pyplot as plt

class shipmodel:

    def __init__(self, x, y, chi):
        # ship shape
        self.veh_info_L = 4.88  # USV_Length
        self.veh_info_B = 2.44  # Usv_beam
        self.veh_info_buffer = 2.5*self.veh_info_L
        self.veh_info_headDir = 5
        self.veh_info_ssz = 10 # buffer = safe separation zone (ssz)
        self.dt = 0.1
        self.tf = 10

        # dynamic coefficient
        self.au = -1.1391
        self.bu = 0.0028
        self.bu_bias = 0.6836
        self.a11 = 0.0161
        self.a12 = 0.0052
        self.bv = 0.002
        self.bv_bias = 0.0068
        self.a21 = 8.2861
        self.a22 = 0.9860
        self.br = 0.0307
        self.br_bias = 1.3276
        
        # for calculation of ssz
        self.x_tmp = self.veh_info_L*np.array([0.5,1,0.5,-1,-1,0.5])
        self.y_tmp = self.veh_info_B*np.array([1,0,-1,-1,1,1])
        self.xyr = np.sqrt(np.power(self.x_tmp, 2)+np.power(self.y_tmp, 2))
        self.theta = np.arctan2(self.y_tmp, self.x_tmp)

        self.vehx = np.cos(self.theta+chi)*self.xyr+x
        self.vehy = np.sin(self.theta+chi)*self.xyr+y
        # self.theta = np.linspace(0,360,361)/180*np.pi


    # dynamics of vehicle   
    def WAMV_CNU(self, states, control):

        Tp = control[0]
        Ts = control[1]

        u = states[0]
        v = states[1]
        r = states[2]
        x = states[3]
        y = states[4]
        psi = states[5]


        xdot = np.array([self.au*u+self.bu*(Tp+Ts)+self.bu_bias,
                        self.a11*v-self.a12*r+self.bv*(Tp-Ts)*self.veh_info_B/2+self.bv_bias,
                        self.a21*v-self.a22*r+self.br*(Tp-Ts)*self.veh_info_B/2+self.br_bias,  
                        u*np.cos(psi)-v*np.sin(psi),
                        u*np.sin(psi)+v*np.cos(psi),
                        r]) 
        return xdot                                         

    # integration of state
    def integration(self, states, control, dt,  method='Euler'):
        if method == 'Euler':
            s_states = states + dt * self.WAMV_CNU(states, control)
        elif method == 'RK45':
            k1 =  self.WAMV_CNU(states, control) * dt
            k2 =  self.WAMV_CNU(states+0.5*k1, control) * dt
            k3 =  self.WAMV_CNU(states+0.5*k2, control) * dt
            k4 =  self.WAMV_CNU(states+k3, control) * dt
            k = ( k1 + 2*k2 + 2*k3 + k4 )/ 6
            s_states = states + k
        else :
            raise ValueError("Available method input velue is Euler or RK45 ")

        return s_states

    def ship_shape(self, pos_x, pos_y, course_angle, speed):

        posx =pos_x + self.vehx*np.cos(course_angle) - self.vehy*np.sin(course_angle)
        posy = pos_y + self.vehx*np.sin(course_angle) + self.vehy*np.cos(course_angle)

        ssz_r = self.veh_info_ssz; 
        ssz_x = ssz_r*np.cos(self.theta) + pos_x
        ssz_y = ssz_r*np.sin(self.theta) + pos_y

        # heading line depending on speed
        hl_x = np.array([pos_x, pos_x + (self.veh_info_headDir * speed) * np.cos(course_angle)]) 
        hl_y = np.array([pos_y, pos_y + (self.veh_info_headDir * speed) * np.sin(course_angle)])

        ship_data = [posx, posy, ssz_x, ssz_y, hl_x, hl_y]

        return ship_data
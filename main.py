# Read the waypoint from informed RRT*
from LidarSimModel import LidarSimmodel
from EnvModel import EnvModel
import waypoints as wpt
import sys
from shipmodel import *
# import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt

global x_max, y_max

# def ShipModel():
#     # ship shape
#     veh_info_L = 4.88  # USV_Length
#     veh_info_B = 2.44  # Usv_beam
#     veh_info_buffer = 2.5 * veh_info_L
#     veh_info_headDir = 5
#     veh_info_ssz = 10  # buffer = safe separation zone (ssz)
#     dt = 0.1
#     tf = 10
#
#     # dynamic coefficient
#     au = -1.1391
#     bu = 0.0028
#     bu_bias = 0.6836
#     a11 = 0.0161
#     a12 = 0.0052
#     bv = 0.002
#     bv_bias = 0.0068
#     a21 = 8.2861
#     a22 = 0.9860
#     br = 0.0307
#     br_bias = 1.3276
#
#     # for calculation of ssz
#     vehx = veh_info_L * np.array([0.5, 1, 0.5, -1, -1, 0.5])
#     vehy = veh_info_B * np.array([1, 0, -1, -1, 1, 1])
#     theta = np.linspace(0, 360, 361) / 180 * np.pi
class custom_functions:
    def ssa(angle):
        '''
        각도 값을 [-pi pi)값으로 변환해주는 함수
        :return:
        '''
        if type(angle) == list:
            angle = [(angle + [np.pi])[i] % (2 * np.pi) - np.pi for i in range(len(angle + [np.pi]))]
        else:
            angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

        return angle

    def LOSchi(x, y, Delta, R_switch, wpt_pos_x, wpt_pos_y, waypoint_index):
        '''
        % Initialization:
        % The active waypoint(xk, yk) where k = 1, 2, ..., n is a persistent
        % integer should be initialized to the first waypoint, k = 1, using
        % >> clear LOSchi
        %
        % Inputs:
        %   (x, y): craft North - East positions(m)
        % Delta: positive look - ahead distance(m)
        % R_switch: go to next waypoint when the along - track distance x_e
        % is less than R_switch(m)
        % wpt.pos.x = [x1, x2,.., xn]': array of waypoints expressed in NED (m)
        % wpt.pos.y = [y1, y2,.., yn]': array of waypoints expressed in NED (m)
        % U: speed(m / s), only needed for computation of omega_chi_d
        % chi: course angle(rad), only needed for computation of omega_chi_d
        %
        % Feasibility citerion:
        % The switching parameter R_switch > 0  must satisfy, R_switch < dist,
        % where dist is the distance between the two waypoints at k and k + 1:
        % dist = sqrt((wpt.pos.x(k + 1) - wpt.pos.x(k)) ^ 2
        % + (wpt.pos.y(k + 1) - wpt.pos.y(k)) ^ 2);
        %
        % Outputs:
        % chi_d: desired course angle(rad)
        '''
        global k  # active waypoint index(initialized by: clear LOSchi)
        global xk  # active waypoint(xk, yk) corresponding to integer k
        global yk

        # Initialization of(xk, yk) and (xk_next, yk_next)

        try:
            if k:

                # check if R_switch is smaller than the minimum distance between the waypoints
                if R_switch > min(np.sqrt(np.diff(wpt_pos_x) ** 2 + np.diff(wpt_pos_y) ** 2)):
                    print("The distances between the waypoints must be larger than R_switch")
                    sys.exit(1)

                # check input parameters
                if (R_switch < 0):
                    print("R_switch must be larger than zero")
                    sys.exit(1)
                if (Delta < 0):
                    print("Delta must be larger than zero")
                    sys.exit(1)

        except NameError:
            k = 0                                   # else, use the last one in the array
            xk = wpt_pos_x[len(wpt_pos_x)-1]
            yk = wpt_pos_y[len(wpt_pos_y)-1]

        # Read next waypoint(xk_next, yk_next) from wpt.pos
        n = len(wpt_pos_x)
        if k < n:  # if there are more waypoints, read next one
            xk_next = wpt_pos_x[k]
            yk_next = wpt_pos_y[k]
        else:
            xk_next = wpt_pos_x[len(wpt_pos_x)-1]
            yk_next = wpt_pos_y[len(wpt_pos_y)-1]

        # Print active waypoint
        # fprintf('Active waypoint:\n')
        # fprintf('  (x%1.0f, y%1.0f) = (%.2f, %.2f) \n', k, k, xk, yk);

        # Compute the desired course angle
        pi_p = math.atan2(yk_next - yk, xk_next - xk)     # path - tangential angle w.r.t.to North

        # along - track and cross - track errors(x_e, y_e) expressed in NED
        x_e = (x - xk) * math.cos(pi_p) + (y - yk) * math.sin(pi_p)
        y_e = -(x - xk) * math.sin(pi_p) + (y - yk) * math.cos(pi_p)

        # Waypoint update(1)
        # if the next waypoint satisfy the switching criterion, k = k + 1
        # d = sqrt((xk_next - xk) ^ 2 + (yk_next - yk) ^ 2);
        # if ((d - x_e < R_switch) & & (k < n))
        #   % k = k + 1;
        # xk = xk_next;             # update active waypoint
        #yk = yk_next;
        #end

        # Waypoint update(2)
        x_error = -xk_next + x
        y_error = -yk_next + y
        R = math.sqrt(x_error ** 2 + y_error ** 2)
        if ((R < R_switch) and (k < n)):
            k = k + 1
            waypoint_index = waypoint_index + 1
            xk = xk_next     # update active waypoint
            yk = yk_next

        # LOS guidance law
        Kp = 1 / Delta
        chi_d = pi_p - math.atan(Kp * y_e)

        return chi_d, y_e, waypoint_index

    def sat(x, xmax):
        if abs(x) >= xmax:
            y = np.sign(x) * xmax
        else:
            y = x
        return y

    def WAMV_CNU(states, control):
         Tp = float(control[0])
         Ts = float(control[1])

         u = states[0,0]
         v = states[1,0]
         r = states[2,0]
         x = states[3,0]
         y = states[4,0]
         psi = states[5,0]

         xdot = [[-1.1391*u+0.0028*(Tp+Ts)+0.6836],
                 [0.0161*v-0.0052*r+0.002*(Tp-Ts)*2.44/2+0.0068],
                 [8.2861*v-0.9860*r+0.0307*(Tp-Ts)*2.44/2+1.3276],
                 [u*math.cos(psi)-v*math.sin(psi)],
                 [u*math.sin(psi)+v*math.cos(psi)],
                 [r]]

         return xdot

def draw_shipmodel(x, y, u, v, chi):
    ownship = shipmodel(x, y, psi+np.arctan2(v, u))
    plt.plot(ownship.vehx, ownship.vehy, color='b')
    plt.plot(path_x, path_y, 'b--')
    plt.draw()

# def draw_lidar(x, y, psi):

Lida = LidarSimmodel()
Env = EnvModel(env_config_path='./config/toy1/yaml')
all_obstacle_segments, connected_components = Env.update_obstacles()

# Determine Start/Goal point
#point(list) -> point = [x, y]
start_pt = [10, 10]
goal_pt = [490, 490]

# main graph plot
# plt.plot(start_pt[0], start_pt[1], 'r.')
# plt.plot(wpt.x_value, wpt.y_value, 'r.')
# plt.grid()
# plt.ylabel('$y$-position [m]')
# plt.xlabel('$x$-position [m]')
# plt.text(start_pt[0], start_pt[1], 'Start', fontsize=10)
# plt.text(goal_pt[0], goal_pt[1], 'Goal', fontsize=10)
# plt.title('LOS + PID control')
# plt.plot()

t_stack = []
x_stack = []
chi_ref_stack = []
chi_d_stack = []
y_e_stack = []
U_ref_stack = []
u_d_stack = []
control_stack = []
path_x = []
path_y = []

x_max = 500
y_max = 500
axis = [0, x_max, 0, y_max]

# Own ship modul_waypoints
h = np.array([0.1000])
N = 200000

# initial values for x = [u, v, r, x, y, psi]]
x = np.zeros([6, 1])
x[0,0] = 1
x[5,0] = math.pi/4

# PID course autopilot (Nomoto gains)
T = 1
m = 41.4
K = T / m
wn = 1.5
zeta = 1
Kp = m * wn ** 2
Kd = m * (2 * zeta * wn - 1 / T)
Td = Kd / Kp
Ti = 10 / wn
Ki = Kp * (1 / Ti)

# Reference model
#wn_d = 0.5    natural frequency
#wn_d = 0.9    natural frequency
wn_d = 1.2    #natural frequency
zeta_d = 1.0  #relative damping factor
omega_d = 0
a_d = 0

# Propeler dynamics
# Load condition
mp = 25   #payload mass(kg), max value 45 kg
rp = [0, 0, -0.35]   #location of payload (m)

# Current
V_c = 1   # current speed (m/s)
B = [[1, 1], [1.22, -1.22]]  # input matrix
Binv = np.linalg.inv(B)

# MAIN LOOP (TEST)
simdata = np.zeros([N + 1, 25])  # table for simulation data
R_switch = 4
# Delta = 20   default
Delta = 50

wpt_pos_x = wpt.x_value
wpt_pos_y = wpt.y_value

psi_init = math.atan2((wpt.y_value[1] - wpt.y_value[0]), (wpt.x_value[1] - wpt.x_value[0]))

# Set ownship initial position at the first waypoint
wp_index = 0
x[3, 0] = wpt_pos_x[wp_index]
x[4, 0] = wpt_pos_y[wp_index]
x[5, 0] = psi_init

# xd = [[0], [0], [0]]
xd = np.array([[psi_init], [0], [0]])

chi_error_past = 0
chi_error_sum = 0   #integral state

surge_error_past = 0
surge_error_sum = 0
Kp_surge = 1500
Kd_surge = 100
Ki_surge = 100

# surge reference speed
# U_ref = 1.5    successful
U_ref = 2       #successful

u_d_min = 1e-10
x_hat = np.zeros([5, 1])
pre_waypoint_index = 1
waypoint_index = 1
num = len(wpt_pos_x)
movie_iter = 0

# axis([0 x_max 0 y_max])
# axis equal
f_iter = 0
T_max = 400
T_min = -400

fig = plt.figure(figsize=(8, 8), dpi=100)

for i in range(1,N+2):
    t = (i - 1) * h    # time(s)

    u = x[0, 0]
    v = x[1, 0]
    U = math.sqrt(u*u+v*v)                  # speed
    psi = x[5, 0]                                # heading angle
    beta_crab = custom_functions.ssa(math.atan2(v, u))    # crab angle
    chi = psi + beta_crab                     # course angle

    Vx = U * math.cos(chi)             # ownship velocity in x - direction
    Vy = U * math.sin(chi)             # ownship velocity in y - direction

    # Docking
    total_distance = math.sqrt((x[3, 0] - wpt_pos_x[len(wpt_pos_x)-1]) ** 2 + (x[4 ,0] - wpt_pos_y[len(wpt_pos_y)-1])**2)
    # Goal reached check

    if total_distance <= R_switch:
        alpha = 1e-10
        u_d = custom_functions.sat(u_d - alpha * total_distance, 5)
        print('Docking')
        if total_distance <= 0.3 * R_switch:
            #ownship = shipModel(x[3], x[4], chi, U)
            # hold on
            print('Stop')
            break
    else:
        u_d = U_ref

    # Guidance
    chi_ref, y_e, waypoint_index = custom_functions.LOSchi(x[3, 0], x[4, 0], Delta, R_switch, wpt_pos_x, wpt_pos_y, waypoint_index)

    # Low pass filter
    Ad = np.array([[0, 1, 0], [0, 0, 1],[-wn_d**3, -(2 * zeta_d + 1)*wn_d**2, -(2 * zeta_d + 1)*wn_d]])

    Bd = np.array([[0], [0], [wn_d ** 3]])

    xd_dot = Ad.dot(xd) + Bd * chi_ref
    chi_d = xd[0]
    omega_d = xd[1]

    ## PID control: Course
    chi_error = custom_functions.ssa(chi_d - chi)
    chi_error_dot = (chi_error - chi_error_past) / h
    chi_error_past = chi_error

    ## PID control: Surge
    surge_error = u_d - U
    surge_error_dot = (surge_error - surge_error_past) / h
    surge_error_past = surge_error
    surge_error_sum = surge_error_sum + surge_error * h
    tau_X = Kp_surge * surge_error + Kd_surge * surge_error_dot + Ki_surge * surge_error_sum
    tau_N = Kp * chi_error + Kd * chi_error_dot + Ki * chi_error_sum

    chi_error_sum = chi_error_sum + h * chi_error

    control = Binv.dot([[tau_X[0]], [tau_N[0]]])

    for index in range(0, 2):
        if control[index] > T_max:           # saturation, physical limits
            control[index] = T_max
        elif control[index] < T_min:
            control[index] = T_min

    ## store simulation data in a table
    # t(1), x'(2:7), chi_ref(8), chi_d(9), y_e(10), U_ref(11), u_d(12),
    # control'(13:14)

    # output = np.hstack([t[0,:], x[0,:], chi_ref[0,:], chi_d[0,:],  y_e[0,:], U_ref[0,:], u_d[0,:], np.transpose(control)[0,:]])
    t_stack.append(t)
    x_stack.append(np.transpose(x))
    chi_ref_stack.append(t)
    chi_d_stack.append(chi_d)
    y_e_stack.append(y_e)
    U_ref_stack.append(U_ref)
    u_d_stack.append(u_d)
    control_stack.append(np.transpose(control))

    # Euler integration(k + 1)
    # print(x)
    x = x + h * custom_functions.WAMV_CNU(x, control)
    xd = xd + h * xd_dot

    if i % 10 == 0:
        path_x.append(x[3])
        path_y.append(x[4])
        #clear graph
        plt.cla()
        plt.plot(start_pt[0], start_pt[1], 'r.')
        plt.plot(wpt.x_value, wpt.y_value, 'r.')
        plt.grid()
        plt.ylabel('$y$-position [m]')
        plt.xlabel('$x$-position [m]')
        plt.text(start_pt[0], start_pt[1], 'Start', fontsize=10)
        plt.text(goal_pt[0], goal_pt[1], 'Goal', fontsize=10)
        plt.title('LOS + PID control')
        plt.plot()

        draw_shipmodel(x[3], x[4], x[0], x[1], x[5])
        Lida.lidar_sim(x[3][0], x[4][0], x[5][0], all_obstacle_segments)
        plt.plot(Lida.laser_data_xy)
        # liadarM1 = lia.LidarSimmodel()
        # liadarM1.lidar_sim(x[3], x[4], x[5], [[0, 0, 20, 20], [100, 120, 150, 150]])

        plt.pause(0.001)
        #s = plt.text('Time = %8.2f', t)
        # Time_display = plt.text(10, 200, s, fontsize=10)

        # f_iter = f_iter + 1
        # F(f_iter) = getframe(gcf)  # to get the current frame for movie

# str_video = 'PID_Waypoint_WAMV'
# video = VideoWriter(str_video, 'Motion JPEG AVI')
# video.FrameRate = 5  # (frames per second) this number depends on the sampling time and the number of frames you have
#open(video)
#writeVideo(video, F)
# close(video)

# load('PID_Waypoint_WAMV.mat')

# PLOTS

t = t_stack
u = np.array([x[0,0] for x in x_stack])
v = np.array([x[0,1] for x in x_stack])
r = np.array([x[0,2] for x in x_stack])
U = np.array(np.sqrt(u * u + v * v))     # speed
x = np.array([x[0,3] for x in x_stack])
y = np.array([x[0,4] for x in x_stack])
psi = np.array([x[0,5] for x in x_stack])      # heading angle
v_u_tmp = np.array([math.atan2(v[tmp], u[tmp]) for tmp in range(len(v))])
beta_c = custom_functions.ssa(v_u_tmp)    # crab angle
chi = psi + beta_c           # course angle

chi_ref = chi_ref_stack
chi_d = chi_d_stack[:]
y_e = y_e_stack[:]
U_ref = U_ref_stack[:]
U_d = u_d_stack[:]
Tp = np.array([control[0,0] for control in control_stack])
Ts = np.array([control[0,1] for control in control_stack])

T_max = 400
T_min = -400
T_max = T_max * np.ones([len(t), 1])
T_min = T_min * np.ones([len(t), 1])

plt.figure()
plt.plot(t, T_max, label='T_{max}')
plt.plot(t, T_min, label='T_{min}')
plt.plot(t, Tp, color='g', label='$T_p$')
plt.plot(t, Ts, color='k', label='$T_s$', linewidth=2)
plt.grid()
plt.xlabel('time (s)')
plt.xlim([t[0], t[-1]])
plt.legend()
plt.title('Thrust (N)')

plt.figure()
plt.plot(t, y_e, linewidth=2)
plt.xlabel('time (s)')
plt.title('Cross-track error')
plt.grid()
plt.xlim((t[0], t[-1]))

# 이 밑에 plot이 잘못된 듯
plt.figure()
# plt.plot(t, np.array([ch_r * [180/math.pi] for ch_r in chi_ref]), label='reference')
plt.plot(t, np.array([ch_d * [180/math.pi] for ch_d in chi_d]), '--', label='filtered')
plt.plot(t, np.array([ch * (180/math.pi) for ch in chi]), ':', label='actual')
plt.xlim((t[0], t[-1]))
plt.xlabel('time (s)')
plt.title('Course angle (deg)')
plt.legend()

plt.figure()
plt.plot(t, U_d, label='Reference')
plt.plot(t, U, label='Acutal')
plt.xlim((t[0], t[-1]))
plt.xlabel('time (s)')
plt.ylabel('Surge velocity (m/s)')
plt.grid()
plt.legend()

plt.show()
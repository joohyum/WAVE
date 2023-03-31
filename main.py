# Read the waypoint from informed RRT*
from LidarSimModel import LidarSimmodel
from EnvModel import EnvModel
import waypoints as wpt
import sys
from shipmodel import *
from ShipModel_otter import *
import math
import yaml
import numpy as np
from matplotlib import pyplot as plt

global x_max, y_max, x_min, y_min
global fig

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
            xk = wpt_pos_x[-1]
            yk = wpt_pos_y[-1]

        # Read next waypoint(xk_next, yk_next) from wpt.pos
        n = len(wpt_pos_x)
        if k < n:  # if there are more waypoints, read next one
            xk_next = wpt_pos_x[k]
            yk_next = wpt_pos_y[k]
        else:
            xk_next = wpt_pos_x[-1]
            yk_next = wpt_pos_y[-1]

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

def collision_avoidance(chi_d, r, theta):
    if all(r) == 0 and all(theta) == 0:### 처음 collision_avoidance함수가 실행될 때는 아직 r, theta값에 대한 데이터가 없으므로 이때 최초로 한 번만 돌아감
        safezone = []
        return chi_d, safezone
    else:
        safezone = np.array(r)
        safezone_temp = np.array(safezone)
        for i, r in enumerate(r):
            safezone[i] = r
        for i in range(len(safezone)-1):### 같은 숫자일 때 미소숫자 차이로 Ture가 나오는 경우가 있어 0.0001를 더해서 보정함
            if (safezone[i] > safezone[i + 1] + 0.0001):  ### 배가 바라볼 때를 기준으로 장애물 왼쪽에 마진을 줌(배가 들어가서 회전할 공간을 주기 위해서), 배에 맞게 최적화가 필요할 부분
                if i - 2 <= 0:
                    continue
                else:
                    # safezone_temp[i - 2] = 0
                    safezone_temp[i - 1] = 0
            if (safezone[i] + 0.0001 < safezone[i + 1]):  ### 배가 바라볼 때를 기준으로 장애물 오른쪽에 마진을 줌(배가 들어가서 회전할 공간을 주기 위해서), 배에 맞게 최적화가 필요한 부분
                if i+2 >= len(safezone):
                    continue
                else:
                    safezone_temp[i + 1] = 0
                    safezone_temp[i + 2] = 0
        safezone = safezone_temp
        for i in range(len(safezone)):  ### 위에서 라이다에 장애물이 들어오면 장애물에 마진을 준 양 끝지점에 safezone을 0으로 만들었으니까 라이다에 긁힌 부분 다 0으로 만들어주는 부분
            if safezone[i] + 0.0001 < np.max(safezone):
                safezone[i] = 0
        # 위 코드로 인해서 결국 safezone의 값은 최대값(50)이나 0 밖에 존재하지 않도록 코드를 작성함. 적당한 안전거리를 정해서 코드를 최적화 시켜야함
        for i in range(len(safezone) - 1):  ### chi_d가 safezone이 0인 지역 안에 있나 확인하고, 그 안에 있다면 chi_d를 safezone으로 바꾸는 코드(가능한 각을 적게 틀도록)
            if (chi_d >= theta[i] and chi_d <= theta[i + 1] and (safezone[i] == 0 or safezone[i + 1] == 0)):
                for x in range(int(len(safezone) / 2)):  ### 왼쪽과 오른쪽 중에 어느 쪽으로 회피하는게 이전에 목표했던 chi_d와 가까운지 확인하고 그 쪽으로 수정하는 코드
                    try:
                        if safezone[i - x] > 0:
                            chi_d = theta[i - x]
                            break
                        elif safezone[i + x] > 0:
                            chi_d = theta[i + x]
                            break
                        else:
                            continue
                    except IndexError: ### 위 상황을 만족하는 경우가 없을 경우(가운데 장애물이 크게 있는 경우) 양쪽으로 회피)
                        if i > len(safezone) - i:
                            chi_d = theta[-1]
                            break
                        else:
                            chi_d = theta[0]
                            break

    return chi_d, safezone

def draw_shipmodel(x, y, u, v, psi, path_x, path_y):
    ownship = shipmodel(x, y, psi+np.arctan2(v, u))
    ax.plot(ownship.vehx, ownship.vehy, color='b')
    ax.plot(path_x, path_y, 'b--')

def draw_lidar(x, y, laser_data_xy):
    for laser_data in laser_data_xy:
        ax.plot([x, laser_data[0]], [y, laser_data[1]], ':', color='r', linewidth='1')

def draw_obstacle(connected_components):
    ax.scatter(connected_components[:,0], connected_components[:,1], marker='.', s=1, c='black', alpha=0.2)

def plot_main(fig, wpt_pos_x, wpt_pos_y, connected_components, safezone=[], psi=0, t=[], Ts=[], Tp=[], x=0, y=0, Lida=None):
    # clear graph
    global ax, bx, cx

    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid()
    ax.set_ylabel('$y$-position [m]')
    ax.set_xlabel('$x$-position [m]')
    # ax.text(start_pt[0], start_pt[1], 'Start', fontsize=10)
    # ax.text(goal_pt[0], goal_pt[1], 'Goal', fontsize=10)
    ax.set_title('LOS + PID control')
    ax.plot(wpt_pos_x, wpt_pos_y, 'r.')
    draw_obstacle(connected_components)

    bx.clear()
    bx.set_rlim(0, 50)
    if Lida != None:
        r = np.sqrt((x-Lida.laser_data_xy[:,0])**2 + (y-Lida.laser_data_xy[:,1])**2)
        theta = np.arctan2((Lida.laser_data_xy[:,1]-y), (Lida.laser_data_xy[:,0]-x))
        bx.plot([0, psi], [0, 10], linewidth=2)
        bx.fill(theta, r, color='blue', alpha=0.1)
        if len(safezone) == 36:
            bx.fill(theta, safezone, color='green', alpha=0.1)

    cx.clear()
    cx.set_title('Thrust(N)')
    cx.set_xlabel('time(s)')
    cx.set_ylim([0, 400])
    cx.plot(t, Tp, color='g', label='$T_p$')
    cx.plot(t, Ts, color='k', label='$T_s$')
    cx.legend()
    cx.grid()

    plt.draw()
def click_ck(event):
    global wpt_pos_x, wpt_pos_y
    # button 1: 마우스 좌클릭
    if event.button == 1:
        wpt_pos_x.append(event.xdata)
        wpt_pos_y.append(event.ydata)
        ax.plot(wpt_pos_x, wpt_pos_y, 'r.')
        plt.draw()
    # button 3: 마우스 우클릭
    elif event.button == 3:
        run_sim(x)

def run_sim(x):
    psi_init = math.atan2((wpt_pos_y[1] - wpt_pos_y[0]), (wpt_pos_x[1] - wpt_pos_x[0]))

    # Set ownship initial position at the first waypoint
    wp_index = 0
    x[3, 0] = wpt_pos_x[wp_index]
    x[4, 0] = wpt_pos_y[wp_index]
    x[5, 0] = psi_init

    #라이다 센서 초기값
    r = [0]
    theta = [0]

    # xd = [[0], [0], [0]]
    xd = np.array([[psi_init], [0], [0]])

    chi_error_past = 0
    chi_error_sum = 0  # integral state

    surge_error_past = 0
    surge_error_sum = 0
    Kp_surge = 1500
    Kd_surge = 100
    Ki_surge = 100

    # surge reference speed
    # U_ref = 1.5    successful
    U_ref = 2  # successful

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

    for i in range(1, N + 2):
        t = (i - 1) * h  # time(s)

        u = x[0, 0]
        v = x[1, 0]
        U = math.sqrt(u * u + v * v)  # speed
        psi = x[5, 0]  # heading angle
        beta_crab = custom_functions.ssa(math.atan2(v, u))  # crab angle
        chi = psi + beta_crab  # course angle

        Vx = U * math.cos(chi)  # ownship velocity in x - direction
        Vy = U * math.sin(chi)  # ownship velocity in y - direction

        # Docking
        total_distance = math.sqrt((x[3, 0] - wpt_pos_x[-1]) ** 2 + (x[4, 0] - wpt_pos_y[-1]) ** 2)

        # Goal reached check
        if total_distance <= R_switch:
            alpha = 1e-10
            u_d = custom_functions.sat(u_d - alpha * total_distance, 5)
            print('Docking')
            if total_distance <= 0.3 * R_switch:
                # ownship = shipModel(x[3], x[4], chi, U)
                # hold on
                print('Stop')
                break
        else:
            u_d = U_ref

        # Guidance
        chi_ref, y_e, waypoint_index = custom_functions.LOSchi(x[3, 0], x[4, 0], Delta, R_switch, wpt_pos_x, wpt_pos_y, waypoint_index)

        # Collision Avoidance
        chi_ref, safezone = collision_avoidance(chi_ref, r, theta)

        # Low pass filter
        Ad = np.array([[0, 1, 0], [0, 0, 1], [-wn_d ** 3, -(2 * zeta_d + 1) * wn_d ** 2, -(2 * zeta_d + 1) * wn_d]])

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
            if control[index] > T_max:  # saturation, physical limits
                control[index] = T_max
            elif control[index] < T_min:
                control[index] = T_min

        ## store simulation data in a table
        # t(1), x'(2:7), chi_ref(8), chi_d(9), y_e(10), U_ref(11), u_d(12),
        # control'(13:14)

        # output = np.hstack([t[0,:], x[0,:], chi_ref[0,:], chi_d[0,:],  y_e[0,:], U_ref[0,:], u_d[0,:], np.transpose(control)[0,:]])
        t_stack.append(t)
        # x_stack.append(np.transpose(x))
        # chi_ref_stack.append(t)
        # chi_d_stack.append(chi_d)
        # y_e_stack.append(y_e)
        # U_ref_stack.append(U_ref)
        # u_d_stack.append(u_d)
        control_stack.append(np.transpose(control))
        Tp = np.array([control[0, 0] for control in control_stack])
        Ts = np.array([control[0, 1] for control in control_stack])

        # Euler integration(k + 1)
        # print(x)
        x = x + h * custom_functions.WAMV_CNU(x, control)
        xd = xd + h * xd_dot

        if i % 10 == 0:
            path_x.append(x[3])
            path_y.append(x[4])

            # lida run
            Lida.lidar_sim(x[3][0], x[4][0], x[5][0], all_obstacle_segments)
            r = np.sqrt((x[3] - Lida.laser_data_xy[:, 0]) ** 2 + (x[4] - Lida.laser_data_xy[:, 1]) ** 2)
            theta = np.arctan2((Lida.laser_data_xy[:, 1] - x[4]), (Lida.laser_data_xy[:, 0] - x[3]))

            # 메인 그래프 반복
            plot_main(fig, wpt_pos_x, wpt_pos_y, connected_components, safezone, psi, t_stack, Ts, Tp, x[3], x[4], Lida)

            # shipmodel plot
            draw_shipmodel(x[3], x[4], x[0], x[1], x[5], path_x, path_y)

            # lida, obstacle plot
            draw_obstacle(connected_components)
            draw_lidar(x[3][0], x[4][0], Lida.laser_data_xy)

            plt.pause(0.1)

    # PLOTS

    # t = t_stack
    # u = np.array([x[0, 0] for x in x_stack])
    # v = np.array([x[0, 1] for x in x_stack])
    # r = np.array([x[0, 2] for x in x_stack])
    # U = np.array(np.sqrt(u * u + v * v))  # speed
    # x = np.array([x[0, 3] for x in x_stack])
    # y = np.array([x[0, 4] for x in x_stack])
    # psi = np.array([x[0, 5] for x in x_stack])  # heading angle
    # v_u_tmp = np.array([math.atan2(v[tmp], u[tmp]) for tmp in range(len(v))])
    # beta_c = custom_functions.ssa(v_u_tmp)  # crab angle
    # chi = psi + beta_c  # course angle
    #
    # chi_ref = chi_ref_stack
    # chi_d = chi_d_stack[:]
    # y_e = y_e_stack[:]
    # U_ref = U_ref_stack[:]
    # U_d = u_d_stack[:]
    # Tp = np.array([control[0, 0] for control in control_stack])
    # Ts = np.array([control[0, 1] for control in control_stack])
    #
    # T_max = 400
    # T_min = -400
    # T_max = T_max * np.ones([len(t), 1])
    # T_min = T_min * np.ones([len(t), 1])

    # plt.figure()
    # plt.plot(t, T_max, label='T_{max}')
    # plt.plot(t, T_min, label='T_{min}')
    # plt.plot(t, Tp, color='g', label='$T_p$')
    # plt.plot(t, Ts, color='k', label='$T_s$', linewidth=2)
    # plt.grid()
    # plt.xlabel('time (s)')
    # plt.xlim([t[0], t[-1]])
    # plt.legend()
    # plt.title('Thrust (N)')
    #
    # plt.figure()
    # plt.plot(t, y_e, linewidth=2)
    # plt.xlabel('time (s)')
    # plt.title('Cross-track error')
    # plt.grid()
    # plt.xlim((t[0], t[-1]))
    #
    # plt.figure()
    # plt.plot(t, np.array([ch_d * [180 / math.pi] for ch_d in chi_d]), '--', label='filtered')
    # plt.plot(t, np.array([ch * (180 / math.pi) for ch in chi]), ':', label='actual')
    # plt.xlim((t[0], t[-1]))
    # plt.xlabel('time (s)')
    # plt.title('Course angle (deg)')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(t, U_d, label='Reference')
    # plt.plot(t, U, label='Acutal')
    # plt.xlim((t[0], t[-1]))
    # plt.xlabel('time (s)')
    # plt.ylabel('Surge velocity (m/s)')
    # plt.grid()
    # plt.legend()
    #
    # plt.show()

with open('config/toy1.yaml', encoding='UTF-8') as f:
    _cfg = yaml.safe_load(f)

x_max = _cfg['area']['x_max']
x_min = _cfg['area']['x_min']
y_max = _cfg['area']['y_max']
y_min = _cfg['area']['y_min']

Lida = LidarSimmodel()
Env = EnvModel(env_config_path='./config/toy1.yaml')
all_obstacle_segments, connected_components = Env.update_obstacles()

# Determine Start/Goal point
start_pt = [10, 10]
goal_pt = [490, 490]

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

wpt_pos_x = []
wpt_pos_y = []

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 1, 1)
bx = fig.add_subplot(2, 2, 3, projection='polar')
cx = fig.add_subplot(2, 2, 4)

cid1 = plt.gcf().canvas.mpl_connect('button_press_event', click_ck)
plot_main(fig, wpt_pos_x, wpt_pos_y, connected_components)
plt.show()
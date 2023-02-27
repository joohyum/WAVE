import sys
import math
import numpy as np

class custom_functions:
    def ssa(angle):
        '''
        각도 값을 [-pi pi)값으로 변환해주는 함수
        :return:
        '''
        angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
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

        if k:

            # check if R_switch is smaller than the minimum distance between the waypoints
            if R_switch > min(math.sqrt(np.diff(wpt_pos_x) ** 2 + np.diff(wpt_pos_y) ** 2)):
                print("The distances between the waypoints must be larger than R_switch")
                sys.exit(1)

            # check input parameters
            if (R_switch < 0):
                print("R_switch must be larger than zero")
                sys.exit(1)
            if (Delta < 0):
                print("Delta must be larger than zero")
                sys.exit(1)

            k = 0                # set first waypoint as the active waypoint
            xk = wpt_pos_x[k]
            yk = wpt_pos_y[k]

        # Read next waypoint(xk_next, yk_next) from wpt.pos
        n = len(wpt_pos_x)
        if k < n:         # if there are more waypoints, read next one
            xk_next = wpt_pos_x[k + 1]
            yk_next = wpt_pos_y[k + 1]

        else:             # else, use the last one in the array
            xk_next = wpt_pos_x[len(wpt_pos_x)]
            yk_next = wpt_pos_y[len(wpt_pos_y)]

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

    def WAMV_CNU(states, control):
         Tp = control[0]
         Ts = control[1]

         u = states[0]
         v = states[1]
         r = states[2]
         x = states[3]
         y = states[4]
         psi = states[5]

         xdot = [[-1.1391*u+0.0028*(Tp+Ts)+0.6836],
                 [0.0161*v-0.0052*r+0.002*(Tp-Ts)*2.44/2+0.0068],
                 [8.2861*v-0.9860*r+0.0307*(Tp-Ts*2.44/2+1.3276)],
                 [u*math.cos(psi)-v*math.sin(psi)],
                 [u*math.sin(psi)+v*math.cos(psi)],
                 [r]]

         return xdot
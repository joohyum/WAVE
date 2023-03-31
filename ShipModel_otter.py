import numpy as np

class ShipModel:

    def __init__(self):
        # ship shape
        self.T_n = 1.0  # propeller time constants (s)
        self.veh_info_L = 2.0  # USV Length(m)
        self.veh_info_B = 1.08  # USV Beam(m)
        self.m = 55.0           # mass (kg)
        self.veh_info_buffer = 2.5*self.veh_info_L
        self.veh_info_headDir = 5
        self.veh_info_ssz = 10 # buffer = safe separation zone (ssz)
        self.dt = 0.05
        self.tf = 10

        # trim: theta = -7.5 deg corresponds to 13.5 cm less height aft maximum load
        self.trim_setpoint = 280

        # trim_setpoint is a step input, which is filtered using the state trim_moment
        self.trim_moment = 0

        # Main data
        g   = 9.81         # acceleration of gravity (m/s^2)
        rho = 1025         # density of water

        rg = np.array([0.2, 0, -0.2]) # CG for hull only (m)
        R44 = 0.4 * self.veh_info_B      # radii of gyrations (m)
        R55 = 0.25 * self.veh_info_L
        R66 = 0.25 * self.veh_info_L
        T_yaw = 1          # time constant in yaw (s)
        Umax = 6 * 0.5144  # max forward speed (m/s)

        # Load condition
        self.mp = 25               # payload mass (kg), max value 45 kg
        rp = np.array([0, 0, -0.35])     # location of payload (m)

        # Data for one pontoon
        self.B_pont  = 0.25      # beam of one pontoon (m)
        y_pont  = 0.395     # distance from centerline to waterline area center (m)
        Cw_pont = 0.75      # waterline area coefficient (-)
        Cb_pont = 0.4       # block coefficient, computed from m = 55 kg

        # Inertia dyadic, volume displacement and draft
        nabla = (self.m+self.mp)/rho                          # volume
        self.T = nabla / (2 * Cb_pont * self.B_pont*self.veh_info_L)        # draft
        Ig_CG = self.m * np.diag([R44**2, R55**2, R66**2])     # only hull in CG
        rg = (self.m*rg + self.mp*rp)/(self.m+self.mp)            # CG location corrected for payload
        self.Ig = Ig_CG - self.m * self.Smtrx(rg)**2 - self.mp * self.Smtrx(rp)**2   # hull + payload in CO

        # Experimental propeller data including lever arms
        self.l1 = -y_pont                            # lever arm, left propeller (m)
        self.l2 = y_pont                             # lever arm, right propeller (m)
        self.k_pos = 0.02216/2                       # Positive Bollard, one propeller 
        self.k_neg = 0.01289/2                       # Negative Bollard, one propeller 
        self.n_max =  np.sqrt((0.5*24.4 * g)/self.k_pos)     # maximum propeller rev. (rad/s)
        self.n_min = -np.sqrt((0.5*13.6 * g)/self.k_neg)     # minimum propeller rev. (rad/s)
        # n_max modified
        self.n_max = self.n_max*2
        
        # MRB and CRB (Fossen 2021)
        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
        #               O3       Ig ]
        MRB_CG = np.zeros((6, 6))
        MRB_CG[0:3, 0:3] = (self.m + self.mp) * np.identity(3)
        MRB_CG[3:6, 3:6] = self.Ig

        self.H1 = self.Hmtrx(rg)               # Transform MRB and CRB from the CG to the CO 
        MRB = self.H1.T @ MRB_CG @ self.H1 
        
        # Hydrodynamic added mass (best practise)
        Xudot = -0.1 * self.m    
        Yvdot = -1.5 * self.m 
        Zwdot = -1.0 * self.m 
        Kpdot = -0.2 * self.Ig[0,0]
        Mqdot = -0.8 * self.Ig[1,1] 
        Nrdot = -1.7 * self.Ig[2,2] 

        self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])    
        
        # System mass and Coriolis-centripetal matrices
        self.M = MRB + self.MA 
        
        # Hydrostatic quantities (Fossen 2021)
        Aw_pont = Cw_pont * self.veh_info_L * self.B_pont     # waterline area, one pontoon 
        I_T = 2 * (1/12)*self.veh_info_L*self.B_pont**3 * (6*Cw_pont**3/((1+Cw_pont)*(1+2*Cw_pont))) + 2 * Aw_pont * y_pont**2 
        I_L = 0.8 * 2 * (1/12) * self.B_pont * self.veh_info_L**3 
        KB = (1/3)*(5*self.T/2 - 0.5*nabla/(self.veh_info_L*self.B_pont)) 
        BM_T = I_T/nabla        # BM values
        BM_L = I_L/nabla 
        KM_T = KB + BM_T        # KM values
        KM_L = KB + BM_L 
        KG = self.T - rg[2] 
        GM_T = KM_T - KG        # GM values
        GM_L = KM_L - KG 

        G33 = rho * g * (2 * Aw_pont)       # spring stiffness
        G44 = rho * g *nabla * GM_T 
        G55 = rho * g *nabla * GM_L 

        G_CF = np.diag([0, 0, G33, G44, G55, 0])    # spring stiffness matrix in the CF
        LCF = -0.2 
        self.H2 = self.Hmtrx(np.array([LCF, 0, 0]))                # transform G_CF from the CF to the CO 
        self.G = self.H2.T @ G_CF @ self.H2 

        # Natural frequencies
        w3 = np.sqrt(G33/self.M[2,2])          
        w4 = np.sqrt(G44/self.M[3,3]) 
        w5 = np.sqrt(G55/self.M[4,4]) 

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -24.4 * g / Umax            # specified using the maximum speed        
        Yv = 0 
        Zw = -2 * 0.3 *w3 * self.M[2,2]       # specified using relative damping factors
        Kp = -2 * 0.2 *w4 * self.M[3,3] 
        Mq = -2 * 0.4 *w5 * self.M[4,4] 
        Nr = -self.M[5,5] / T_yaw             # specified using the time constant in T_yaw
        self.ldt = [Xu, Yv, Zw, Kp, Mq, Nr]
        
        # for calculation of ssz
        self.vehx = self.veh_info_L*np.array([0.5,1,0.5,-1,-1,0.5])
        self.vehy = self.veh_info_B*np.array([1,0,-1,-1,1,1])
        self.theta = np.linspace(0,360,361)/180*np.pi 


    # dynamics of vehicle   
    def otter(self, states, n, V_c, beta_c):
        # State and current variables
        nu = states[:6]                                  # velocities
        eta = states[6:]                                 # positions
        U = np.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)       # speed
        u_c = V_c * np.cos(beta_c - eta[-1])            # current surge velocity
        v_c = V_c * np.sin(beta_c - eta[-1])            # current sway velocity
        nu_r = nu - np.array([u_c, v_c, 0, 0, 0, 0])              # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = (self.m+self.mp) * self.Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -self.Smtrx(np.matmul(self.Ig, nu[3:6]))
        CRB = self.H1.T @ CRB_CG @ self.H1 
        CA = self.m2c(self.MA, nu_r) 
        CA[5,0] = 0  # Assume that the Munk moment in yaw can be neglected
        CA[5,1] = 0  # These terms, if nonzero, must be balanced by adding nonlinear damping
        C = CRB + CA 

        # Control forces and moments - with propeller revolution saturation 
        Thrust = np.zeros(2) 
        for i in range(2):
            if n[i] > self.n_max:              # saturation, physical limits
                n[i] = self.n_max  
            elif n[i] < self.n_min:
                n[i] = self.n_min  
            
            if n[i] > 0:                         
                Thrust[i] = self.k_pos * n[i]*abs(n[i])     # positive thrust (N) 
            else :
                Thrust[i] = self.k_neg * n[i]*abs(n[i])     # negative thrust (N) 

        # Control forces and moments
        tau = np.array([Thrust[0]+Thrust[1], 0, 0, 0, 0, -self.l1*Thrust[0]-self.l2*Thrust[1]]) 

        # Linear damping using relative velocities + nonlinear yaw damping
        Xh = self.ldt[0] * nu_r[0] 
        Yh = self.ldt[1] * nu_r[1]  
        Zh = self.ldt[2] * nu_r[2] 
        Kh = self.ldt[3] * nu_r[3] 
        Mh = self.ldt[4] * nu_r[4] 
        Nh = self.ldt[5] * (1 + 10 * abs(nu_r[5])) * nu_r[5] 

        tau_damp = np.array([Xh, Yh, Zh, Kh, Mh, Nh]) 

        # Strip theory: cross-flow drag integrals for Yh and Nh
        tau_crossflow = self.crossFlowDrag(self.veh_info_L, self.B_pont, self.T ,nu_r) 

        # Ballast
        g_0 = np.array([0, 0, 0, 0, self.trim_moment, 0]) 

        # Kinematics
        J = self.eulerang(eta[3],eta[4],eta[5]) 

        # Time derivative of the state vector - numerical integration  see ExOtter.m 
        sum_tau = (tau + tau_damp + tau_crossflow - np.matmul(C, nu_r) - np.matmul(self.G, eta) - g_0) 
        nu_dot = np.matmul(np.linalg.inv(self.M), sum_tau)  # USV dynamics
        x_dot = np.append(nu_dot, J@nu)   
            
        self.trim_moment = self.trim_moment + 0.05 * (self.trim_setpoint - self.trim_moment) 
        
        return  x_dot                                   

    def Smtrx(self, a):
            """
            S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
            The cross product satisfies: a x b = S(a)b. 
            """
            S = np.array([ [ 0,    -a[2],    a[1] ],
                           [ a[2],   0,     -a[0] ],
                           [-a[1],   a[0],      0 ]  ])

            return S
    
    def Hmtrx(self, r):
        """
        H = Hmtrx(r) computes the 6x6 system transformation matrix
        H = [eye(3)     S'
            zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)
        If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
        CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
        force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG 
        """

        H = np.identity(6,float)
        H[0:3, 3:6] = self.Smtrx(r).T

        return H 

    def m2c(self, M, nu):
        """
        C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
        mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
        """

        M = 0.5 * (M + M.T)     # systematization of the inertia matrix

        if (len(nu) == 6):      #  6-DOF model
        
            M11 = M[0:3,0:3]
            M12 = M[0:3,3:6] 
            M21 = M12.T
            M22 = M[3:6,3:6] 
        
            nu1 = nu[0:3]
            nu2 = nu[3:6]
            dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
            dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

            #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
            #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
            C = np.zeros( (6,6) )    
            C[0:3,3:6] = -self.Smtrx(dt_dnu1)
            C[3:6,0:3] = -self.Smtrx(dt_dnu1)
            C[3:6,3:6] = -self.Smtrx(dt_dnu2)
                
        else:   # 3-DOF model (surge, sway and yaw)
            #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
            #      0             0             M(1,1)*nu(1)
            #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
            C = np.zeros( (3,3) ) 
            C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
            C[1,2] =  M[0,0] * nu[0] 
            C[2,0] = -C[0,2]       
            C[2,1] = -C[1,2]
            
        return C

    def crossFlowDrag(self, L,B,T,nu_r):
        """
        tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
        integrals for a marine craft using strip theory. 
        M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
        """

        def Hoerner(B,T):
            """
            CY_2D = Hoerner(B,T)
            Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of beam 
            B and draft T.The data is digitized and interpolation is used to compute 
            other data point than those in the table
            """
            
            # DATA = [B/2T  C_D]
            DATA1 = np.array([
                0.0109,0.1766,0.3530,0.4519,0.4728,0.4929,0.4933,0.5585,0.6464,0.8336,
                0.9880,1.3081,1.6392,1.8600,2.3129,2.6000,3.0088,3.4508, 3.7379,4.0031 
                ])
            DATA2 = np.array([
                1.9661,1.9657,1.8976,1.7872,1.5837,1.2786,1.2108,1.0836,0.9986,0.8796,
                0.8284,0.7599,0.6914,0.6571,0.6307,0.5962,0.5868,0.5859,0.5599,0.5593 
                ])

            CY_2D = np.interp( B / (2 * T), DATA1, DATA2 )
                
            return CY_2D

        rho = 1026               # density of water
        n = 20                   # number of strips

        dx = L/20             
        Cd_2D = Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve

        Yh = 0
        Nh = 0
        xL = -L/2
        
        for i in range(0,n+1):
            v_r = nu_r[1]             # relative sway velocity
            r = nu_r[5]               # yaw rate
            Ucf = abs(v_r + xL * r) * (v_r + xL * r)
            Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
            Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
            xL += dx
            
        tau_crossflow = np.array([0, Yh, 0, 0, 0, Nh],float)

        return tau_crossflow

    def eulerang(self, phi,theta,psi):
        def Rzyx(phi,theta,psi):
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            cth  = np.cos(theta)
            sth  = np.sin(theta)
            cpsi = np.cos(psi)
            spsi = np.sin(psi)
            R = np.array([[cpsi*cth,   -spsi*cphi+cpsi*sth*sphi,   spsi*sphi+cpsi*cphi*sth],
                          [spsi*cth,   cpsi*cphi+sphi*sth*spsi,   -cpsi*sphi+sth*spsi*cphi],
                          [-sth,       cth*sphi,                   cth*cphi ]])
            return R
        
        def Tzyx(phi,theta):
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            cth  = np.cos(theta)
            sth  = np.sin(theta)
            T = np.array([[1,  sphi*sth/cth,  cphi*sth/cth],
                          [0,  cphi,           -sphi],
                          [0,  sphi/cth,      cphi/cth]])
            return T
        J1 = Rzyx(phi,theta,psi)
        J2 = Tzyx(phi,theta)
        J1_0 = np.hstack([J1, np.zeros((3,3))])
        J2_0 = np.hstack([np.zeros((3,3)), J2])
        J = np.vstack([J1_0, J2_0])
        return J

    def ship_shape(self, pos_x, pos_y, course_angle, speed):

        posx =pos_x + self.vehx*np.cos(course_angle) - self.vehy*np.sin(course_angle)
        posy = pos_y + self.vehx*np.sin(course_angle) + self.vehy*np.cos(course_angle)

        ssz_r = self.veh_info_ssz  
        ssz_x = ssz_r*np.cos(self.theta) + pos_x
        ssz_y = ssz_r*np.sin(self.theta) + pos_y

        # heading line depending on speed
        hl_x = np.array([pos_x, pos_x + (self.veh_info_headDir * speed) * np.cos(course_angle)]) 
        hl_y = np.array([pos_y, pos_y + (self.veh_info_headDir * speed) * np.sin(course_angle)])

        ship_data = [posx, posy, ssz_x, ssz_y, hl_x, hl_y]

        return ship_data
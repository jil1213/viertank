import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import warnings

from libmimo import mimo_rnf, mimo_acker, poly_transition
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete


class struct:
    pass;

class LinearizedSystem:
    """Realizes a linear dynamic system (A,B,C,D) with its operating point (x_equi,u_equi,y_equi)
    """    
    
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi):
        """
        Args:
            A (ndarray): system map
            B (ndarray): input map
            C (ndarray): output map
            D (ndarray): feedthrough map
            x_equi (ndarray): equilibrium state
            u_equi (ndarray): equilibrium input
            y_equi (ndarray): equilibrium output
        """        
        self.A=A
        self.B=B
        self.C=C
        self.D=D
        self.x_equi=x_equi
        self.u_equi=u_equi
        self.y_equi=y_equi

    #Ackermannformel
    def acker(self,eigs):
        """Compute controller gains by the the Ackermann-Formula

        Args:
            eigenvalues (list): list of ndarrays or lists containing the eigenvalues of the subsystems

        Returns:
            ndarray: controller gains
        """        
        return mimo_acker(self.A,self.B,eigs)

    #Berechnung des Ausgangs
    def output(self,t,x, controller):
        """Compute system output

        Args:
            t (float or ndarray): actual time
            x (ndarry): absolute system state 
            controller (function): controller to compute input u from x

        Returns:
            float or ndarray: actual output value C(x-x_equi)+D(u-u_equi)+y_equi
        """        
        
        #Regler auswerten (wichtig falls Durchgriff existiert)
        u = controller(t,x)        
        if x.ndim==1:
            y = self.C@(x-self.x_equi) + self.y_equi + self.D@(u-self.u_equi)
        else:
            x_equi=self.x_equi.reshape((self.x_equi.shape[0],1))
            u_equi=self.u_equi.reshape((self.u_equi.shape[0],1))
            y_equi=self.y_equi.reshape((self.y_equi.shape[0],1))
            y = self.C@(x-x_equi)+y_equi+self.D@(u-u_equi)
        return y

class DiscreteLinearizedSystem(LinearizedSystem):
    """Realizes a linear discrete time dynamic system (A,B,C,D) with its 
       operating point (x_equi,u_equi,y_equi) and sampling time Ta
    """
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi,Ta):
        """_summary_

        Args:
            A (ndarray): system map
            B (ndarray): input map
            C (ndarray): output map
            D (ndarray): feedthrough map
            x_equi (ndarray): equilibrium state
            u_equi (ndarray): equilibrium input
            y_equi (ndarray): equilibrium output
            Ta: sampling time
        """        
        super().__init__(A,B,C,D,x_equi,u_equi,y_equi)
        self.Ta=Ta

    #Quadratisch optimaler Regler
    def lqr(self,Q,R):
        """Compute controller gains 

        Args:
            Q (ndarray): positive semi-definite state weight matrix
            R (ndarray): positive definite input weight matrix

        Returns:
            _type_: _description_
        """        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Reglerverstärkungen berechnet werden
        K_lqr=np.zeros((R.shape[0],Q.shape[0]))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return K_lqr

    def rest_to_rest_trajectory(self,ya,yb,N,kronecker,maxderi=None):
        return DiscreteFlatnessBasedTrajectory(self,ya,yb,N,kronecker,maxderi)

class ContinuousLinearizedSystem(LinearizedSystem):
    """Realizes a linear continuous time dynamic system (A,B,C,D) with its 
       operating point (x_equi,u_equi,y_equi) 
    """
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi):
        super().__init__(A,B,C,D,x_equi,u_equi,y_equi)
        
    def discretize(self,Ta):
        assert(Ta>0)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        ##bitte korrigieren und die korrekten diskreten Systemmatrizen bestimmen
        sysd = DiscreteLinearizedSystem(self.A,self.B,self.C,self.D,self.x_equi,self.u_equi,self.y_equi,Ta)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return sysd

    #Quadratisch optimaler Regler
    def lqr(self,Q,R):
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Reglerverstärkungen berechnet werden
        K_lqr=np.zeros((R.shape[0],Q.shape[0]))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return K_lqr

    def rest_to_rest_trajectory(self,ya,yb,T,kronecker,maxderi=None):
        return ContinuousFlatnessBasedTrajectory(self,ya,yb,T,kronecker,maxderi)
        
    #linearisiertes Zustandsraumodell des Systems
    def model(self,t,x,controller): 
        """Implements the Modell 
        
        dx = A*(x-x0)+B*(u-u0)
        
        Inputs are computed using the Callback-Function `controller`.
        Variables are absolute and not related to the equilibrium. 

        Args:
            t (float): actual time
            x (ndarray): system state (absolute)
            controller (function): Callback for computing the input

        Returns:
            ndarray: time-derivatives of the system state
        """

        dim_u=self.u_equi.shape[0]
        dim_x=self.x_equi.shape[0]
        dim_y=self.y_equi.shape[0]
        
        #Auslenkung aus Ruhelage berechnen
        x_rel=np.array(x).flatten()-self.x_equi
        x_rel=x_rel.flatten()

        #Eingang auswerten
        u=controller(t,x)
        u=np.array(u).flatten()
        
        #Abweichung aus Ruhelage
        u_rel=u-self.u_equi

        #Differentialgleichung auswerten 
        dx=(self.A@x_rel+self.B@u_rel).flatten()


        #Ableitungen zurückgeben
        return dx


    

class VierTank:
    """Modell des Tanksystems"""

    def __init__(self,st):
        self.AS = np.array([[0,st.AS12,st.AS13,0,0],
                            [0,0,st.AS23,st.AS24,0],
                            [0,0,0,st.AS34,st.AS30],
                            [0,0,0,0,st.AS40]])
        self.st = st
        
    def get_input_from_flow(self,i,q):
        """Compute input voltage from a given pump volume flow.

        Args:
            i (int): index pump/tank (1-2)
            q (float): positive volume flow

        Returns:
            float: pump voltage
        """       
        if i==1:
            uA0=self.st.uA01
            Ku=self.st.Ku1
        else:
            uA0=self.st.uA02
            Ku=self.st.Ku2
        if np.isscalar(q):
            if q<=0:
                u=0
            else:
                u=uA0+q/Ku
        else:
            u=np.zeros_like(q)
            _ = (q>0)
            u[_]=uA0+q[_]/Ku
        
        return u

    def get_level_from_outflow(self,i,j,q):
        """Compute the water level from a given outflow.

        Args:
            i (int): source tank 1-4
            j (int): target tank 1-5, 5 is reservoir
            q (float): positive flow  

        Returns:
            float: fluid level 
        """        
        x=(q/self.AS[i-1,j-1])**2/2/self.st.g-self.st.hV
        return x;

    def get_outflow_from_level(self,i,j,x):
        """Compute outflow from a given water level

        Args:
            i (int): index source tank (1-4)
            j (int): index destination tank (1-5), 5 is reservoir
            x (float or ndarray): water level in source tank

        Returns:
            float or ndarray: positive flow
        """        
        g=self.st.g
        AS=self.AS[i-1,j-1]
        hV=self.st.hV
        if np.isscalar(x):
            if x<=-hV:
                q=0
            else:
                q=np.sqrt(2*g*(x+hV))*AS
        else:
            q=np.zeros_like(x)
            _=x>-hV
            q[_]=np.sqrt(2*g*(x[_]+hV))*AS
        return q 
   
    def get_flow_from_input(self,i,u):
        """Compute pump volume flow from pump voltage. The characteristics are linear in the range `[uA0,uAmax]` and constant outside
           this interval.

        Args:
            i (int): index pump/tank (1-2)
            u (float): pump voltage

        Returns:
            float: flow
        """       
        if i==1:
            uA0=self.st.uA01
            Ku=self.st.Ku1
        else:
            uA0=self.st.uA02
            Ku=self.st.Ku2
        if np.isscalar(u):
            if u<=uA0:
                q=0
            elif u<self.st.uAmax:
                q=Ku*(u-uA0)
            else:
                q=Ku*(self.st.uAmax-uA0)
        else:
            q=np.zeros_like(u)
            _ = (u>uA0)
            q[_]=Ku*(u[_]-uA0)
            
            _ = (u>self.st.uAmax)
            q[_]=Ku*(self.st.uAmax-uA0)        
        return q

    def model(self,t,x,controller):
        """Implements the model of the four-tank system by returning the derivatives of the state 
        for a given state and a given input. The inputs are passed in a callback function in order, for example 
        be able to implement a controller, for example.

        Args:
            t (float): current time
            x (ndarray): system state (fluid level)
            controller (function): callback for computing the pump voltages 

        Returns:
            ndarray: time derivative of the system state
        """

        # Regler auswerten
        u = controller(t,x) #Eingänge werden aus dem Controller berechnet

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Füllstandsunterschiede und Abflüsse berechnen, um Abflussraten zwischen den Tanks zu bestimmen
        # diese werden implizit in der Funktion get_outflow_from_level berechnet
        q01 = self.get_flow_from_input(1, u[0])
        q02 = self.get_flow_from_input(2, u[1])
        q12 = self.get_outflow_from_level(1, 2, x[0])
        q13 = self.get_outflow_from_level(1, 3, x[0])
        q23 = self.get_outflow_from_level(2, 3, x[1])
        q24 = self.get_outflow_from_level(2, 4, x[1])
        q34 = self.get_outflow_from_level(3, 4, x[2])
        q30 = self.get_outflow_from_level(3, 5, x[2])
        q40 = self.get_outflow_from_level(4, 5, x[3])

        # Fallunterscheidung für Füllstand
        if x[0] > -self.st.hV and x[0] < 0:
            A1 = self.st.AT  # Standard-Tankquerschnitt (oberhalb Ventilniveau)
        else:
            A1 = 2 * self.st.AR  # Querschnitt der Abflussschläuche (unterhalb Ventilniveau)

        if x[1] > -self.st.hV and x[0] < 0:
            A2 = self.st.AT
        else:
            A2 = 2 * self.st.AR

        if x[2] > -self.st.hV and x[0] < 0:
            A3 = self.st.AT
        else:
            A3 = 2 * self.st.AR

        if x[3] > -self.st.hV and x[0] < 0:
            A4 = self.st.AT
        else:
            A4 = self.st.AR
        
        
        # Aufstellen der Bewegungsgleichungen nach Gl. 4a-d
        dx=np.zeros_like(x)
      
        dx[0] = (q01 - q12 - q13) / A1
        dx[1] = (q02 + q12 - q23 - q24) / A2
        dx[2] = (q13 + q23 - q34 - q30) / A3
        dx[3] = (q24 + q34 - q40) / A4

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return dx
  
    def output(self, t, x, controller = None):
        """Berechnet die Ausgänge des Systems

        Args:
            t (float oder ndarray): Aktuelle Zeit, kann ein bestimmter Zeitpunkt oder ein Vektor sein
            x (ndarray): Zustände als 4xN Array (die Zustände der 4 Tanks über N Zeitpunkte) 
            controller (function, optional): Optionaler Controller, wird derzeit noch nicht verwendet

        Returns:
            ndarray: Ausgänge y.
        """
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Ausgänge berechnet werden
        # die Ausgänge lassen sich direkt aus den Füllständen der Tanks 3 und 4 bestimmen
        # die if Schleife unterscheidet die Fälle ob t ein konkreter zeitpunkt oder ein Vektor ist
        if np.isscalar(t):
            y=np.zeros((2,))
            y[0] = x[2]  # Füllstand in Tank 3
            y[1] = x[3]  # Füllstand in Tank 4
        else:
            y=np.zeros((2,t.shape[0]))
            y[0] = x[2]  # Füllstand in Tank 3
            y[1] = x[3]  # Füllstand in Tank 4
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return y


    def equilibrium(self,y):
        """
        Berechnet die Ruhelage (x̄, ū) für ein gegebenes Ziel ȳ
        
        Args:
            y (ndarray): Gewünschte Ausgänge [y1, y2]

        Returns:
            struct: Enthält x̄, ū, ȳ sowie einen Statuscode
        """
        equi = struct()

        x=np.zeros((4,))
        u=np.zeros((2,))
   
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier die sollten die korrekten Ruhelagen in Abhängigkeit des zugehörigen Ausgangs berechnet werden
        #Folgende Werte sollen in der Variable equi.status hinterlegt werden  (gegebenenfalls auch mehrere mit oder verküpft)
        #  0: Ruhelage zulässig
        #  1: negativer Volumenstrom durch ein Ventil oder Füllhöhe eines Tanks kleiner 0 (Flüssigkeit nur im Ablauf über Ventil)
        #  2: maximaler Füllstand eines Tanks überschritten
        #  4: maximale Pumpenspannung überschritten

        # Ausgang in der Angabe gegeben zu 
        x[2] = y[0]
        x[3] = y[1]
        #x1 und x2 können über die Füllstände Q24 und Q13 bestimmt werden (diese lassen sich aus Tank 3 und 4 berechnen)
        Q24 = self.get_outflow_from_level(4,5,x[3])-self.get_outflow_from_level(3,4,x[2]) # Q40-Q34 = Q24 -> Ausgänge y für Q40 und Q34 bekannt, Bestimmung für x2
        x[1] = self.get_level_from_outflow(2,4, Q24)
        Q13 = self.get_outflow_from_level(3,4,x[2])+self.get_outflow_from_level(3,5,x[2])-self.get_outflow_from_level(2,3,x[1]) # Q34 + Q30 -Q23 = Q13 -> Bestimmung für x1
        x[0] = self.get_level_from_outflow(1,3, Q13)
        # Eingänge können über Bewegungsgleichungen 4a und b bestimmt werden
        Q01 = self.get_outflow_from_level (1,2,x[0])+self.get_outflow_from_level (1,3,x[0]) # Q12 + Q13 = Q01
        u[0] = self.get_input_from_flow(1,Q01)
        Q02 = self.get_outflow_from_level(2,3,x[1])+self.get_outflow_from_level(2,4,x[1])-self.get_outflow_from_level (1,2,x[0]) # Q23 + Q24 - Q12 = Q02
        u[1] = self.get_input_from_flow(2,Q02)
      

        equi.status = 0
        if np.any(x < 0):        # Prüfen, ob ein Wert in x < 0 ist equi.status |= 1 # Status mit 1 verknüpfen (Bitwise-OR)
                 equi. status|= 1 #status mit 1 verknüpfen 
        
        if np.any(x > self.st.hT):  # Prüfen, ob ein Wert in x › Tankhöhe ist
                 equi. status|= 2   # Status mit 2 verknüpfen (Bitwise-OR)

        if np.any(u > self.st.uAmax):   # Prüfen, ob ein Wert in u › 12 ist
                 equi.status|= 4        # Status mit 4 verknüpfen (Bitwise-OR)
        equi.x=x 
        equi.u=u
        equi.y=y

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        return equi
        
    def linearize(self,equi, debug=False):
        # Berechnung der Systemmatrizen des linearen Systems

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        dx = self.model(0, equi.x, lambda t, x: equi.u)

        # Jacobi berechnen jacobian geht nur mit sympy matrizen- Wie hier umsetzung?
        #A = dx.jacobian(equi.x)
        #B = dx.jacobian(equi.u)
        
        #einsetzen der Ruhelage 1 
        #A = A.subs(equi.x)
        #B = B.subs(equi.u)
        #Hier die sollten die korrekten Matrizen angegeben werden
        A=np.zeros((4,4))
        B=np.zeros((4,2))
        C=np.zeros((2,4))
        D=np.zeros((2,2))

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        
        return ContinuousLinearizedSystem(A,B,C,D,equi.x,equi.u,equi.y)
    

    def verify_linearization(self,linear_model,eps_x=1e-6,eps_u=1e-6):
        """Compare linear state space model with its approximate Taylor linearization of non-linear model
        Args:
            linear_model (LinearizedModel):  the linearized model to comparey

        Returns:
            tuple: approximated system matrices

        """
        A_approx=np.zeros_like(linear_model.A)
        B_approx=np.zeros_like(linear_model.B)
        C_approx=np.zeros_like(linear_model.C)
        D_approx=np.zeros_like(linear_model.D)
        x_equi=linear_model.x_equi
        u_equi=linear_model.u_equi
        y_equi=linear_model.y_equi
        dim_x=len(x_equi)
        dim_u=len(u_equi)
        dim_y=len(y_equi)

        if np.isscalar(eps_x):
            eps_x=np.ones((dim_x,))*eps_x
        if np.isscalar(eps_u):
            eps_u=np.ones((dim_u,))*eps_u

        for jj in range(dim_x):
            _fnu=lambda t,x:u_equi
            x_equi1=np.array(x_equi)
            x_equi2=np.array(x_equi)
            x_equi2[jj]+=eps_x[jj]
            x_equi1[jj]-=eps_x[jj]
            #print("Ruhelage:",x_equi)
            dx=(self.model(0,x_equi2,_fnu)-self.model(0,x_equi1,_fnu))/2/eps_x[jj]
            A_approx[:,jj]=dx
            dy=(self.output(0,x_equi2,_fnu)-self.output(0,x_equi1,_fnu))/2/eps_x[jj]
            C_approx[:,jj]=dy
            
        error_A=np.abs(linear_model.A-A_approx)
        idx= np.unravel_index(np.argmax(error_A, axis=None), error_A.shape)
        print("Maximaler absoluter Fehler in Matrix A Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_A[idx[0],idx[1]])+".")
        scale_A=np.hstack([np.where(abs(A_approx[:,jj:jj+1]) > eps_x[jj], abs(A_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
        error_rel_A=error_A/scale_A
        idx= np.unravel_index(np.argmax(error_rel_A, axis=None), error_rel_A.shape)
        print("Maximaler relativer Fehler in Matrix A Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_A[idx[0],idx[1]])+".")

        error_C=np.abs(linear_model.C-C_approx)
        idx= np.unravel_index(np.argmax(error_C, axis=None), error_C.shape)
        print("Maximaler absoluter Fehler in Matrix C Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_C[idx[0],idx[1]])+".")
        scale_C=np.hstack([np.where(abs(C_approx[:,jj:jj+1]) > eps_x[jj], abs(C_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
        error_rel_C=error_C/scale_C
        idx= np.unravel_index(np.argmax(error_rel_C, axis=None), error_rel_C.shape)
        print("Maximaler relativer Fehler in Matrix C Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_C[idx[0],idx[1]])+".")
        
        for jj in range(dim_u):
            url1=np.array(u_equi)
            url2=np.array(u_equi)
            eps_u[jj]=1e-6
            url1[jj]-=eps_u[jj]
            url2[jj]+=eps_u[jj]
            _fnu1=lambda t,x:url1
            _fnu2=lambda t,x:url2
            dx=(self.model(0,x_equi,_fnu2)-self.model(0,x_equi,_fnu1))/2/eps_u[jj]
            B_approx[:,jj]=dx
            dy=(self.output(0,x_equi,_fnu2)-self.output(0,x_equi,_fnu1))/2/eps_x[jj]
            D_approx[:,jj]=dy
        error_B=np.abs(linear_model.B-B_approx)
        idx= np.unravel_index(np.argmax(error_B, axis=None), error_B.shape)
        print("Maximaler absoluter Fehler in Matrix B Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_B[idx[0],idx[1]])+".")
        scale_B=np.hstack([np.where(abs(B_approx[:,jj:jj+1]) > eps_x[jj], abs(B_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
        error_rel_B=error_B/scale_B
        idx= np.unravel_index(np.argmax(error_rel_B, axis=None), error_rel_B.shape)
        print("Maximaler relativer Fehler in Matrix B Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_B[idx[0],idx[1]])+".")

        error_D=np.abs(linear_model.D-D_approx)
        idx= np.unravel_index(np.argmax(error_D, axis=None), error_D.shape)
        print("Maximaler absoluter Fehler in Matrix D Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_D[idx[0],idx[1]])+".")
        scale_D=np.hstack([np.where(abs(D_approx[:,jj:jj+1]) > eps_x[jj], abs(D_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
        error_rel_D=error_D/scale_D
        idx= np.unravel_index(np.argmax(error_rel_D, axis=None), error_rel_D.shape)
        print("Maximaler relativer Fehler in Matrix D Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_D[idx[0],idx[1]])+".")
        return A_approx, B_approx, C_approx

    def generate_model_test_data(self,filename,count):
        import pickle
        x=np.random.rand(4,count)*10-self.st.hV
        u=np.random.rand(2,count)*self.st.uAmax
        dx=np.zeros_like(x)
        y=np.zeros_like(u)
        for ii in range(count):
            dx[:,ii]=self.model(0,x[:,ii],lambda t,x: u[:,ii])
            y[:,ii]=self.output(0,x[:,ii],lambda t,x: u[:,ii])
            
        with open(filename, 'wb') as f: 
            pickle.dump([x, u, dx, y], f)
        f.close()

    def verify_model(self,filename):
        import pickle
        with open(filename,'rb') as f:
            x, u, dx_load, y_load = pickle.load(f)
        f.close()
        dx=np.zeros_like(x)
        y=np.zeros_like(u)
        count=dx.shape[1]
        for ii in range(count):
            dx[:,ii]=self.model(0,x[:,ii],lambda t,x: u[:,ii])
            y[:,ii]=self.output(0,x[:,ii],lambda t,x: u[:,ii])
        error_dx_abs_max=np.max(np.linalg.norm(dx-dx_load,2,axis=0))
        error_dx_rel_max=np.max(np.linalg.norm(dx-dx_load,2,axis=0)/np.linalg.norm(dx_load,2,axis=0))
        error_y_abs_max=np.max(np.linalg.norm(y-y_load,2,axis=0))
        error_y_rel_max=np.max(np.linalg.norm(y-y_load,2,axis=0)/np.linalg.norm(y_load,2,axis=0))
        dx_load_max=np.max(np.linalg.norm(dx_load,2,axis=0))
        y_load_max=np.max(np.linalg.norm(y_load,2,axis=0))
        print("Maximaler absoluter Fehler in Modellgleichung (euklidische Norm):",error_dx_abs_max)
        print("Maximaler relativer Fehler in Modellgleichung (euklidische Norm):",error_dx_rel_max)
        print("Maximaler absoluter Fehler in Ausgangsgleichung (euklidische Norm):",error_y_abs_max)
        print("Maximaler relativer Fehler in Ausgangsgleichung (euklidische Norm):",error_y_rel_max)
        
    

class ContinuousFlatnessBasedTrajectory:
    """Zeitkontinuierliche flachheitsbasierte Trajektorien-Planung zum Arbeitspunktwechsel 
    für das lineare zeitkontinuierliche Modelle, die aus der Linearisierung im Arbeitspunkt abgeleitete worden sind.

    Args:
       ya, ye (numpy.array):  Anfangs- und Endwerte den Ausgang (absolut)
       T (float): Überführungszeit
       linearized_system: Entwurfsmodel
       kronecker: zu verwendende Steuerbarkeitsindizes
       maxderi: maximal Differenzierbarkeitsanforderungen für flachen Ausgang (None entspricht maxderi=kronecker)
    """
    def __init__(self,linearized_system,ya,yb,T,kronecker,maxderi=None):
        self.linearized_system=linearized_system
        self.T=T
        ya_rel=np.array(ya)-linearized_system.y_equi
        yb_rel=np.array(yb)-linearized_system.y_equi
        self.kronecker=np.array(kronecker,dtype=int)
        if maxderi==None:
            self.maxderi=self.kronecker
        else:
            self.maxderi=self.maxderi
            
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier bitte benötigte Zeilen wieder "dekommentieren" und Rest löschen
        #self.A_rnf, Brnf, Crnf, self.M, self.Q, S = mimo_rnf(linearized_system.A, linearized_system.B, linearized_system.C, kronecker)
        self.A_rnf=np.zeros((4,4))
        self.M=np.eye(2)
        self.Q=np.eye(4)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        #Umrechnung stationäre Werte zwischen Ausgang und flachem Ausgang
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Anfangs und Endwerte für den flachen Ausgang berechnet werden
        #Achtung: Hier sollten alle werte relativ zum Arbeitspunkt angegeben werden

        self.eta_a=np.zeros_like(ya_rel)
        self.eta_b=np.zeros_like(yb_rel)

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    # Trajektorie des flachen Ausgangs
    def flat_output(self,t,index,derivative):
        tau = t  / self.T
        if derivative==0:
            return self.eta_a[index] + (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau,0,self.maxderi[index])
        else:
            return (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau,derivative,self.maxderi[index])/self.T**derivative 

    #Zustandstrajektorie 
    def state(self,t):
        tv=np.atleast_1d(t)
        dim_u=np.size(self.linearized_system.u_equi)
        dim_x=np.size(self.linearized_system.x_equi)
        dim_t=np.size(tv)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        state=np.zeros((dim_x,dim_t))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        if np.isscalar(t):
            state = state.flatten()
            
        return state

    #Ausgangstrajektorie
    def output(self,t):
        tv=np.atleast_1d(t)
        dim_u=np.size(self.linearized_system.u_equi)
        dim_y=np.size(self.linearized_system.y_equi)
        dim_x=np.size(self.linearized_system.x_equi)

        x_abs=self.state(tv)
        u_abs=self.input(tv)
        x_rel=x_abs-self.linearized_system.x_equi.reshape((dim_x,1))
        u_rel=u_abs-self.linearized_system.u_equi.reshape((dim_u,1))
        y_rel=self.linearized_system.C@x_rel+self.linearized_system.D@u_rel
        y_abs=y_rel+self.linearized_system.y_equi.reshape((dim_y,1))
        if (np.isscalar(t)):
            y_abs=result[:,0]
        return y_abs

    #Eingangstrajektorie
    def input(self,t):
        tv=np.atleast_1d(t)
        dim_u=np.size(self.linearized_system.u_equi)
        eta=list()
        for index in range(dim_u):
            eta=eta+[self.flat_output(tv,index,deri) for deri in range(self.kronecker[index])]
        xrnf=np.vstack(eta)
        v=-self.A_rnf[self.kronecker.cumsum()-1,:]@xrnf
        for jj in range(self.kronecker.shape[0]):
            v[jj,:]+=self.flat_output(tv,jj,self.kronecker[jj])
        result=(np.linalg.inv(self.M)@v)+self.linearized_system.u_equi.reshape((dim_u,1))
        if (np.isscalar(t)):
            result=result[:,0]
        return result

class DiscreteFlatnessBasedTrajectory:
    """Zeitdiskrete flachheitsbasierte Trajektorien-Planung zum Arbeitspunktwechsel 
    für lineare Modelle, die aus der Linearisierung im Arbeitspunkt abgeleitet worden sind.

    Args:
       ya, ye (numpy.array):  Anfangs- und Endwerte den Ausgang (absolut)
       T (float): Überführungszeit
       linearized_system: zeitdiskretes Entwurfsmodel
       kronecker: zu verwendende Steuerbarkeitsindizes
       maxderi: maximale Differenzierbarkeitsanforderungen für die Komponenten des flachen Ausgangs (bei None werden die Kronecker-Indizes gewählt; maxderi=kronecker)
    """
    def __init__(self,linearized_system,ya,yb,N,kronecker,maxderi=None):
        self.linearized_system=linearized_system
        self.N=N

        dim_u=np.size(linearized_system.u_equi)
        dim_x=np.size(linearized_system.x_equi)

        #Abstand von der Ruhelage berechnen
        ya_rel=np.array(ya)-linearized_system.y_equi
        yb_rel=np.array(yb)-linearized_system.y_equi
        self.kronecker=np.array(kronecker,dtype=int)

        #Glattheitsanforderungen an Trajektorie
        if maxderi==None:
            self.maxderi=self.kronecker
        else:
            self.maxderi=self.maxderi
            
        #Matrizen der Regelungsnormalform holen
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier bitte benötigte Zeilen wieder "einkommentieren" und Rest löschen
        #self.A_rnf, Brnf, Crnf, self.M, self.Q, S = mimo_rnf(linearized_system.A, linearized_system.B, linearized_system.C, kronecker)
        self.A_rnf=np.zeros((3,3))
        self.M=np.eye(2)
        self.Q=np.eye(3)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        #Umrechnung stationäre Werte zwischen Ausgang und flachem Ausgang
        
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Anfangs und Endwerte für den flachen Ausgang berechnet werden
        #Achtung: Hier sollten alle Werte relativ zum Arbeitspunkt angegeben werden

        self.eta_a=np.zeros_like(ya_rel)
        self.eta_b=np.zeros_like(yb_rel)

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        
    def flat_output(self,k,index,shift=0):
        """Berechnet zeitdiskret die Trajektorie des flachen Ausgangs

        Parameter:
        ----------
        k: diskrete Zeitpunkte als Vektor oder Skalar
        shift: Linksverschiebung der Trajektorie
        index: Komponente des flachen Ausgangs"""

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Werte für die um "shift" nach links verschobene Trajektorie des flachen Ausgangs zurückgegeben werden

        eta= np.zeros_like(k)

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return eta

    #Zustandstrajektorie 
    def state(self,k):
        kv=np.atleast_1d(k)
        dim_u=np.size(self.linearized_system.u_equi)
        dim_x=np.size(self.linearized_system.x_equi)
        dim_k=np.size(k)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        state=np.zeros((dim_x,dim_k))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        if np.isscalar(k):
            state = state.flatten()
        return state

    #Ausgangstrajektorie
    def output(self,k):
        kv=np.atleast_1d(k)
        dim_y=np.size(self.linearized_system.y_equi)
        dim_x=np.size(self.linearized_system.x_equi)
        dim_u=np.size(self.linearized_system.u_equi)

        x_abs=self.state(kv)
        u_abs=self.input(kv)
        x_rel=x_abs-self.linearized_system.x_equi.reshape((dim_x,1))
        u_rel=u_abs-self.linearized_system.u_equi.reshape((dim_u,1))
        y_rel=self.linearized_system.C@x_rel+self.linearized_system.D@u_rel
        y_abs=y_rel+self.linearized_system.y_equi.reshape((dim_y,1))
        if (np.isscalar(k)):
            y_abs=result[:,0]
        return y_abs

    #Eingangstrajektorie
    def input(self,k):

        #Zeitargument vektorisieren
        kv=np.atleast_1d(k)

        #Anzahl der Eingänge
        dim_u=np.size(self.linearized_system.u_equi)

        #Anzahl der Zeitpunkte
        dim_k=np.size(kv)
        
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        input=np.zeros((dim_u,dim_k))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        if (np.isscalar(k)):
            input=input[:,0]
        return input


def plot_results(t,x,u):
    plt.figure(figsize=(15,7))
    plt.subplot(3,2,1,ylabel="$x_1$ in cm")
    plt.grid()
    leg=["Soll","Ist"]
    for v in x:
        plt.plot(t,v[0,:]*100)
    plt.legend(leg)
    plt.subplot(3,2,2,ylabel="$x_2$ in cm")
    plt.grid()
    for v in x:
        plt.plot(t,v[1,:]*100)
    plt.legend(leg)
    plt.subplot(3,2,3,ylabel="$x_3$ in cm")
    plt.grid()
    for v in x:
        plt.plot(t,v[2,:]*100)
    plt.legend(leg)
    plt.subplot(3,2,4,ylabel="$x_4$ in cm")
    plt.grid()
    for v in x:
        plt.plot(t,v[3,:]*100)
    plt.legend(leg)
    plt.subplot(3,2,5,ylabel="Eingang 1 l/s")
    plt.grid()
    for v in u:
        plt.plot(t,v[0,:])
    plt.legend(leg)
    plt.subplot(3,2,6,ylabel="Eingang 2 l/s")
    plt.grid()
    for v in u:
        plt.plot(t,v[1,:])
    plt.legend(leg)

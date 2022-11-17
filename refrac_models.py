"""
Class to make direc and inverse refraction models from 3 layers.
"""
import numpy as np
import matplotlib.pyplot as plt

class refrac_models:
    def __init__(self, dx, ng, z1, z2, v1, v2, v3, s_type="no_source"):
        self.dx = dx
        self.ng = ng
        self.s_n = {"hammer":0.11, "hydraulic_hammer":0.07, "shot_gun":0.03, "no_source":0}
        self.s_type = s_type
        self.noise_level = self.s_n[self.s_type]
        self.z1 = z1
        self.z2 = z2
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    
    def direc_model(self):
        x_t = np.column_stack((np.arange(self.dx,self.dx*self.ng+self.dx,self.dx), np.zeros((self.ng,3)))) 
        x_t[:,1] = x_t[:,0]/self.v1
        if self.v1<self.v2: x_t[:,2] = x_t[:,0]/self.v2 + 2*self.z1*np.cos(np.arcsin(self.v1/self.v2))/self.v1
        else: x_t[:,2] = x_t[:,0]/self.v1
        if self.v2<self.v3: x_t[:,3] = x_t[:,0]/self.v3 + 2*self.z1*np.cos(np.arcsin(self.v1/self.v3))/self.v1 + 2*self.z2*np.cos(np.arcsin(self.v2/self.v3))/self.v2
        else: x_t[:,3] = x_t[:,2] 
        x_t = self.source_noise(x_t)
        return x_t

    def inverse_model(self, x_t, x1, xo1, x2, xo2, x3, xo3):
        x_picks = np.column_stack((x_t[:,0], np.min(x_t[:,1:],axis=1)))
        rec_1, v1, _ = self.rec_refra(x_picks, x1, xo1)
        rec_2, v2, a2 = self.rec_refra(x_picks, x2, xo2)
        rec_3, v3, a3 = self.rec_refra(x_picks, x3, xo3)
        if v1 < v2: z1 = (a2*v1)/(2*np.cos(np.arcsin(v1/v2)))
        else: z1 = 0
        if v2 < v3: z2 = (a3-(2*z1*np.cos(np.arcsin(v1/v3))/v1))*v2/(2*np.cos(np.arcsin(v2/v3)))
        else: z2 = z1
        return v1,v2,v3,z1,z2,rec_1,rec_2,rec_3

    def rec_refra(self,data, x, x_ofset):
        param_rec = np.polyfit(data[x:x_ofset,0],data[x:x_ofset,1],1)
        rec = data[x:(x_ofset),0]*param_rec[0] + param_rec[1]
        return rec,1/param_rec[0],param_rec[1]

    def source_noise(self, x_t):
        x_t[:,1:] = x_t[:,1:] + x_t[:,1:]*np.random.default_rng().uniform(low=-self.noise_level,high=self.noise_level,size=(self.ng,1))
        return x_t

class shot_refrac(refrac_models):
    def __init__(self, dx, ng, z1, z2, v1, v2, v3, dv=6, long_w=0.08,picks= False, s_type='no_source'):
        super().__init__(dx, ng, z1, z2, v1, v2, v3, s_type)
        self.dt = 0.001
        self.w = self.sin_pulse 
        self.dv = dv
        self.picks = picks
        self.long_w = long_w
    
    def graf_shot(self):
        x_t = self.direc_model()
        s_matrix = self.make_seismic_matrix(x_t)
        t = np.arange(0,s_matrix.shape[0]*self.dt,self.dt)
        np.random.seed(42)

        _,ax = plt.subplots(1,figsize = (20,10))
        ax.set_title('Shot Gather', fontsize=32)
        ax.set_ylabel('Time [ms]', fontsize=23)
        ax.set_xlabel('Offset \n[m]', fontsize=23)
        ax.axis([0,self.ng*self.dx*self.dv,
                max(np.arange(0,np.max(x_t[:,1:])+0.005,self.dt)),
                min(np.arange(0,np.max(x_t[:,1:])+0.005,self.dt))]) 
        ax.set_yticks(np.arange(0,np.max(x_t[:,1:])+0.005,(np.max(x_t[:,1:])+0.005)/10))
        ax.set_yticklabels(np.int32(np.arange(0,np.max(x_t[:,1:])+0.005,(np.max(x_t[:,1:])+0.005)/10)*1000),fontsize=20)
        ax.set_xticks(np.arange(0,self.ng*self.dx*self.dv,self.ng*self.dx*self.dv/10))
        ax.set_xticklabels(np.arange(self.dx,self.ng*self.dx+self.dx,(self.ng*self.dx+self.dx)/10,dtype=int),fontsize=20)

        for i in range(self.ng):
            tr = 10*np.convolve(s_matrix[:,i],self.w(self.long_w,amp=2.5,db=40),mode='same')+i*self.dx*self.dv
            tr += np.random.normal(0,self.noise_level*15,s_matrix.shape[0])
            ax.plot(tr,t,color='k',lw=.5)
            tr[tr<i*self.dx*self.dv]=i*self.dx*self.dv
            ax.fill_betweenx(t,tr,i*self.dx*self.dv,color='k',lw=0.0001,alpha=0.85)

            
        if self.picks: 
            ax.plot(x_t[:,0]*self.dv-self.dx*self.dv,np.min(x_t[:,:],axis=1),'.',color='r',markersize=15)
            plt.show()
            _,ax1 = plt.subplots(1,figsize = (20,10))
            ax1.set_title('Tiempos de Arribo', fontsize=28)
            ax1.set_ylabel('Time [ms]',fontsize=23)
            ax1.set_xlabel('Offset \n[m]',fontsize=23)
            ax1.set_yticks(np.arange(0,np.max(x_t[:,1:])+0.005,(np.max(x_t[:,1:])+0.005)/10))
            ax1.set_yticklabels(np.int32(np.arange(0,np.max(x_t[:,1:])+0.005,(np.max(x_t[:,1:])+0.005)/10)*1000),fontsize=20)
            ax1.set_xticks(np.arange(self.dx*self.dv,self.ng*self.dx*self.dv+self.dx,(self.ng*self.dx*self.dv+self.dx*self.dv)/10))
            ax1.set_xticklabels(np.arange(self.dx,self.ng*self.dx+self.dx,(self.ng*self.dx+self.dx)/10,dtype=int),fontsize=20)
            ax1.plot(self.direc_model()[:,0]*self.dv,np.min(self.direc_model()[:,1:],axis=1),'.',color='r',markersize=15)
            plt.show()
        return plt.show()      

    def make_seismic_matrix(self, x_t):
        l_t = int((np.max(x_t[:,1:])/self.dt)+0.005/self.dt)
        l_w = int(self.long_w/self.dt)
        l_t=l_w + 1 if l_t < l_w else l_t 
        s_matrix = np.zeros((l_t,self.ng))
        s_matrix[(np.round(x_t[:,1]/self.dt).astype(int),np.round(x_t[:,0]/self.dx).astype(int)-1)] = 0.5
        s_matrix[(np.round(x_t[:,2]/self.dt).astype(int),np.round(x_t[:,0]/self.dx).astype(int)-1)] = 0.5
        s_matrix[(np.round(x_t[:,3]/self.dt).astype(int),np.round(x_t[:,0]/self.dx).astype(int)-1)] = 0.5
        s_matrix[0,:] = 0
        return s_matrix

    def wavelet_source(self, f=30., long = 0.10, ph=90.):
        dt =self.dt
        l=np.arange(0.0,long,dt)
        w1=(1.-2.*((np.pi*f*(l-long/2))**2.))*np.exp(-1.*((np.pi*f*(l-long/2))**2.))
        rw=np.fft.fft(w1)
        rw[0:int((len(l)/2))]=rw[0:int(len(l)/2)]*(1j)
        rw[int(len(l)/2):len(l)]=rw[int(len(l)/2):len(l)]*(-1j)
        hw=np.fft.ifft(rw).real
        w=(np.cos(np.pi*ph/180.)*w1+np.sin(np.pi*ph/180.)*hw)
        return w/max(abs(w)) 

    def sinc(self, f=30, long=0.04):
        impulse = np.zeros(int(long/self.dt))
        impulse[int(long/2/self.dt):] = np.sinc(np.pi*f*(np.arange(0.0,long/2,self.dt)-long/2))
        return impulse

    def sin_pulse(self, f=120, long=0.08, amp=1.3, db=30):
        f=120
        impulse = np.zeros(int(long/self.dt))
        impulse[int(long/2/self.dt):] = -(amp-db*np.arange(0.0,long/2,self.dt))*np.sin(np.pi*f*(np.arange(0.0,long/2,self.dt)-long/2))
        return  impulse

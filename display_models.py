"""
Class to make Refractions Displays:
"""
from ipywidgets import widgets
import matplotlib.pyplot as plt 
import numpy as np
from IPython.display import display
import refrac_models

class DisplayDirec:
    def __init__(self,dx=5,ng=48,v1=2000, v2=3000, v3=4000, 
                z1=10, z2=20,s_type='no_source',picks=False):
        self.dx = dx
        self.ng = ng
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.z1 = z1
        self.z2 = z2
        self.s_type = s_type
        self.picks = picks

    def shot_display(self):
        l_Dx = widgets.Label("Dist Geóf")
        Dx = widgets.BoundedIntText( value=self.dx, min=1, max=50, step=1)
        Dx.layout.width='60%'
        l_Ng = widgets.Label('Núm Geóf')
        Ng = widgets.BoundedIntText( value=self.ng, min=1, max=96, step=1)
        Ng.layout.width='60%'
        l_So = widgets.Label('Fuente')
        So = widgets.Dropdown( value ='no_source', options = ['no_source', 'hammer', 'hydraulic_hammer', 'shot_gun'])
        So.layout.width='90%' 
        Pi = widgets.Checkbox(value=False,description='Picks',disabled=False,indent=False)       
        Pi.layout.width='60%' 
        l_Z1 = widgets.Label('Espesor 1')
        Z_1 = widgets.BoundedIntText( value=self.z1, min=5, max=100, step=1)
        Z_1.layout.width='60%'
        l_Z2 = widgets.Label('Espesor 2')
        Z_2 = widgets.BoundedIntText( value=self.z2, min=5, max=100, step=1)
        Z_2.layout.width='60%'
        l_V1 = widgets.Label('Velocidad 1')
        V_1 = widgets.BoundedIntText( value=self.v1, min=1000, max=5000, step=100)
        V_1.layout.width='70%'  
        l_V2 = widgets.Label('Velocidad 2')
        V_2 = widgets.BoundedIntText( value=self.v2, min=1000, max=5000, step=100)
        V_2.layout.width='70%' 
        l_V3 = widgets.Label('Velocidad 3')
        V_3 = widgets.BoundedIntText( value=self.v3, min=1000, max=5000, step=100)
        V_3.layout.width='70%'

        def interac(Dx,Ng,So,Pi,Z_1,Z_2,V_1,V_2,V_3):          
            model = refrac_models.shot_refrac(dx=Dx,ng=Ng,v1=V_1, v2=V_2, v3=V_3, 
                z1=Z_1, z2=Z_2, s_type=So, picks=Pi)
            model.graf_shot()

        controls_ad = widgets.VBox([l_Dx, Dx, l_Ng, Ng,
                                 l_So, So, Pi]) 
        controls_gm = widgets.VBox([l_Z1, Z_1,
                                 l_Z2, Z_2, l_V1, V_1,
                                 l_V2, V_2, l_V3, V_3])

        output = widgets.interactive_output(interac,{'Dx': Dx, 'Ng': Ng, 
                        'So':So, 'Pi':Pi, 'Z_1':Z_1, 'Z_2':Z_2, 'V_1':V_1,
                        'V_2':V_2, 'V_3':V_3})
        ui = widgets.HBox([controls_gm,output,controls_ad])
        return display(ui)

class DisplayInverse():
    def __init__(self, dx=5,ng=48,v1=2000, v2=3000, v3=4000, 
                z1=10, z2=20, s_type='shot_gun'):
        self.dx = dx
        self.ng = ng
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.z1 = z1
        self.z2 = z2
        self.s_type = s_type
        self.picks = True 
        self.model = refrac_models.shot_refrac(dx=self.dx, ng=self.ng, s_type=self.s_type,
                                        v1=self.v1, v2=self.v2, v3=self.v3, z1=self.z1, z2=self.z2)    
        self.x_t = self.model.direc_model()

    def inverse_display(self):
        s_matrix = self.model.make_seismic_matrix(self.x_t)
        t = np.arange(0,s_matrix.shape[0]*self.model.dt,self.model.dt)
        np.random.seed(42)

        _,ax = plt.subplots(figsize = (24,12))
        ax.set_title('Shot Gather', fontsize=32)
        ax.set_yticks(np.arange(0,np.max(self.x_t[:,1:])+0.005,(np.max(self.x_t[:,1:])+0.005)/10))
        ax.set_yticklabels(np.int32(np.arange(0,np.max(self.x_t[:,1:])+0.005,(np.max(self.x_t[:,1:])+0.005)/10)*1000),fontsize=20)                   
        ax.set_ylabel('Time [ms]',fontsize=23)
        ax.set_xlabel('Offset \n[m]',fontsize=23)
        ax.set_xticks(np.arange(0,self.ng*self.dx*self.model.dv,self.ng*self.dx*self.model.dv/10))
        ax.set_xticklabels(np.arange(self.dx,self.ng*self.dx+self.dx,(self.ng*self.dx+self.dx)/10,dtype=int),fontsize=23)           
        ax.axis([0,self.ng*self.dx*self.model.dv,
                max(np.arange(0,np.max(self.x_t[:,1:])+0.005,self.model.dt)),
                min(np.arange(0,np.max(self.x_t[:,1:])+0.005,self.model.dt))])    

        for i in range(self.ng):
            tr = 10*np.convolve(s_matrix[:,i],self.model.w(amp=2.5,db=40),mode='same')+i*self.dx*self.model.dv
            tr += np.random.normal(0,self.model.noise_level*15,s_matrix.shape[0])
            ax.plot(tr,t,color='k',lw=.5)
            tr[tr<i*self.dx*self.model.dv]=i*self.dx*self.model.dv
            ax.fill_betweenx(t,tr,i*self.dx*self.model.dv,color='k',lw=0.0001,alpha=0.85)
        ax.plot(self.x_t[:,0]*self.model.dv-self.dx*self.model.dv,np.min(self.x_t[:,:],axis=1),'.',color='r',markersize=15)
        plt.show()

        l_Dx1 = widgets.Label('Directo',layout=widgets.Layout(display="flex", justify_content="center"))
        Dx1 = widgets.IntRangeSlider( value=[0, 8], min=0, max=self.model.ng, step=1,layout=widgets.Layout(display="flex", justify_content="center"))
        l_Dx2 = widgets.Label('Refractor 1',layout=widgets.Layout(display="flex", justify_content="center"))
        Dx2 = widgets.IntRangeSlider( value=[9, 24], min=0, max=self.model.ng, step=1,layout=widgets.Layout(display="flex", justify_content="center"))
        l_Dx3 = widgets.Label('Refractor 2',layout=widgets.Layout(display="flex", justify_content="center"))
        Dx3 = widgets.IntRangeSlider( value=[25, 47], min=0, max=self.model.ng, step=1,layout=widgets.Layout(display="flex", justify_content="center"))        
        def interac_1(Dx1,Dx2,Dx3):  
            _,ax1 = plt.subplots(figsize = (22,12))
            ax1.set_title('Tiempos de Arribo', fontsize=28)
            ax1.set_yticks(np.arange(0,np.max(self.x_t[:,1:])+0.005,(np.max(self.x_t[:,1:])+0.005)/10))
            ax1.set_yticklabels(np.int32(np.arange(0,np.max(self.x_t[:,1:])+0.005,(np.max(self.x_t[:,1:])+0.005)/10)*1000),fontsize=20)
            ax1.set_ylabel('Time [ms]',fontsize=23)
            ax1.set_xlabel('Offset \n[m]',fontsize=23)
            ax1.plot(self.x_t[:,0]*self.model.dv,np.min(self.x_t[:,1:],axis=1),'.',color='r',markersize=15)
            ax1.set_xticks(np.arange(self.dx*self.model.dv,self.ng*self.dx*self.model.dv+self.dx,(self.ng*self.dx*self.model.dv+self.dx*self.model.dv)/10))
            ax1.set_xticklabels(np.arange(self.dx,self.ng*self.dx+self.dx,(self.ng*self.dx+self.dx)/10,dtype=int),fontsize=23) 
            ax1.plot(self.x_t[:,0]*self.model.dv,np.min(self.x_t[:,1:],axis=1),'.',color='r',markersize=15)

            inverse = self.model.inverse_model(self.x_t, x1=Dx1[0], xo1=Dx1[1], x2=Dx2[0], xo2=Dx2[1], x3=Dx3[0], xo3=Dx3[1])
            ax1.plot(self.x_t[Dx1[0]:Dx1[1],0]*self.model.dv,inverse[5],'-',color='b')
            ax1.plot(self.x_t[Dx2[0]:Dx2[1],0]*self.model.dv,inverse[6],'-',color='g')
            ax1.plot(self.x_t[Dx3[0]:Dx3[1],0]*self.model.dv,inverse[7],'-',color='c')   
            return plt.show()
        def interac_2(Dx1,Dx2,Dx3):
            inverse = self.model.inverse_model(self.x_t, x1=Dx1[0], xo1=Dx1[1], x2=Dx2[0], xo2=Dx2[1], x3=Dx3[0], xo3=Dx3[1])
            _,ax2 = plt.subplots(figsize = (22,12))
            ax2.invert_yaxis()
            ax2.set_title('Modelo', fontsize=28)
            ax2.set_yticks(np.arange(0,int(inverse[3]+inverse[4]+10),2))
            ax2.set_yticklabels(np.arange(0,int(inverse[3]+inverse[4]+10),2),fontsize=20)
            ax2.set_ylabel('Profundidad [m]', fontsize=23)
            ax2.set_xticks([]) 

            ax2.fill_between(np.arange(0,50,1),np.zeros(50)+inverse[3],0,color='aquamarine')  
            ax2.fill_between(np.arange(0,50,1),np.zeros(50)+inverse[4]+inverse[3],np.zeros(50)+inverse[3],color='whitesmoke')  
            ax2.fill_between(np.arange(0,50,1),np.zeros(50)+inverse[4]+inverse[3],np.zeros(50)+inverse[3]+inverse[4]+10,color='salmon')   
            if inverse[0]<inverse[1]: 
                ax2.text(22.5, inverse[3]/2-0.5, "V1 = "+str(int(inverse[0])),fontsize=23)
                ax2.text(22.5, inverse[3]+inverse[4]/2-0.5, "V2 = "+str(int(inverse[1])),fontsize=23)
            if inverse[1]<inverse[2]: ax2.text(22.5, inverse[3]+inverse[4]+5, "V3 = "+str(int(inverse[2])),fontsize=23)
            return plt.show()
               
        test1 = widgets.VBox([l_Dx1,Dx1])
        test2 = widgets.VBox([l_Dx2,Dx2])
        test3 = widgets.VBox([l_Dx3,Dx3])
        controls = widgets.HBox([test1,test2,test3])
        output_1 = widgets.interactive_output(interac_1,{'Dx1': Dx1,'Dx2': Dx2,'Dx3': Dx3})
        output_2 = widgets.interactive_output(interac_2,{'Dx1': Dx1,'Dx2': Dx2,'Dx3': Dx3})
        ui = widgets.VBox([output_1,controls,output_2])
        return display(ui)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import torch
import NeuralNetwork
import numpy as np

#---------------------------------------------------------------------------------------------------------
#Re-plot graphic
def print_axis(ax):
    ejeX = [-15,15]
    ejeY = [-15,15]
    zeros = [0,0]
    ax.plot(ejeX, zeros, 'k')
    ax.plot(zeros, ejeY, 'k')
    plt.xlim(-10,10)
    plt.ylim(-10,10)


#---------------------------------------------------------------------------------------------------------
#Data classification using pytorch
def data_classification(X,d,neurons,epoch_p,lerning_rate,min_error, ax, ax_e, canvas, canvas_e, error_gui, epoch_gui):
    
    X_T = torch.tensor(X,dtype=torch.float)
    D_T = torch.tensor(d,dtype=torch.float)
    NN = NeuralNetwork.NeuralNetwork(neurons)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=NN.parameters(), lr=lerning_rate)
    e=1
    errors=[]
    epoch = 0

    while epoch<epoch_p and e>min_error:
        y = NN(X_T)
        e = loss_fn(y,D_T)
        optimizer.zero_grad()
        e.backward()
        optimizer.step()
        errors.append(float(e))

        epoch+=1
        
        error_gui.config(text="Error: "+str(float(e)))
        epoch_gui.config(text="Epoca: "+str(epoch))

    for i in range(len(X)):
        if d[i][0] == 1:
            ax.plot(X[i][0],X[i][1], '.b')
        else:
            ax.plot(X[i][0],X[i][1], '.r')


    v = np.linspace(-10, 10, 100)

    xc, yc = np.meshgrid(v,v)
    zc=[]
    for i in range(len(xc)):
        temp = torch.tensor(list(zip(xc[i],yc[i])), dtype=torch.float) 
        y = NN(temp)
        zc.append(y.detach().numpy().flatten())
    ax.contourf(xc, yc, zc, 100, cmap='coolwarm')

    ax_e.cla()
    ax_e.plot(errors)
    ax_e.set_xticklabels([])

    print_axis(ax)
    canvas.draw()
    canvas_e.draw()



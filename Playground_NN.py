import matplotlib.pyplot as plt
from tkinter import *
import torch
import torch.nn as nn
import torch.optim as optim
import NeuralNetwork
import numpy as np

#---------------------------------------------------------------------------------------------------------
#Re-plot graphic
def print_axis(ax):
    ejeX = [0,1]
    ejeY = [0,1]
    zeros = [0,0]
    ax.plot(ejeX, zeros, 'k')
    ax.plot(zeros, ejeY, 'k')
    plt.xlim(0,1)
    plt.ylim(0,1)


#---------------------------------------------------------------------------------------------------------
#Data classification using pytorch
def data_classification(X, Y, neurons, epoch_p, lerning_rate, min_error, ax, ax_e, canvas, canvas_e, error_gui, epoch_gui):
    X = np.array(X)
    Y = np.array(Y)
    X_Tensor = torch.tensor(X,dtype=torch.float)
    Y_Tensor = torch.tensor(Y,dtype=torch.float)
    NN = NeuralNetwork.NeuralNetwork(neurons)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(NN.parameters(), lr=lerning_rate)

    errors=[]
    for epoch in range(epoch_p):
        output = NN(X_Tensor)
        output[output<0.5] = 0
        output[output>=0.5] = 1
        loss = loss_fn(output, Y_Tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        errors.append(float(loss))
        
        error_gui.config(text="Error: "+str(float(loss)))
        epoch_gui.config(text="Epoca: "+str(epoch))

        if (epoch + 1) % 50 == 0:
            ax.cla()
            ax_e.cla()

            Y_S = Y.squeeze()
            plt.scatter(X[Y_S == 0][:, 0], X[Y_S == 0][:, 1], color='red', label='Class 0')
            plt.scatter(X[Y_S == 1][:, 0], X[Y_S == 1][:, 1], color='blue', label='Class 1')
            
            xxyy = np.arange(0, 1.1, 0.01)
            X_, Y_ = np.meshgrid(xxyy, xxyy)
            Grid = np.column_stack((X_.ravel(), Y_.ravel()))
            Grid_Tensor = torch.tensor(Grid, dtype=torch.float)
            with torch.no_grad():
                Grid_Out = NN(Grid_Tensor)
                Grid_Pred = Grid_Out.reshape(X_.shape)

            ax.contourf(X_, Y_, Grid_Pred.numpy(), alpha=0.5, cmap='coolwarm')

            ax_e.plot(errors)
            ax_e.set_xticklabels([])
            
            print_axis(ax)
            canvas.draw()
            canvas_e.draw()


    ax.cla()
    ax_e.cla()
    Y_S = Y.squeeze()
    plt.scatter(X[Y_S == 0][:, 0], X[Y_S == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[Y_S == 1][:, 0], X[Y_S == 1][:, 1], color='blue', label='Class 1')
    


    xxyy = np.arange(0, 1.1, 0.01)
    X_, Y_ = np.meshgrid(xxyy, xxyy)
    Grid = np.column_stack((X_.ravel(), Y_.ravel()))
    Grid_Tensor = torch.tensor(Grid, dtype=torch.float)
    with torch.no_grad():
        Grid_Out = NN(Grid_Tensor)
        Grid_Pred = Grid_Out.reshape(X_.shape)

    ax.contourf(X_, Y_, Grid_Pred.numpy(), alpha=0.5, cmap='coolwarm')

    ax_e.plot(errors)
    ax_e.set_xticklabels([])

    print_axis(ax)
    canvas.draw()
    canvas_e.draw()

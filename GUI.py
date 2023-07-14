import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import threading
import Playground_NN as PGNN

#---------------------------------------------------------------------------------------------------------
#Plot data
def plot_point(event):
    ix, iy = event.xdata, event.ydata
    X.append((ix, iy))
    if red_or_blue:
        d.append([1])
        ax.plot(ix,iy, '.b')
    else:
        d.append([0])
        ax.plot(ix,iy, '.r')
    canvas.draw()

#---------------------------------------------------------------------------------------------------------
#Switch data class
def change_color():
    global red_or_blue
    red_or_blue = not(red_or_blue)
    if red_or_blue:
        red_red_button.config(text="Blue", background="blue")
    else:
        red_red_button.config(text="Red", background="red")

#---------------------------------------------------------------------------------------------------------
#Clean values and graphic
def clean_screen():
    global X, d
    X = []
    d = []
    ax.cla()
    ax_e.cla()
    ax_e.set_xticklabels([])
    PGNN.print_axis(ax)
    canvas.draw()
    canvas_e.draw()
    
#---------------------------------------------------------------------------------------------------------
#Initializing values
X = []
d = []
red_or_blue = True

fig_e, ax_e= plt.subplots(facecolor='#8D96DA')
ax_e.set_xticklabels([])
fig, ax= plt.subplots(facecolor='#8D96DA')

mainwindow = Tk()
canvas_e = FigureCanvasTkAgg(fig_e, master = mainwindow)
canvas_e.get_tk_widget().place(x=600, y=460, width=300, height=130)
PGNN.print_axis(ax)

#---------------------------------------------------------------------------------------------------------
#Initializing GUI
mainwindow.geometry('910x600')
mainwindow.wm_title('Red neuronal multicapa')
eta = StringVar(mainwindow, 0)
epoch = StringVar(mainwindow, 0)
error = StringVar(mainwindow, 0)
func_value = IntVar(mainwindow, 0)
num_neurons = IntVar(mainwindow, 0)

#---------------------------------------------------------------------------------------------------------
#Place all components
canvas = FigureCanvasTkAgg(fig, master = mainwindow)
canvas.get_tk_widget().place(x=10, y=10, width=580, height=580)
fig.canvas.mpl_connect('button_press_event', plot_point)

Eta_label = Label(mainwindow, text = "Eta: ")
Eta_label.place(x=750, y=60)
Eta_entry = Entry(mainwindow, textvariable=eta)
Eta_entry.place(x=750, y=80) 

Epoch_label = Label(mainwindow, text = "Num. Epoch: ")
Epoch_label.place(x=600, y=110)
Epoch_entry = Entry(mainwindow, textvariable=epoch)
Epoch_entry.place(x=600, y=130)

error_label = Label(mainwindow, text = "Min. Error: ")
error_label.place(x=750, y=110)
error_entry = Entry(mainwindow, textvariable=error)
error_entry.place(x=750, y=130)

neurons_label = Label(mainwindow, text = "Num. Neuronas: ")
neurons_label.place(x=600, y=60)
neurons_entry = Entry(mainwindow, textvariable=num_neurons)
neurons_entry.place(x=600, y=80)

start_button = Button(mainwindow, text="Go!", command=lambda:threading.Thread(target=lambda:PGNN.data_classification(X,
                                                                                                                   d,
                                                                                                                   int(num_neurons.get()),
                                                                                                                   float(epoch.get()),
                                                                                                                   float(eta.get()),
                                                                                                                   float(error.get()), 
                                                                                                                   ax, 
                                                                                                                   ax_e, 
                                                                                                                   canvas, 
                                                                                                                   canvas_e,
                                                                                                                   error_actual,
                                                                                                                   epoch_count)).start())
start_button.place(x=600, y=210)

red_red_button = Button(mainwindow, text="Blue", background="blue", width=10, command=change_color)
red_red_button.place(x=600, y=240)

clean_button = Button(mainwindow, text="Clean", command=clean_screen)
clean_button.place(x=600, y=300)

epoch_count = Label(mainwindow, text="Epoca: 0")
epoch_count.place(x=600, y=420)

error_actual = Label(mainwindow, text="Error: 0")
error_actual.place(x=700, y=420)

#---------------------------------------------------------------------------------------------------------
#Show GUI
mainwindow.mainloop()
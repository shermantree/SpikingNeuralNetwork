
import tkinter
import tkinter.font
from tkinter import messagebox
from engine import snn
from util import dataloader
from util import weightloader
from decimal import Decimal

class Mainwindow():
    def __init__(self):
        self.model = None
        self.window = tkinter.Tk()
        self.canvas = tkinter.Canvas(self.window, width = 320, height = 240, relief="solid", bd=0)

        self.window.title("Spiking Neural Network Simulator by TAEHYUNG KIM")
        self.window.geometry("1080x640+100+100")
        self.window.resizable =(False, False)

        self.cmem = None
        self.vth = None
        self.spilkewidth = None
        self.spikeenergy = None
        self.singleinputtime = None

        self.cmemfield = tkinter.Entry(self.window, width=10)
        self.vthfield = tkinter.Entry(self.window, width=10)
        self.spikewidthfield = tkinter.Entry(self.window, width=10)
        self.singleinputtimefield = tkinter.Entry(self.window, width=10)
        self.spikeenergyfield = tkinter.Entry(self.window, width=10)
        self.conductancefilefield = tkinter.Entry(self.window, width=10)

        self.networklist = tkinter.Listbox(self.window, selectmode='extended', width = 15, height=0)


        self.individualcorrect = [0,0,0,0,0,0,0,0,0,0]
        self.individualnumber = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.individualbar = []
        self.totalacc = 0
        self.samplenumber = 0
        self.correct = 0

        self.acc = tkinter.StringVar()
        self.totalacclabel = tkinter.Label(self.window, textvariable=self.acc, width = 10, height = 1, relief = 'solid')
        self.num = tkinter.StringVar()
        self.currentsamplelabel = tkinter.Label(self.window, textvariable = self.num, width = 10, height = 1, relief = 'solid')
        self.individualacc = []
        self.energyneuronlabelval = tkinter.StringVar()
        self.energysynapselabelval = tkinter.StringVar()
        self.energytotallabelval = tkinter.StringVar()
        self.poweraveragelabelval = tkinter.StringVar()
        self.energyneuronlabel = tkinter.Label(self.window, textvariable=self.energyneuronlabelval, width=10, height=1, relief='solid')
        self.energysynapselabel = tkinter.Label(self.window, textvariable=self.energysynapselabelval, width=10, height=1, relief='solid')
        self.energytotallabel = tkinter.Label(self.window, textvariable=self.energytotallabelval, width=10, height=1, relief='solid')
        self.poweraveragelabel = tkinter.Label(self.window, textvariable=self.poweraveragelabelval, width=10, height=1,
                                              relief='solid')

        self.energyneuron = 0
        self.energysynapse = 0

        font = tkinter.font.Font(family="맑은 고딕", size=8, slant="italic")
        for i in range(len(self.individualcorrect)):
            self.individualacc.append(tkinter.StringVar())
            self.individualbar.append(self.canvas.create_rectangle(15+i*30 - 5, 220, 15+i*30 + 5, 20, fill="blue"))
        for i in range(len(self.individualcorrect)):
            tkinter.Label(self.window, textvariable=self.individualacc[i],font=font).place(x=645 + 30 * i, y=300)

        self.button = tkinter.Button(self.window, overrelief="solid", width=15, command=self.run, repeatdelay=1000,
                                repeatinterval=100, text="RUN")


        self.draw_background()
        self.attatch()

    def attatch(self):
        self.networklist.insert(0, "Single Layer")
        self.networklist.insert(1, "Multi Layer")
        self.networklist.selection_set(first=0)
        self.networklist.place(x=160, y=40)
        self.cmemfield.place(x=200, y=100)
        self.vthfield.place(x=200, y=130)
        self.spikewidthfield.place(x=200, y=160)
        self.singleinputtimefield.place(x=200, y=190)
        self.spikeenergyfield.place(x=200, y=220)
        self.conductancefilefield.place(x=200, y=250)
        self.cmemfield.insert(0, '100p')
        self.vthfield.insert(0, '0.5')
        self.spikewidthfield.insert(0, '100n')
        self.singleinputtimefield.insert(0, '10u')
        self.spikeenergyfield.insert(0, '10p')
        self.conductancefilefield.insert(0, "None")
        self.canvas.place(x=640, y=40)
        self.button.place(x=80, y=280)
        self.poweraveragelabel.place(x=740, y=560)
        self.energytotallabel.place(x=740, y=520)
        self.energyneuronlabel.place(x=740, y=480)
        self.energysynapselabel.place(x=740, y=440)
        self.totalacclabel.place(x=740, y=400)
        self.currentsamplelabel.place(x= 740, y= 360)
        self.window.mainloop()

    def draw_background(self):

        for acc in range(11):
            self.canvas.create_line(0,220-20*acc,340,220-20*acc, fill="black")
        for acc in range(11):
            tkinter.Label(self.window, text = str(acc*10) + " %").place(x=600, y=250 - acc*20)
        for num in range(10):
            tkinter.Label(self.window, text=str(num)).place(x=650+30*num, y=280)

        tkinter.Label(self.window, text="[acc]").place(x=600, y=20)
        tkinter.Label(self.window, text="[number]").place(x=760, y=320)
        tkinter.Label(self.window, text="Network structure: ").place(x=20, y=40)
        tkinter.Label(self.window, text="Membrane capacitance: ").place(x=20, y=100)
        tkinter.Label(self.window, text="Threshold voltage: ").place(x=20, y=130)
        tkinter.Label(self.window, text="Spike width: ").place(x=20, y=160)
        tkinter.Label(self.window, text="Time / single input: ").place(x=20, y=190)
        tkinter.Label(self.window, text="Energy / spike: ").place(x=20, y=220)
        tkinter.Label(self.window, text="Average Power: ").place(x=620, y=560)
        tkinter.Label(self.window, text="Total Energy: ").place(x=620, y=520)
        tkinter.Label(self.window, text="Neuron Energy: ").place(x=620, y=480)
        tkinter.Label(self.window, text="Synapse Energy: ").place(x=620, y=440)
        tkinter.Label(self.window, text="Total Accuracy: ").place(x=620, y=400)
        tkinter.Label(self.window, text="Current Sample: ").place(x=620, y=360)
        tkinter.Label(self.window, text="Conductance File: ").place(x=20, y=250)
        tkinter.Label(self.window, text="F").place(x=300, y=100)
        tkinter.Label(self.window, text="V").place(x=300, y=130)
        tkinter.Label(self.window, text="s").place(x=300, y=160)
        tkinter.Label(self.window, text="s").place(x=300, y=190)
        tkinter.Label(self.window, text="J").place(x=300, y=220)


        #self.canvas.pack()

    def convert_unit(self, input):
        '''
        :param input: string form of number ex) 10u, 400m, 1p
        :return: floating point representation for input
        '''

        multiple = 0
        head = 0

        if 'p' == input[-1]:
            multiple = 1e-12
            head = input[:-1]
        elif 'n' in input[-1]:
            multiple = 1e-9
            head = input[:-1]
        elif 'u' in input[-1]:
            multiple = 1e-6
            head = input[:-1]
        elif 'm' in input[-1]:
            multiple = 1e-3
            head = input[:-1]
        else:
            multiple = 1
            head = input

        try:
            return (float)(head) * multiple
        except:
            return False


    def load_weight(self):
        '''
        :brief: 미리 저장된 weight file을 로딩
        :return:
        '''
        c = self.conductancefilefield.get()
        if ".csv" in c:
            print ("Getting weight from file... ")
        else:
            c = False
            print ("Using default conductance...")

        print ("WW", self.networklist.curselection())
        if self.networklist.curselection()[0] == 0:
            weight = weightloader.WeightloaderSinglelayer("weight.csv", c)
            self.model.minconductance = weight.get_minconductance()
            return weight.get_weight()

        elif self.networklist.curselection()[0] == 1:
            # TODO: open two files and
            weight = weightloader.WeightloaderMultilayer("weight9757_1.csv", "weight9757_2.csv", c)
            self.model.minconductance = weight.get_minconductance()
            return weight.get_weight()

    def update_energy(self, energy , samplenum):
        self.energysynapse = energy[0]
        self.energyneuron = energy[1]
        energytotal = self.energyneuron + self.energysynapse
        time = (samplenum + 1)*self.singleinputtime

        self.energyneuronlabelval.set('%.2E' % Decimal(str(self.energyneuron)) + " J")



        self.energysynapselabelval.set(str('%.2E' % Decimal(str(self.energysynapse))) + " J")
        self.energytotallabelval.set(str('%.2E' % Decimal(str(energytotal))) + " J")
        self.poweraveragelabelval.set(str('%.2E' % Decimal(str(energytotal/time))) + " W")



    def update_accuracy(self, correct, label):
        '''
        :brief: update accuracy value and graph
        :param correct:
        :param label: l
        :return:
        '''
        if correct: self.correct += 1

        self.totalacc = 100*(self.correct / self.samplenumber)
        self.totalacc = round(self.totalacc, 2)
        self.acc.set(str(self.totalacc ) + " %")

        # update individual accuracy and label
        self.individualnumber[(int)(label)] += 1
        if correct: self.individualcorrect[(int)(label)] += 1
        for i in range(len(self.individualnumber)):
            if self.individualnumber[i] != 0:
                temp = 100*(self.individualcorrect[i] / self.individualnumber[i])
                self.individualacc[i].set(str(round(temp, 1)))

        # update histogram
        for i in range(len(self.individualbar)):
            if self.individualnumber[i] != 0:
                self.canvas.coords(self.individualbar[i], 15 + i * 30 - 5, 220, 15 + i * 30 + 5, 220-200*self.individualcorrect[i] / self.individualnumber[i])
        self.window.update_idletasks()

        pass

    def check_input(self):

        print ("Checking inputs...")

        cmem = self.convert_unit(self.cmemfield.get())
        vth = self.convert_unit(self.vthfield.get())
        spikewidth = self.convert_unit(self.spikewidthfield.get())
        singleinputtime = self.convert_unit(self.singleinputtimefield.get())
        spikeenergy = self.convert_unit(self.spikeenergyfield.get())

        # if one of the parameters are not number
        if not cmem:
            messagebox.showinfo(title="Error", message="wrong input: \n Membrand capacitance")
            return False
        elif not vth:
            messagebox.showinfo(title="Error", message="wrong input: \n Threshold voltage")
            return False
        elif not spikewidth:
            messagebox.showinfo(title="Error", message="wrong input: \n Spike width")
            return False
        elif not singleinputtime:
            messagebox.showinfo(title="Error", message="wrong input: \n Time / single input")
            return False
        elif not spikeenergy:
            messagebox.showinfo(title="Error", message="wrong input: \n Energy / spike")
            return False

        self.cmem = cmem
        self.vth = vth
        self.spikeenergy = spikeenergy
        self.spilkewidth = spikewidth
        self.singleinputtime = singleinputtime
        return True

    def run(self):
        if not self.check_input(): return

        print(self.networklist.get(self.networklist.curselection()))
        print ("running..")



        self.individualcorrect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.individualnumber = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.totalacc = 0
        self.samplenumber = 0
        print (self.cmem)
        if self.networklist.get(self.networklist.curselection()) == "Single Layer":
            print("running single-layer")
            # model 만들고
            self.model = snn.IdealSNN(structure = [784, 10], pulsewidth = self.spilkewidth, inputlength = self.singleinputtime, cmem = self.cmem, vt = self.vth)

            # weight 로딩하고
            self.model.set_weight_ideal(self.load_weight())
        else:
            print ("running multi-layer")
            self.model = snn.IdealSNN(structure=[784, 128, 10], pulsewidth=self.spilkewidth, inputlength=self.singleinputtime,
                                 cmem=self.cmem, vt=self.vth)
            self.model.set_weight_ideal(self.load_weight())

        # dataloader 켜고
        data = dataloader.Dataloader(pulsewidth = self.spilkewidth, inputlength = self.singleinputtime)
        self.currentsamplelabel['text'] = "aaaa"
        for i, item in enumerate(data):
            print (i)
            self.samplenumber += 1
            self.num.set(str(self.samplenumber) + " / 10000")

            self.model.reset()
            self.model.load_input(item)
            correct, label = self.model.run()
            self.update_accuracy(correct, label)
            self.update_energy(self.model.get_energy(), i)


        # 샘플 하나씩 로딩하면서 acc 도출.


w = Mainwindow()
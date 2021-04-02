import threading
from tkinter import *
from PIL import Image, ImageTk
        

class ViewerTerminal(threading.Thread):

    def __init__(self, tournament):
        self.lock = threading.Lock()
        self.tournament = tournament

    def run(self):
        self.initDisplay()
        
        while True:
            self.root.update_idletasks()
            self.root.update()

    def initDisplay(self):
        self.root = Tk()
        self.frame = Frame(self.root, width= 256, height= 356)
        self.initGameDisplay()
        self.initGameLogTextBox()
        self.initUserInputTextBox()

    def initGameDisplay(self):
        self.canvas = Canvas(self.root, width= 256, height= 256)
        self.canvas.pack()

    def initGameLogTextBox(self):
        self.textBox = Text(self.frame, height= 100, width= 256)
        self.textBox.config(state= DISABLED)
        self.textBox.pack()

    def initUserInputTextBox(self):
        self.entryBox = Entry(self.frame, height= 10, width= 256)
        self.entryBox.config(state= NORMAL)
        self.frame.bind("<KeyPress>", self.keyPress)
        self.textBox.pack()

    def keyPress(self, event):
        if event.keysym == 'Return':
            self.userInputEvent()

    def userInputEvent(self):
        text = self.entryBox.get()
        self.tournament.executeUserInput(text)
        self.entryBox.delete('1.0', END)

    def updateTextBox(self, text):
        self.lock.acquire()
        self.textBox.config(state= NORMAL)
        self.textBox.insert(CURRENT, text)
        self.textBox.config(state= DISABLED)
        self.lock.release()

    def updateGameDisplay(self, obs):
        self.canvas.delete("all")
        img =  ImageTk.PhotoImage(image=Image.fromarray(obs))
        self.canvas.create_image(256, 256, anchor="n", image=img)

if __name__ == "__main__":
    pass
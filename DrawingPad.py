import tkinter as tk
from PIL import Image, ImageDraw


class DrawingPad:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_save = tk.Button(root, text="Save", command=self.save_image)
        btn_save.pack()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.line([x1, y1, x2, y2], fill='black', width=10)

    def save_image(self):
        self.image = self.image.resize((28, 28))
        self.image.save('digit.png')
        self.root.destroy()

    @staticmethod
    def run():
        root = tk.Tk()
        DrawingPad(root)
        root.mainloop()

import os
from tkinter import messagebox

import joblib

MODEL_PATH = 'mnist_classifier.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load('mnist_classifier.pkl')
    print("[✅] You already have the model, no need to train.")
else:
    print("[🕜] Downloading the dataset.")
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', as_frame=False, parser='pandas')
    X, Y = mnist.data, mnist.target

    print("[✅] Dataset downloaded successfully.")

    X_train = X
    Y_train = Y
    X_train_scaled = X_train

    print("[🕜] Training model.")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_scaled, Y_train)

    print("[✅] Model trained.")

    joblib.dump(model, MODEL_PATH)

import tkinter as tk

class PixelDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Classifier")
        self.grid_size = 28
        self.cell_size = 20
        self.canvas_width = self.grid_size * self.cell_size
        self.canvas_height = self.grid_size * self.cell_size

        self.canvas = tk.Canvas(
            root, width=self.canvas_width, height=self.canvas_height,
            bg='black', highlightthickness=0
        )
        self.canvas.pack()

        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.rects = [[None] * self.grid_size for _ in range(self.grid_size)]
        self.draw_grid()

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)

        btn = tk.Button(root, text="Make Prediction", command=self.get_data)
        btn.pack()

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                fill = 'white' if self.grid[i][j] == 1 else 'black'
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=fill, outline='gray'
                )
                self.rects[i][j] = rect

    def start_draw(self, event):
        self.set_pixel(event.x, event.y)

    def draw(self, event):
        self.set_pixel(event.x, event.y)

    def set_pixel(self, x, y):
        col = x // self.cell_size
        row = y // self.cell_size

        brush_radius = 1

        for i in range(-brush_radius, brush_radius + 1):
            for j in range(-brush_radius, brush_radius + 1):
                r = row + i
                c = col + j
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    distance = (i ** 2 + j ** 2) ** 0.5

                    if distance <= brush_radius:
                        intensity = max(0, 1.0 - distance / (brush_radius + 0.5))
                        gray_value = int(intensity * 255)

                        if gray_value > self.grid[r][c]:
                            self.grid[r][c] = gray_value
                            color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
                            self.canvas.itemconfig(self.rects[r][c], fill=color)

    def zoom(self, event):
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.cell_size = min(50, self.cell_size + 2)
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.cell_size = max(5, self.cell_size - 2)
        else:
            return
        self.canvas.config(
            width=self.grid_size * self.cell_size,
            height=self.grid_size * self.cell_size
        )
        self.draw_grid()

    def clear_screen(self):
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.draw_grid()

    def show_message(self, text):
        messagebox.showinfo("Prediction: ", text)

    def get_data(self):
        data = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                data.append(self.grid[i][j])
        print(data)

        self.clear_screen()
        self.show_message(model.predict([data]))

        return data

root = tk.Tk()
app = PixelDrawer(root)
root.mainloop()

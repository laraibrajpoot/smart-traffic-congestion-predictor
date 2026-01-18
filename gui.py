import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- Load model ----------------
model = joblib.load("traffic_model.pkl")
encoder = joblib.load("junction_model.pkl")

# ---------------- Main Window ----------------
root = tk.Tk()
root.title("Smart Traffic Congestion Predictor")
root.geometry("600x620")
root.configure(bg="#f4f6f8")

# ---------------- Header ----------------
header = tk.Label(
    root,
    text="🚦 Smart Traffic Congestion Predictor",
    font=("Segoe UI", 18, "bold"),
    bg="#1f2933",
    fg="white",
    pady=15
)
header.pack(fill="x")

# ---------------- Input Card ----------------
card = tk.Frame(root, bg="white", padx=20, pady=20)
card.pack(pady=20)

ttk.Style().configure("TLabel", font=("Segoe UI", 11))
ttk.Style().configure("TEntry", padding=6)

ttk.Label(card, text="Junction Name:").grid(row=0, column=0, sticky="w", pady=8)
junction_entry = ttk.Entry(card, width=25)
junction_entry.grid(row=0, column=1, pady=8)

ttk.Label(card, text="Hour (0–23):").grid(row=1, column=0, sticky="w", pady=8)
hour_entry = ttk.Entry(card, width=25)
hour_entry.grid(row=1, column=1, pady=8)

ttk.Label(card, text="Day (1–31):").grid(row=2, column=0, sticky="w", pady=8)
day_entry = ttk.Entry(card, width=25)
day_entry.grid(row=2, column=1, pady=8)

# ---------------- Result Label ----------------
result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Segoe UI", 14, "bold"),
    bg="#f4f6f8",
    fg="#111827"
)
result_label.pack(pady=10)

# ---------------- Chart Area ----------------
fig, ax = plt.subplots(figsize=(4.5, 3))
fig.patch.set_facecolor("#f4f6f8")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

# ---------------- Prediction Function ----------------
def predict_traffic():
    try:
        junction = junction_entry.get()
        hour = int(hour_entry.get())
        day = int(day_entry.get())

        if not junction:
            raise ValueError("Junction cannot be empty")

        junction_encoded = encoder.transform([junction])[0]
        prediction = model.predict([[hour, day, junction_encoded]])
        vehicles = int(prediction[0])

        if vehicles < 50:
            level = "LOW"
            color = "#22c55e"
        elif vehicles < 120:
            level = "MEDIUM"
            color = "#f59e0b"
        else:
            level = "HIGH"
            color = "#ef4444"

        result_label.config(
            text=f"🚗 Vehicles: {vehicles}   |   Congestion: {level}",
            fg=color
        )

        ax.clear()
        ax.bar(["Predicted Vehicles"], [vehicles], color=color, width=0.4)
        ax.set_ylim(0, 200)
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Traffic Congestion Level")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        canvas.draw()

    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# ---------------- Predict Button ----------------
predict_btn = tk.Button(
    root,
    text="🔍 Predict Traffic",
    command=predict_traffic,
    font=("Segoe UI", 13, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    padx=20,
    pady=10,
    bd=0,
    cursor="hand2"
)
predict_btn.pack(pady=15)

# ---------------- Footer ----------------
footer = tk.Label(
    root,
    text="AI-Based Traffic Prediction System | FYP Project",
    bg="#f4f6f8",
    fg="#6b7280",
    font=("Segoe UI", 9)
)
footer.pack(pady=5)

# ---------------- Run App ----------------
root.mainloop()

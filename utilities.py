import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.2, 1.2)

bars = []  # Placeholder for bars
line = None  # Placeholder for line plot

def update(frame):
    global bars, line  # Ensure we modify the global objects

    # Clear previous bars
    for bar in bars:
        bar.remove()
    bars = []  # Reset the list

    # Number of bars dynamically changing
    num_bars = 5 + int(5 * np.sin(frame * 0.1))  # Between 5 and 10 bars
    x_bars = np.linspace(0, 2*np.pi, num_bars)
    y_bars = np.abs(np.sin(x_bars + frame * 0.1))  # Dynamic heights

    # Create new bars with updated count
    bars = ax.bar(x_bars, y_bars, width=0.3, color='green', alpha=0.6)

    # Create or update line plot
    x_line = np.linspace(0, 2*np.pi, 100)
    y_line = np.sin(x_line + frame * 0.1)
    
    # If the line doesn't exist, create it
    global line
    if line is None:
        line, = ax.plot(x_line, y_line, 'r-', lw=2, label="Line Plot (Sine)")
    else:
        line.set_data(x_line, y_line)  # Update existing line

    return [line, *bars]  # Return all updated elements

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=100, blit=False)

plt.legend()
plt.show()

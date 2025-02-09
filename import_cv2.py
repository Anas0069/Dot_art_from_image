from PIL import Image, ImageDraw
import numpy as np

# Load the full image
image_path = r"D:\WhatsApp Image 2025-02-09 at 14.41.56_94a2a2f3.jpg" # Update with your image path
try:
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    print("Image loaded successfully!")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Resize image for better accuracy
scale_factor = 1.2  # Increase resolution for better details
new_width, new_height = int(image.width * scale_factor), int(image.height * scale_factor)
image = image.resize((new_width, new_height))

# Convert image to numpy array
image_array = np.array(image)

# Debugging: Check intensity values
print(f"Intensity range: {image_array.min()} to {image_array.max()}")

# Create a white background canvas
output_image = Image.new("RGB", (new_width, new_height), "white")
draw = ImageDraw.Draw(output_image)

# Parameters for dot placement
dot_spacing = 3  # Smaller spacing for more detail
min_dot_size = 1  # Minimum dot size
max_dot_size = 5  # Maximum dot size

# Compute intensity-based dot sizes dynamically
for y in range(0, new_height, dot_spacing):
    for x in range(0, new_width, dot_spacing):
        intensity = image_array[y, x]
        
        # Invert intensity (0 = white, 255 = black)
        inverted_intensity = 255 - intensity

        # Normalize intensity for dot size (brighter areas get larger dots)
        dot_size = int(np.interp(inverted_intensity, [0, 255], [min_dot_size, max_dot_size]))

        # Dot color: Darker areas get black dots, lighter areas get gray dots
        gray_value = int(np.interp(inverted_intensity, [0, 255], [0, 150]))  # Range: 0 (black) to 150 (gray)
        dot_color = (gray_value, gray_value, gray_value)

        # Draw the dot
        draw.ellipse(
            [(x - dot_size, y - dot_size), (x + dot_size, y + dot_size)],
            fill=dot_color,
            outline=dot_color,
        )

# Save or display the result
output_image.save("creation_of_adam_white_background.png")
print("Dot art saved as 'creation_of_adam_white_background.png'")
output_image.show()

from PIL import Image, ImageDraw
import numpy as np

def image_to_dot_art(image_path, output_path, scale_factor=0.5, dot_spacing=4, bold_dot_size=5, light_dot_size=3):
    """
    Converts an image into high-definition dot art with a black background.
    
    Parameters:
        - image_path: Path to the input image.
        - output_path: Path to save the output image.
        - scale_factor: Resizes the image to process faster.
        - dot_spacing: Space between dots.
        - bold_dot_size: Size of bright highlight dots.
        - light_dot_size: Size of shadow dots.
    """

    # Load the image and convert to grayscale
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Resize the image
    new_width, new_height = int(image.width * scale_factor), int(image.height * scale_factor)
    image = image.resize((new_width, new_height))

    # Convert image to numpy array
    image_array = np.array(image)

    # Create a blank canvas with black background
    output_image = Image.new("RGB", (new_width, new_height), "black")
    draw = ImageDraw.Draw(output_image)

    # Iterate over the image to place dots
    for y in range(0, new_height, dot_spacing):
        for x in range(0, new_width, dot_spacing):
            intensity = image_array[y, x]

            # Define dot size and color based on intensity values
            if intensity > 200:  # Very bright highlights
                dot_size = bold_dot_size + 1  # Slightly larger for bright areas
                dot_color = (255, 255, 255)  # White
            elif intensity > 160:  # Normal highlights
                dot_size = bold_dot_size
                dot_color = (230, 230, 230)  # Light gray-white
            elif intensity > 100:  # Shadowed areas
                dot_size = light_dot_size
                dot_color = (120, 120, 120)  # Gray
            else:
                continue  # Background remains black

            # Draw the dot
            draw.ellipse(
                [(x - dot_size, y - dot_size), (x + dot_size, y + dot_size)],
                fill=dot_color,
                outline=dot_color,
            )

    # Save the final dot art image
    output_image.save(output_path)
    print(f"Dot art saved as {output_path}")

# Example Usage
image_path = "input.jpg"  # Change this to your image path
output_path = "dot_art_output.png"
image_to_dot_art(image_path, output_path)

🎨 Advanced Dot Art Generator
Transform any image into stunning dot art using advanced computer vision and machine learning techniques. This generator analyzes image features like edges, corners, textures, and gradients to intelligently place dots that recreate the visual structure of your original image.
✨ Features

🔍 Advanced Feature Detection: Multi-scale edge detection, corner detection, texture analysis, and gradient computation
🤖 Machine Learning Integration: Uses DBSCAN clustering and K-means for optimal dot placement
🎯 Intelligent Dot Sizing: Variable dot sizes based on local image features and complexity
🖼️ Multiple Art Styles: Classic, inverse, modern, and sketch styles
⚡ Quality Settings: Fast, balanced, and high-quality processing options
🔄 Batch Processing: Process multiple images at once
🎨 Anti-aliasing: 2x supersampling for smooth, professional results
📊 Visual Analysis: Shows feature importance maps alongside results

🖼️ Sample Results
StyleDescriptionClassicBlack background with white dotsInverseWhite background with black dotsModernDark blue background with light blue dotsSketchLight background with dark dots
🚀 Quick Start
Installation
bashpip install opencv-python numpy matplotlib scikit-learn scipy
Basic Usage
pythonfrom dot_art_generator import create_dot_art

# Generate dot art with one line!
result = create_dot_art(
    image_path="input_image.jpg",
    output_path="dot_art_output.png",
    style='classic',
    quality='high'
)
Command Line Usage
bashpython dot_art_generator.py
📖 Detailed Usage
Single Image Processing
pythonfrom dot_art_generator import AccurateDotArtGenerator

# Create custom generator
generator = AccurateDotArtGenerator(
    output_width=1400,           # Output image width
    output_height=1000,          # Output image height
    min_dot_size=1,             # Minimum dot size
    max_dot_size=8,             # Maximum dot size
    dot_density=0.8,            # Dot density (0.1-1.0)
    edge_sensitivity=0.4,       # Edge detection sensitivity
    detail_preservation=0.8     # Detail preservation level
)

# Generate dot art
result = generator.generate_dot_art(
    "input.jpg", 
    "output.png", 
    style='classic'
)
Batch Processing
pythonfrom dot_art_generator import batch_create_dot_art

# Process all images in a folder
batch_create_dot_art(
    input_folder="input_images/",
    output_folder="output_images/",
    style='classic',
    quality='high'
)
🎨 Style Options
Classic Style

Background: Black
Dots: White
Best for: High contrast, dramatic effect

Inverse Style

Background: White
Dots: Black
Best for: Clean, minimalist look

Modern Style

Background: Dark blue
Dots: Light blue
Best for: Contemporary, digital aesthetic

Sketch Style

Background: Off-white
Dots: Dark gray
Best for: Hand-drawn, artistic feel

🔧 Advanced Configuration
Custom Parameters
pythongenerator = AccurateDotArtGenerator(
    output_width=2000,           # Higher resolution
    output_height=1500,
    min_dot_size=1,             # Tiny dots for detail
    max_dot_size=12,            # Larger dots for emphasis
    dot_density=0.9,            # Maximum dots
    edge_sensitivity=0.5,       # High edge detection
    detail_preservation=0.9     # Maximum detail
)
Feature Weights
The algorithm combines multiple image features:

Edges (30%): Canny edge detection at multiple scales
Corners (20%): Harris corner detection
Gradients (20%): Sobel gradient magnitude
Intensity Variance (15%): Local intensity variations
Texture (10%): Local Binary Pattern analysis
Morphology (5%): Morphological gradient

🔍 How It Works

Image Preprocessing: Loads and resizes image while maintaining aspect ratio
Feature Extraction: Analyzes edges, corners, textures, gradients, and morphological features
Importance Mapping: Combines features into a single importance map
Adaptive Dot Placement: Places dots based on local image complexity
Position Refinement: Uses DBSCAN clustering to optimize dot positions
Advanced Rendering: Renders with 2x supersampling and anti-aliasing

🛠️ Requirements

Python 3.7+
OpenCV 4.0+
NumPy
Matplotlib
Scikit-learn
SciPy

📦 Installation
Via pip (recommended)
bashpip install -r requirements.txt
Manual installation
bashpip install opencv-python>=4.0.0
pip install numpy>=1.19.0
pip install matplotlib>=3.3.0
pip install scikit-learn>=0.24.0
pip install scipy>=1.6.0
🎯 Supported Formats
Input formats:

JPEG (.jpg, .jpeg)
PNG (.png)
BMP (.bmp)
TIFF (.tiff)
WebP (.webp)

Output format:

PNG (recommended for best quality)

🚨 Troubleshooting
Common Issues
Error: "Cannot load image"

Check if the file path is correct
Ensure the image file exists and is not corrupted
Try a different image format

Low quality results

Increase the quality parameter to 'high'
Adjust dot_density for more dots
Try different edge_sensitivity values

Processing too slow

Use 'fast' quality setting for quick results
Reduce output resolution
Lower dot_density parameter

Performance Tips

Use smaller input images for faster processing
'Fast' quality is good for testing parameters
'High' quality for final artwork
Batch processing is more efficient for multiple images

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

OpenCV community for computer vision tools
Scikit-learn for machine learning algorithms
NumPy and SciPy for numerical computing
Matplotlib for visualization

📞 Support
If you encounter any issues or have questions:

Open an issue on GitHub
Check the troubleshooting section above
Review the examples in the /examples directory
Made with ❤️ by [Your Name]
Transform your images into beautiful dot art with the power of computer vision and machine learning!

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy import ndimage
import random
from typing import Tuple, List, Optional
import os
import math

class AccurateDotArtGenerator:
    def __init__(self, 
                 output_width: int = 1200,
                 output_height: int = 800,
                 min_dot_size: int = 1,
                 max_dot_size: int = 8,
                 dot_density: float = 0.6,
                 edge_sensitivity: float = 0.3,
                 detail_preservation: float = 0.7):
        """
        Advanced Dot Art Generator with accurate feature detection
        
        Args:
            output_width: Width of output image
            output_height: Height of output image
            min_dot_size: Minimum dot size
            max_dot_size: Maximum dot size
            dot_density: Overall dot density (0.1-1.0)
            edge_sensitivity: How sensitive to edges (0.1-1.0)
            detail_preservation: How much detail to preserve (0.1-1.0)
        """
        self.output_width = output_width
        self.output_height = output_height
        self.min_dot_size = min_dot_size
        self.max_dot_size = max_dot_size
        self.dot_density = dot_density
        self.edge_sensitivity = edge_sensitivity
        self.detail_preservation = detail_preservation
        
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced preprocessing with multiple analysis layers
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Resize maintaining aspect ratio
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > self.output_width / self.output_height:
            new_w = self.output_width
            new_h = int(self.output_width / aspect_ratio)
        else:
            new_h = self.output_height
            new_w = int(self.output_height * aspect_ratio)
            
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        return image, gray
    
    def extract_comprehensive_features(self, gray: np.ndarray) -> dict:
        """
        Extract multiple types of features for accurate dot placement
        """
        features = {}
        
        # 1. Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges_coarse = cv2.Canny(gray, 30, 100, apertureSize=5)
        features['edges'] = np.maximum(edges_fine, edges_coarse)
        
        # 2. Advanced corner detection
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        features['corners'] = corners
        
        # 3. Texture analysis using Local Binary Patterns
        features['texture'] = self._calculate_texture_map(gray)
        
        # 4. Gradient magnitude and direction
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_magnitude'] = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_direction'] = np.arctan2(grad_y, grad_x)
        
        # 5. Intensity variations
        features['intensity'] = gray.astype(np.float32)
        features['intensity_variance'] = ndimage.generic_filter(gray, np.var, size=5)
        
        # 6. Frequency domain features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        features['frequency'] = magnitude_spectrum
        
        # 7. Morphological features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        features['morphology'] = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        return features
    
    def _calculate_texture_map(self, gray: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        Calculate texture using simplified Local Binary Pattern
        """
        texture_map = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                pattern = 0
                
                # Sample 8 neighbors
                neighbors = [
                    gray[i-radius, j-radius], gray[i-radius, j], gray[i-radius, j+radius],
                    gray[i, j+radius], gray[i+radius, j+radius], gray[i+radius, j],
                    gray[i+radius, j-radius], gray[i, j-radius]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern += 2**k
                
                texture_map[i, j] = pattern
        
        return texture_map
    
    def calculate_importance_map(self, features: dict, gray: np.ndarray) -> np.ndarray:
        """
        Calculate pixel importance for dot placement using multiple features
        """
        h, w = gray.shape
        importance = np.zeros((h, w), dtype=np.float32)
        
        # Normalize all features to 0-1 range
        normalized_features = {}
        for key, feature in features.items():
            if feature.dtype in [np.float64, np.float32]:
                min_val, max_val = feature.min(), feature.max()
                if max_val > min_val:
                    normalized_features[key] = (feature - min_val) / (max_val - min_val)
                else:
                    normalized_features[key] = np.zeros_like(feature)
            else:
                normalized_features[key] = feature.astype(np.float32) / 255.0
        
        # Weighted combination of features
        weights = {
            'edges': 0.3,
            'corners': 0.2,
            'gradient_magnitude': 0.2,
            'intensity_variance': 0.15,
            'texture': 0.1,
            'morphology': 0.05
        }
        
        for feature_name, weight in weights.items():
            if feature_name in normalized_features:
                importance += normalized_features[feature_name] * weight
        
        # Apply Gaussian smoothing to avoid noise
        importance = cv2.GaussianBlur(importance, (3, 3), 0.5)
        
        # Enhance important regions
        importance = np.power(importance, 1.0 - self.detail_preservation)
        
        return importance
    
    def generate_adaptive_dots(self, importance_map: np.ndarray, 
                             gray: np.ndarray) -> List[Tuple[int, int, int, float]]:
        """
        Generate dots with adaptive placement and sizing
        """
        h, w = importance_map.shape
        dots = []
        
        # Calculate adaptive grid size based on image complexity
        complexity = np.std(importance_map)
        base_grid_size = max(3, int(15 * (1 - complexity)))
        
        # Use importance-based sampling
        for y in range(0, h, base_grid_size):
            for x in range(0, w, base_grid_size):
                # Sample local region
                y_end = min(y + base_grid_size, h)
                x_end = min(x + base_grid_size, w)
                
                local_importance = importance_map[y:y_end, x:x_end]
                local_gray = gray[y:y_end, x:x_end]
                
                if local_importance.size == 0:
                    continue
                
                # Find the most important point in this region
                local_max_idx = np.unravel_index(np.argmax(local_importance), local_importance.shape)
                global_y = y + local_max_idx[0]
                global_x = x + local_max_idx[1]
                
                importance_score = importance_map[global_y, global_x]
                
                # Threshold for dot placement
                placement_threshold = 0.1 + (1 - self.dot_density) * 0.4
                
                if importance_score > placement_threshold:
                    # Calculate dot size based on local features
                    local_variance = np.var(local_gray)
                    normalized_variance = min(1.0, local_variance / 100.0)
                    
                    size_factor = 0.3 * importance_score + 0.7 * normalized_variance
                    dot_size = int(self.min_dot_size + size_factor * (self.max_dot_size - self.min_dot_size))
                    dot_size = max(self.min_dot_size, min(self.max_dot_size, dot_size))
                    
                    # Calculate dot intensity (for variable opacity)
                    intensity = 1.0 - (gray[global_y, global_x] / 255.0)
                    
                    dots.append((global_x, global_y, dot_size, intensity))
        
        return dots
    
    def refine_dot_positions(self, dots: List[Tuple[int, int, int, float]]) -> List[Tuple[int, int, int, float]]:
        """
        Refine dot positions to avoid overlap and improve distribution
        """
        if len(dots) < 2:
            return dots
        
        positions = np.array([[dot[0], dot[1]] for dot in dots])
        
        # Use DBSCAN to find clusters of nearby dots
        clustering = DBSCAN(eps=8, min_samples=2)
        clusters = clustering.fit_predict(positions)
        
        refined_dots = []
        
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise points (isolated dots)
                cluster_mask = clusters == cluster_id
                cluster_dots = [dots[i] for i in np.where(cluster_mask)[0]]
                refined_dots.extend(cluster_dots)
            else:  # Clustered dots - merge them
                cluster_mask = clusters == cluster_id
                cluster_dots = [dots[i] for i in np.where(cluster_mask)[0]]
                
                if len(cluster_dots) > 1:
                    # Merge cluster into single representative dot
                    avg_x = np.mean([dot[0] for dot in cluster_dots])
                    avg_y = np.mean([dot[1] for dot in cluster_dots])
                    max_size = max([dot[2] for dot in cluster_dots])
                    avg_intensity = np.mean([dot[3] for dot in cluster_dots])
                    
                    refined_dots.append((int(avg_x), int(avg_y), max_size, avg_intensity))
                else:
                    refined_dots.extend(cluster_dots)
        
        return refined_dots
    
    def render_advanced_dots(self, dots: List[Tuple[int, int, int, float]], 
                           canvas_shape: Tuple[int, int],
                           style: str = 'classic') -> np.ndarray:
        """
        Render dots with advanced styling and anti-aliasing
        """
        h, w = canvas_shape
        
        # Style configurations
        if style == 'classic':
            bg_color = (0, 0, 0)  # Black
            dot_color = (255, 255, 255)  # White
        elif style == 'inverse':
            bg_color = (255, 255, 255)  # White
            dot_color = (0, 0, 0)  # Black
        elif style == 'modern':
            bg_color = (15, 15, 30)  # Dark blue
            dot_color = (200, 220, 255)  # Light blue
        else:  # sketch
            bg_color = (245, 245, 240)  # Off-white
            dot_color = (40, 40, 50)  # Dark gray
        
        # Create high-resolution canvas for anti-aliasing
        scale_factor = 2
        hr_canvas = np.full((h * scale_factor, w * scale_factor, 3), bg_color, dtype=np.uint8)
        
        for x, y, size, intensity in dots:
            # Scale up coordinates and size
            hr_x, hr_y = x * scale_factor, y * scale_factor
            hr_size = size * scale_factor
            
            # Adjust color based on intensity
            if style in ['classic', 'modern']:
                current_color = tuple(int(c * intensity) for c in dot_color)
            else:
                # For inverse and sketch styles, use intensity differently
                blend_factor = intensity
                current_color = tuple(
                    int(bg_color[i] * (1 - blend_factor) + dot_color[i] * blend_factor)
                    for i in range(3)
                )
            
            # Draw dot with slight randomness for organic feel
            dot_type = random.choice(['circle', 'circle', 'circle', 'square'])  # Prefer circles
            
            if dot_type == 'circle':
                cv2.circle(hr_canvas, (hr_x, hr_y), hr_size, current_color, -1, cv2.LINE_AA)
            else:  # square
                half_size = hr_size
                cv2.rectangle(hr_canvas, 
                            (hr_x - half_size, hr_y - half_size),
                            (hr_x + half_size, hr_y + half_size),
                            current_color, -1, cv2.LINE_AA)
        
        # Downscale with anti-aliasing
        canvas = cv2.resize(hr_canvas, (w, h), interpolation=cv2.INTER_AREA)
        
        return canvas
    
    def generate_dot_art(self, image_path: str, 
                        output_path: Optional[str] = None,
                        style: str = 'classic',
                        show_preview: bool = True) -> np.ndarray:
        """
        Main method to generate accurate dot art
        """
        print(f"Processing image: {image_path}")
        
        # Load and preprocess
        image, gray = self.preprocess_image(image_path)
        print("✓ Image preprocessed")
        
        # Extract comprehensive features
        features = self.extract_comprehensive_features(gray)
        print("✓ Features extracted")
        
        # Calculate importance map
        importance_map = self.calculate_importance_map(features, gray)
        print("✓ Importance map calculated")
        
        # Generate adaptive dots
        dots = self.generate_adaptive_dots(importance_map, gray)
        print(f"✓ Generated {len(dots)} dots")
        
        # Refine positions
        refined_dots = self.refine_dot_positions(dots)
        print(f"✓ Refined to {len(refined_dots)} dots")
        
        # Render final artwork
        canvas_shape = (self.output_height, self.output_width)
        if image.shape[0] < self.output_height:
            canvas_shape = (image.shape[0], image.shape[1])
            
        dot_art = self.render_advanced_dots(refined_dots, canvas_shape, style)
        print("✓ Dot art rendered")
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, dot_art)
            print(f"✓ Saved to: {output_path}")
        
        # Show preview
        if show_preview:
            self._show_comparison(image, dot_art, importance_map)
        
        return dot_art
    
    def _show_comparison(self, original: np.ndarray, dot_art: np.ndarray, importance_map: np.ndarray):
        """
        Show comparison between original and generated art
        """
        plt.figure(figsize=(18, 6))
        
        # Original
        plt.subplot(1, 3, 1)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        plt.imshow(original_rgb)
        plt.title('Original Image', fontsize=14)
        plt.axis('off')
        
        # Importance map
        plt.subplot(1, 3, 2)
        plt.imshow(importance_map, cmap='hot')
        plt.title('Feature Importance Map', fontsize=14)
        plt.axis('off')
        
        # Dot art
        plt.subplot(1, 3, 3)
        dot_art_rgb = cv2.cvtColor(dot_art, cv2.COLOR_BGR2RGB)
        plt.imshow(dot_art_rgb)
        plt.title('Generated Dot Art', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Easy-to-use wrapper functions
def create_dot_art(image_path: str, 
                  output_path: str = "dot_art_output.png",
                  style: str = 'classic',
                  quality: str = 'high') -> np.ndarray:
    """
    Simple function to create dot art with predefined quality settings
    
    Args:
        image_path: Path to input image
        output_path: Path for output image
        style: 'classic', 'inverse', 'modern', or 'sketch'
        quality: 'fast', 'balanced', or 'high'
    """
    
    if quality == 'fast':
        generator = AccurateDotArtGenerator(
            output_width=800, output_height=600,
            min_dot_size=2, max_dot_size=6,
            dot_density=0.4, edge_sensitivity=0.2, detail_preservation=0.5
        )
    elif quality == 'balanced':
        generator = AccurateDotArtGenerator(
            output_width=1000, output_height=750,
            min_dot_size=1, max_dot_size=7,
            dot_density=0.6, edge_sensitivity=0.3, detail_preservation=0.7
        )
    else:  # high quality
        generator = AccurateDotArtGenerator(
            output_width=1400, output_height=1000,
            min_dot_size=1, max_dot_size=8,
            dot_density=0.8, edge_sensitivity=0.4, detail_preservation=0.8
        )
    
    return generator.generate_dot_art(image_path, output_path, style)

# Batch processing function
def batch_create_dot_art(input_folder: str, 
                        output_folder: str,
                        style: str = 'classic',
                        quality: str = 'balanced'):
    """
    Process multiple images in batch
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"dot_art_{os.path.splitext(filename)[0]}.png"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                print(f"\n--- Processing {filename} ---")
                create_dot_art(input_path, output_path, style, quality)
                print(f"✅ Completed: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    try:
        input_image = "D:\\Projects\\dot_art\\input_image.jpg"  
        output_image = "D:\\Projects\\dot_art\\accurate_dot_art.png"
        
        print("=== Accurate Dot Art Generator ===")
        print(f"Input: {input_image}")
        print(f"Output: {output_image}")
        print()
        
        result = create_dot_art(
            image_path=input_image,
            output_path=output_image,
            style='classic',
            quality='high'
        )
        
        print("\n🎨 Dot art generation completed successfully!")
        print(f"📁 Output saved to: {output_image}")
        
    except FileNotFoundError:
        print("❌ Error: Input image not found!")
        print("Please make sure your image file exists and update the 'input_image' path.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
    batch_create_dot_art("input_images/", "output_images/", style='classic', quality='high')
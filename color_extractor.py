# color_extractor.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
from collections import Counter
import colorsys

class AdvancedColorExtractor:
    def __init__(self, n_colors=5, color_tolerance=30):
        self.n_colors = n_colors
        self.color_tolerance = color_tolerance
        
        # Extended color database for better matching
        self.color_database = self._build_color_database()
    
    def _build_color_database(self):
        """Build comprehensive color database"""
        colors = {}
        
        # Add CSS3 colors - Updated for webcolors 24.8.0+
        try:
            # For webcolors 24.8.0+
            css3_colors = webcolors.names("css3")
            for name in css3_colors:
                hex_color = webcolors.name_to_hex(name, spec="css3")
                rgb = webcolors.hex_to_rgb(hex_color)
                colors[name] = rgb
        except (AttributeError, TypeError):
            # Fallback for older versions
            try:
                for name, hex_color in webcolors.CSS3_HEX_TO_NAMES.items():
                    rgb = webcolors.hex_to_rgb(hex_color)
                    colors[name] = rgb
            except AttributeError:
                # Manual fallback with basic colors
                basic_colors = {
                    'red': (255, 0, 0),
                    'green': (0, 255, 0),
                    'blue': (0, 0, 255),
                    'white': (255, 255, 255),
                    'black': (0, 0, 0),
                    'yellow': (255, 255, 0),
                    'cyan': (0, 255, 255),
                    'magenta': (255, 0, 255),
                    'gray': (128, 128, 128),
                    'orange': (255, 165, 0),
                    'purple': (128, 0, 128),
                    'brown': (165, 42, 42),
                    'pink': (255, 192, 203),
                    'navy': (0, 0, 128),
                    'maroon': (128, 0, 0)
                }
                colors.update(basic_colors)
        
        # Add fashion-specific colors
        fashion_colors = {
            'navy': (0, 0, 128),
            'burgundy': (128, 0, 32),
            'maroon': (128, 0, 0),
            'beige': (245, 245, 220),
            'cream': (255, 253, 208),
            'ivory': (255, 255, 240),
            'khaki': (240, 230, 140),
            'denim': (21, 96, 189),
            'charcoal': (54, 69, 79),
            'rose_gold': (183, 110, 121),
            'mint': (189, 252, 201),
            'coral': (255, 127, 80),
            'salmon': (250, 128, 114),
            'peach': (255, 218, 185),
            'lavender': (230, 230, 250),
            'sage': (158, 169, 138),
            'mustard': (255, 219, 88),
            'rust': (183, 65, 14),
            'emerald': (80, 200, 120),
            'sapphire': (15, 82, 186)
        }
        
        colors.update(fashion_colors)
        return colors
    
    def extract_colors_from_mask(self, image, mask, min_pixels=100):
        """Extract dominant colors from masked region with advanced filtering"""
        
        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Apply mask to image
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) < min_pixels:
            return [], [], []
        
        # Remove very dark and very bright pixels (likely shadows/highlights)
        pixel_brightness = np.mean(masked_pixels, axis=1)
        valid_pixels = masked_pixels[
            (pixel_brightness > 30) & (pixel_brightness < 225)
        ]
        
        if len(valid_pixels) < min_pixels:
            valid_pixels = masked_pixels
        
        # Convert to LAB color space for better clustering
        lab_pixels = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
        lab_pixels = lab_pixels.reshape(-1, 3)
        
        # Determine optimal number of clusters
        n_clusters = min(self.n_colors, len(valid_pixels) // 50, 8)
        n_clusters = max(1, n_clusters)
        
        # K-means clustering in LAB space
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lab_pixels)
        
        # Convert cluster centers back to RGB
        lab_centers = kmeans.cluster_centers_
        rgb_centers = cv2.cvtColor(
            lab_centers.reshape(-1, 1, 3).astype(np.uint8), 
            cv2.COLOR_LAB2RGB
        ).reshape(-1, 3)
        
        # Calculate color percentages
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        colors = []
        percentages = []
        color_names = []
        
        for i, color in enumerate(rgb_centers):
            percentage = label_counts[i] / total_pixels
            if percentage > 0.05:  # Only include colors that make up >5% of the region
                colors.append(color)
                percentages.append(percentage)
                color_names.append(self.get_color_name(color))
        
        # Sort by percentage
        if colors:
            sorted_indices = np.argsort(percentages)[::-1]
            colors = [colors[i] for i in sorted_indices]
            percentages = [percentages[i] for i in sorted_indices]
            color_names = [color_names[i] for i in sorted_indices]
        
        return colors, percentages, color_names
    
    def get_color_name(self, rgb_color):
        """Get the closest color name using advanced color matching"""
        rgb_color = tuple(int(c) for c in rgb_color)
        
        # Try exact match first
        try:
            return webcolors.rgb_to_name(rgb_color)
        except ValueError:
            pass
        
        # Find closest color using Euclidean distance in RGB space
        min_distance = float('inf')
        closest_name = 'unknown'
        
        for name, color in self.color_database.items():
            distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_color, color)))
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        # If distance is too large, try HSV-based matching
        if min_distance > self.color_tolerance:
            closest_name = self._get_hsv_color_name(rgb_color)
        
        return closest_name.replace('_', ' ').title()
    
    def _get_hsv_color_name(self, rgb_color):
        """Get color name based on HSV values"""
        r, g, b = [c/255.0 for c in rgb_color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h = h * 360  # Convert to degrees
        
        # Define color ranges in HSV
        if v < 0.2:
            return 'black'
        elif v > 0.9 and s < 0.1:
            return 'white'
        elif s < 0.1:
            if v < 0.3:
                return 'dark_gray'
            elif v < 0.7:
                return 'gray'
            else:
                return 'light_gray'
        else:
            # Color hue ranges
            if h < 15 or h >= 345:
                return 'red'
            elif h < 45:
                return 'orange'
            elif h < 75:
                return 'yellow'
            elif h < 150:
                return 'green'
            elif h < 210:
                return 'blue'
            elif h < 270:
                return 'purple'
            elif h < 330:
                return 'pink'
            else:
                return 'red'
    
    def get_color_info(self, colors, percentages, color_names):
        """Get detailed color information"""
        color_info = []
        
        for color, percentage, name in zip(colors, percentages, color_names):
            color_info.append({
                'rgb': tuple(int(c) for c in color),
                'hex': '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2])),
                'name': name,
                'percentage': float(percentage)
            })
        
        return color_info

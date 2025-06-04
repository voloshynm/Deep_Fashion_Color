# AI Color Stylist: Deep Learning-Based Fashion Color Harmony Analysis

A state-of-the-art deep learning pipeline that automatically detects clothing items, extracts dominant colors, and provides intelligent color harmony recommendations based on established fashion theory principles.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
- [Model Architecture](#model-architecture)
- [Color Theory Implementation](#color-theory-implementation)
- [Technical Implementation](#technical-implementation)
- [Dataset Information](#dataset-information)
- [Results and Validation](#results-and-validation)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

This project presents a comprehensive deep learning pipeline for automated fashion color harmony analysis, combining computer vision and color theory to democratize fashion expertise. Our system integrates YOLOv8 segmentation for clothing detection, K-means clustering for color extraction, and rule-based harmony analysis to provide real-time fashion guidance.

**Research Paper**: *AI Color Stylist: Deep Learning-Based Fashion Color Harmony Analysis* (Introduction to Deep Learning, May 2025)

The system addresses three core challenges:
1. **Accurate Detection**: Precise clothing item detection and segmentation across diverse imagery
2. **Color Extraction**: Robust extraction of meaningful color palettes from segmented clothing items  
3. **Harmony Assessment**: Principled color harmony evaluation based on established fashion theory

---

## Key Features

- **🎯 Advanced Object Detection**: YOLOv8-medium model achieving 79.5% mAP50 for detection and 64.7% mAP50 for segmentation
- **🎨 Intelligent Color Analysis**: K-means clustering in LAB color space with 88% accuracy in dominant color identification
- **📊 Realistic Harmony Scoring**: Produces realistic assessments (20-70% range) avoiding grade inflation common in other systems
- **⚡ Real-time Performance**: 250ms inference time per image on consumer hardware
- **🔧 Modular Architecture**: Easily extensible for texture, style, or occasion-based recommendations
- **📱 Actionable Recommendations**: Specific suggestions for color improvements with fashion-theory backing

---

## Performance Metrics

| Metric | Detection (Box) | Segmentation (Mask) |
|--------|----------------|-------------------|
| **Precision** | 78.4% | 71.9% |
| **Recall** | 73.9% | 66.1% |
| **mAP50** | 79.5% | 66.8% |
| **mAP50-95** | 64.7% | 45.5% |

**Processing Speed**: 250ms per image (detection + segmentation + color analysis)  
**Color Extraction Accuracy**: 88% in LAB color space  
**Expert Agreement**: 73% agreement on harmony classifications, 81% on recommendations

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: RTX 3080 or better)
- 8GB+ VRAM for optimal performance

### Setup Instructions

1. **Clone the repository**:
```
git clone https://github.com/voloshynm/Deep_Fashion_Color.git
cd Deep_Fashion_Color
```

2. **Create and activate virtual environment**:
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:
```
pip install -r requirements.txt
```

4. **Download pre-trained weights**:
```
# YOLOv8 weights will be downloaded automatically on first run
# Or manually download from: https://github.com/ultralytics/ultralytics
```

5. **Verify installation**:
```
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print('YOLOv8 ready')"
```

---

## Step-by-Step Usage Guide

### **Method 1: Command Line Interface**

1. **Prepare your image**:
   - Ensure the image contains clearly visible clothing items
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Recommended resolution: 256x256 to 2048x2048 pixels

2. **Run the analysis**:
```
python run_analysis.py --image path/to/your/outfit.jpg --output results/
```

3. **View results**:
   - **Detection visualization**: `results/detected_items.jpg`
   - **Color analysis**: `results/color_palette.png`
   - **Harmony report**: `results/harmony_analysis.json`
   - **Recommendations**: `results/recommendations.txt`

### **Method 2: Python Script**

1. **Create analysis script**:
```
from src.color_stylist import ColorStylist
from src.utils import load_image, save_results

# Initialize the stylist
stylist = ColorStylist(model_path='models/yolov8m-seg.pt')

# Load and analyze image
image = load_image('path/to/outfit.jpg')
results = stylist.analyze_outfit(image)

# Extract results
detected_items = results['detected_items']
color_harmony = results['harmony_score']
recommendations = results['recommendations']

print(f"Harmony Score: {color_harmony:.1f}%")
print(f"Recommendations: {recommendations}")
```

2. **Run your script**:
```
python your_analysis_script.py
```

### **Method 3: Interactive Jupyter Notebook**

1. **Launch notebook**:
```
jupyter notebook examples/interactive_analysis.ipynb
```

2. **Follow the step-by-step cells**:
   - Upload image
   - Run detection
   - Analyze colors
   - View recommendations

### **Understanding the Output**

**Detection Results**:
- Bounding boxes around detected clothing items
- Segmentation masks for precise color extraction
- Confidence scores for each detection
- Item categories (shirt, trousers, dress, etc.)

**Color Analysis**:
- Dominant colors for each clothing item (up to 5 colors per item)
- Color names using fashion-specific terminology
- Color temperature classification (warm/cool/neutral)
- Percentage coverage of each color

**Harmony Assessment**:
- Overall harmony score (0-100%)
- Detailed breakdown by color theory principles
- Identification of harmony issues (monotone, clashing, etc.)
- Specific recommendations for improvement

**Sample Output**:
```
=== AI Color Stylist Analysis ===
Detected Items: 3 (shirt, trousers, shoes)
Overall Harmony Score: 42.3%

Issues Identified:
- Monotone palette (too many neutrals)
- Lack of focal point color
- Insufficient contrast

Recommendations:
- Replace gray shirt with coral or teal for better contrast
- Add accent color through accessories
- Consider navy instead of black trousers for softer look
```

---

## Model Architecture

### **YOLOv8 Segmentation Model**
- **Backbone**: CSPDarknet53 feature extractor
- **Architecture**: Feature Pyramid Network for multi-scale detection
- **Parameters**: 71M parameters (YOLOv8-medium)
- **Specialized heads**: Simultaneous bounding box regression, classification, and mask prediction
- **Activation**: SiLU activation functions throughout
- **Modern techniques**: Attention mechanisms and path aggregation networks

### **Training Configuration**
- **Dataset**: DeepFashion2 (491K images, 801K items, 13 categories)
- **Hardware**: Optimized for RTX 3080 (8GB VRAM)
- **Batch size**: 64, Image resolution: 256×256
- **Optimizer**: SGD with momentum 0.937
- **Learning rate**: 0.01 with step decay
- **Data augmentation**: Mosaic, mixup, color jittering
- **Training time**: 38 epochs over 17 hours

### **Color Extraction Pipeline**
- **Algorithm**: K-means clustering (k=5) in LAB color space
- **Preprocessing**: Shadow/highlight removal, minimum region constraints
- **Color naming**: Extended database of 50+ fashion-specific colors
- **Filtering**: Brightness thresholds (30-225), minimum 100 pixels, >5% coverage

---

## Color Theory Implementation

### **Harmony Analysis Principles**

**Temperature Mixing Assessment**:
- Penalizes excessive mixing of warm and cool colors
- Identifies temperature conflicts that create visual discord

**Monotone/Neutral Detection**:
- Flags outfits composed entirely of neutral colors
- Encourages visual interest through color variety

**Contrast and Saturation Analysis**:
- Ensures sufficient contrast between clothing items
- Discourages all-dark or all-washed-out combinations

**Focal Point Identification**:
- Checks for at least one saturated or bright color
- Ensures visual anchor points in the outfit

### **Recommendation Engine**

The `FixedColorHarmonyAnalyzer` class implements:
- **Color classification**: Warm, cool, neutral categorization using HSV thresholds
- **Rule-based evaluation**: Fashion theory principles application
- **Contextual suggestions**: Curated color recommendations (accent colors, versatile neutrals)
- **Actionable feedback**: Specific item identification and color alternatives

---

## Technical Implementation

### **Key Components**

```
# Core classes and modules
src/
├── models/
│   ├── yolo_detector.py      # YOLOv8 clothing detection
│   └── color_extractor.py    # K-means color analysis
├── analysis/
│   ├── harmony_analyzer.py   # Color theory implementation
│   └── recommendation.py     # Styling suggestions
├── utils/
│   ├── preprocessing.py      # Image preprocessing
│   ├── color_utils.py        # Color space conversions
│   └── visualization.py      # Results visualization
└── color_stylist.py          # Main pipeline orchestrator
```

### **Color Space Optimization**

**LAB vs RGB vs HSV Performance**:
- **LAB clustering**: 88% accuracy (perceptually uniform)
- **RGB clustering**: 76% accuracy 
- **HSV clustering**: 82% accuracy

LAB color space provides superior perceptual uniformity, enabling more meaningful color distances for fashion applications.

---

## Dataset Information

### **DeepFashion2 Statistics**
- **Total images**: 491K diverse fashion images
- **Clothing items**: 801K annotated items
- **Categories**: 13 popular clothing categories
- **Annotations**: Bounding boxes, segmentation masks, landmarks, style attributes
- **Split**: 80% training (391K images), 20% validation (100K images)
- **Resolution diversity**: 128×128 to 2048×2048 pixels
- **Average items per image**: 1.6

### **Class Distribution**
| Category | Instances | Performance (mAP50) |
|----------|-----------|-------------------|
| Short-sleeved shirt | 152K | 81% |
| Trousers | 100K+ | 79% |
| Long-sleeved shirt | 80K+ | 80% |
| Sling dress | 8K | 67% |

---

## Results and Validation

### **Detection Performance by Category**
- **Short-sleeved shirts**: 81% accuracy
- **Trousers**: 79% accuracy  
- **Long-sleeved shirts**: 80% accuracy
- **Dresses**: 95% precision, 96% recall
- **Lower performance on rare classes** (sling dresses: 67%) due to limited training examples

### **Color Harmony Validation**
- **Expert agreement**: 73% on harmony classifications
- **Recommendation agreement**: 81% with fashion experts
- **Realistic scoring**: Mean score 42.3% (σ=18.7%) addresses grade inflation
- **Processing speed**: Consistent 250ms per image across diverse inputs

### **Comparison with Existing Systems**
- **Detection improvement**: 79.5% mAP50 vs 36.9% mAP (Mask R-CNN)
- **Real-time performance**: 250ms vs >1000ms (two-stage detectors)
- **Realistic scoring**: 20-70% range vs 90-100% (grade inflation in competitors)

---

## Future Improvements

### **Technical Enhancements**
- **Texture Analysis**: Pattern recognition for richer styling advice
- **3D Pose Integration**: Body-shape-aware recommendations
- **Video Analysis**: Real-time styling feedback for virtual try-on
- **Online Learning**: Temporal trend adaptation

### **User Experience**
- **Personalization**: User preference history integration
- **Cultural Adaptation**: Region-specific color preferences
- **Mobile Application**: On-the-go styling assistance
- **Virtual Try-on**: Integration with AR/VR technologies

### **Domain Expansion**
- **Interior Design**: Color harmony for home decoration
- **Graphic Design**: Brand color palette optimization
- **Accessibility**: Enhanced support for color vision deficiencies

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### **Ways to Contribute**
- 🐛 **Bug Reports**: Report issues with detailed reproduction steps
- 💡 **Feature Requests**: Suggest new functionality or improvements
- 🔧 **Code Contributions**: Submit pull requests for bug fixes or features
- 📚 **Documentation**: Improve documentation and examples
- 🧪 **Testing**: Add test cases and improve coverage

### **Development Setup**
```
# Clone and setup development environment
git clone https://github.com/voloshynm/Deep_Fashion_Color.git
cd Deep_Fashion_Color
pip install -e .
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

### **Contribution Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use this work in your research, please cite our paper:

```
@article{voloshyn2025aicolorstylist,
  title={AI Color Stylist: Deep Learning-Based Fashion Color Harmony Analysis},
  author={Voloshyn, Maksym},
  journal={Introduction to Deep Learning},
  year={2025},
  month={May},
  institution={University of Luxembourg}
}
```

### **Related Work Citations**
```
@inproceedings{ge2019deepfashion2,
  title={DeepFashion2: A versatile benchmark for detection, pose estimation, segmentation and re-identification of clothing images},
  author={Ge, Yuying and Zhang, Ruimao and Wu, Lingyun and Wang, Xiaogang and Tang, Xiaoou and Luo, Ping},
  booktitle={CVPR},
  year={2019}
}

@article{ultralytics2023yolov8,
  title={YOLOv8: A new state-of-the-art computer vision model},
  author={Ultralytics},
  year={2023}
}
```

---

## License

This project is licensed under the **University of Luxembourg License**. See the [LICENSE](LICENSE) file for details.

### **Academic Use**
- ✅ Research and educational purposes
- ✅ Non-commercial applications
- ✅ Citation required for publications

### **Commercial Use**
- 📧 Contact for commercial licensing
- 💼 Enterprise solutions available
- 🤝 Collaboration opportunities welcome

---

## Acknowledgments

- **University of Luxembourg** - Academic support and resources
- **DeepFashion2 Team** - Comprehensive fashion dataset
- **Ultralytics** - YOLOv8 implementation and support
- **Fashion Theory Researchers** - Color harmony principles and validation

---

## Contact

**Author**: Maksym Voloshyn  
**Email**: maksym.voloshyn.002@student.uni.lu  
**Institution**: University of Luxembourg  
**Course**: Introduction to Deep Learning (May 2025)

**Project Links**:
- 🔗 **GitHub Repository**: [https://github.com/voloshynm/Deep_Fashion_Color](https://github.com/voloshynm/Deep_Fashion_Color)
- 📄 **Research Paper**: Available in repository
- 🎯 **Live Demo**: Coming soon

---

*Created with ❤️ for fashion and AI enthusiasts. Democratizing fashion expertise through deep learning.*

**Keywords**: `deep-learning` `computer-vision` `fashion-ai` `color-theory` `yolov8` `segmentation` `style-recommendation` `pytorch` `fashion-technology`
```

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/40467684/cc58e1a0-b8a2-4bc0-91bb-f987e69697e9/AI_Color_Stylist_Report.pdf
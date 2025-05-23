# Personal Stylist AI: Outfit Color Harmonizer

A deep learning-powered personal stylist that recognizes clothing items in photos, assesses outfit color harmony based on fashion color theory, and suggests color improvements to enhance style and coordination.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Architecture](#model-architecture)  
- [Color Theory Implementation](#color-theory-implementation)  
- [How It Works](#how-it-works)  
- [Future Improvements](#future-improvements)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

This project leverages state-of-the-art deep learning models trained on the DeepFashion dataset to detect and segment clothing items in user-uploaded images. It then extracts dominant colors from each clothing item and evaluates the overall outfit using established fashion color theory principles. When the outfit’s color harmony can be improved, the system recommends specific color changes for particular clothing items to create a balanced and stylish look.

---

## Features

- **Clothing Detection & Segmentation:** Uses Mask R-CNN fine-tuned on DeepFashion2 for accurate clothing recognition.
- **Color Extraction:** Identifies dominant colors of each clothing item via pixel clustering.
- **Color Harmony Assessment:** Applies the 3 Color Principle and color wheel relationships (complementary, analogous, triadic).
- **Personalized Suggestions:** Recommends which clothing item to change and suggests new colors to improve outfit harmony.
- **Modular Design:** Easily extendable to include texture, style, or occasion-based recommendations.

---

## Installation

1. Clone the repository:
  git clone https://github.com/yourusername/personal-stylist-ai.git
  cd personal-stylist-ai
2. Create and activate a virtual environment (optional but recommended):
  python3 -m venv venv
  source venv/bin/activate # Linux/macOS
  venv\Scripts\activate # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Download pre-trained DeepFashion/DeepFashion2 weights and place them in the `models/` directory (instructions in `models/README.md`).

---

## Usage

1. Run the inference script on an input image:
  python run_stylist.py --image path/to/outfit.jpg
2. The output will include:
- Detected clothing items with bounding boxes and segmentation masks.
- Extracted dominant colors per item.
- Color harmony assessment report.
- Suggested color changes for improved styling.

---

## Model Architecture

- **Clothing Detection:** Mask R-CNN with ResNet-50 backbone, fine-tuned on DeepFashion2.
- **Color Extraction:** K-means clustering on segmented clothing pixels to find dominant colors.
- **Suggestion Module:** Rule-based system implementing fashion color theory principles.

---

## Color Theory Implementation

- **3 Color Principle:** Limits outfit colors to a dominant, secondary, and accent color.
- **Color Wheel Relationships:** Checks for complementary, analogous, and triadic harmony.
- **Balance Ratios:** Suggests ideal color proportions (e.g., 60% dominant, 30% secondary, 10% accent).

---

## How It Works

1. **Input:** User uploads an outfit photo.
2. **Detection:** Model segments and classifies clothing items.
3. **Color Extraction:** Dominant colors are identified per clothing item.
4. **Assessment:** Outfit colors are analyzed for harmony and balance.
5. **Recommendation:** If disharmony is detected, the system suggests which item to recolor and proposes suitable replacement colors.

---

## Future Improvements

- Incorporate texture and pattern recognition for richer styling advice.
- Add user preference and occasion-based customization.
- Develop a mobile app for on-the-go styling assistance.
- Integrate with virtual try-on technology.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/personal-stylist-ai/issues).

---

## License

This project is licensed under the Univesity of Luxembourg License. See the [LICENSE](LICENSE) file for details.

---

*Created with ❤️ for fashion and AI enthusiasts.*

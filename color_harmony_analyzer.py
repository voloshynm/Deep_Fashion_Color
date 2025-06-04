# color_harmony_analyzer_fixed.py
import json
import colorsys
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class FixedColorHarmonyAnalyzer:
    def __init__(self):
        # Comprehensive color database covering ALL colors from your JSON
        self.color_database = {
            # Grays and Neutrals
            'darkgray': (169, 169, 169), 'gray': (128, 128, 128), 'lightgray': (211, 211, 211),
            'dimgray': (105, 105, 105), 'silver': (192, 192, 192), 'lightslategray': (119, 136, 153),
            'sage': (158, 169, 138), 'gainsboro': (220, 220, 220), 'charcoal': (54, 69, 79),
            'dark_gray': (64, 64, 64),
            
            # Blacks and Dark Colors
            'black': (0, 0, 0),
            
            # Reds and Warm Colors
            'red': (255, 0, 0), 'brown': (165, 42, 42), 'burgundy': (128, 0, 32),
            'maroon': (128, 0, 0), 'indianred': (205, 92, 92), 'lightcoral': (240, 128, 128),
            'darksalmon': (233, 150, 122), 'rosybrown': (188, 143, 143), 'rose_gold': (183, 110, 121),
            'thistle': (216, 191, 216),
            
            # Oranges and Yellows
            'orange': (255, 165, 0), 'yellow': (255, 255, 0), 'darkkhaki': (189, 183, 107),
            'tan': (210, 180, 140), 'wheat': (245, 222, 179), 'peach': (255, 218, 185),
            'mustard': (255, 219, 88),
            
            # Greens
            'green': (0, 128, 0), 'darkolivegreen': (85, 107, 47), 'mint': (189, 252, 201),
            'emerald': (80, 200, 120),
            
            # Blues and Purples
            'blue': (0, 0, 255), 'navy': (0, 0, 128), 'purple': (128, 0, 128),
            'darkslateblue': (72, 61, 139), 'slategray': (112, 128, 144),
            'lightsteelblue': (176, 196, 222),
            
            # Pinks
            'pink': (255, 192, 203), 'coral': (255, 127, 80), 'salmon': (250, 128, 114),
            
            # Neutrals and Beiges
            'white': (255, 255, 255), 'beige': (245, 245, 220), 'cream': (255, 253, 208),
            'ivory': (255, 255, 240), 'khaki': (240, 230, 140), 'denim': (21, 96, 189),
            'rust': (183, 65, 14), 'sapphire': (15, 82, 186)
        }
        
        # Color temperature classification
        self.warm_hues = [(0, 60), (300, 360)]  # Reds, oranges, yellows
        self.cool_hues = [(180, 300)]  # Blues, greens, purples
        self.neutral_colors = ['white', 'black', 'gray', 'silver', 'charcoal', 'beige', 'cream', 
                              'darkgray', 'lightgray', 'dimgray', 'gainsboro', 'sage']
        
        # Color recommendations
        self.color_recommendations = {
            'warm_to_cool': ['navy', 'blue', 'teal', 'sage', 'gray', 'charcoal'],
            'cool_to_warm': ['cream', 'beige', 'coral', 'peach', 'mustard', 'tan'],
            'neutral_safe': ['white', 'black', 'gray', 'navy', 'cream', 'beige'],
            'versatile': ['navy', 'white', 'gray', 'black', 'cream', 'charcoal'],
            'accent_colors': ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'coral', 'emerald'],
            'bright_colors': ['white', 'cream', 'light blue', 'light pink', 'yellow', 'coral']
        }

    def get_color_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """Get RGB values for a color name, with fallback for unknown colors"""
        color_name_lower = color_name.lower().replace(' ', '_')
        
        if color_name_lower in self.color_database:
            return self.color_database[color_name_lower]
        else:
            print(f"Warning: Unknown color '{color_name}' encountered. Adding to database as gray.")
            self.color_database[color_name_lower] = (128, 128, 128)
            return (128, 128, 128)

    def hex_to_hsv(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to HSV"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

    def classify_color_temperature(self, hex_color: str, color_name: str) -> str:
        """Classify color as warm, cool, or neutral"""
        color_name_lower = color_name.lower().replace(' ', '_')
        
        if color_name_lower in self.neutral_colors:
            return 'neutral'
        
        h, s, v = self.hex_to_hsv(hex_color)
        hue_degrees = h * 360
        
        # Check if hue falls in warm ranges
        for warm_start, warm_end in self.warm_hues:
            if warm_start <= hue_degrees <= warm_end:
                return 'warm'
        
        # Check if hue falls in cool ranges
        for cool_start, cool_end in self.cool_hues:
            if cool_start <= hue_degrees <= cool_end:
                return 'cool'
        
        return 'neutral'

    def analyze_outfit_harmony(self, detection_results: List[Dict]) -> Dict:
        """Analyze color harmony with guaranteed recommendations"""
        if not detection_results:
            return {"error": "No clothing items detected"}
        
        # Get unique items (avoid duplicates by taking highest confidence per class)
        unique_items = {}
        for item in detection_results:
            class_name = item['class_name']
            if class_name not in unique_items or item['confidence'] > unique_items[class_name]['confidence']:
                unique_items[class_name] = item
        
        outfit_analysis = {
            'items': [],
            'color_temperatures': {'warm': 0, 'cool': 0, 'neutral': 0},
            'harmony_score': 0,
            'issues': [],
            'recommendations': []
        }
        
        for item in unique_items.values():
            if not item.get('dominant_color'):
                continue
                
            dominant_color = item['dominant_color']
            color_temp = self.classify_color_temperature(
                dominant_color['hex'], 
                dominant_color['name']
            )
            
            # Get HSV values for harmony analysis
            h, s, v = self.hex_to_hsv(dominant_color['hex'])
            
            item_analysis = {
                'class_name': item['class_name'],
                'dominant_color': dominant_color,
                'temperature': color_temp,
                'hsv': {'h': h * 360, 's': s, 'v': v},
                'confidence': item['confidence']
            }
            
            outfit_analysis['items'].append(item_analysis)
            outfit_analysis['color_temperatures'][color_temp] += 1
        
        # Calculate harmony score with strict criteria
        harmony_score, issues = self._calculate_strict_harmony_score(outfit_analysis['items'])
        outfit_analysis['harmony_score'] = harmony_score
        outfit_analysis['issues'] = issues
        
        # Generate recommendations - ALWAYS provide recommendations
        outfit_analysis['recommendations'] = self._generate_guaranteed_recommendations(
            outfit_analysis['items'], issues
        )
        
        return outfit_analysis

    def _calculate_strict_harmony_score(self, items: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate harmony score with strict criteria"""
        if len(items) == 0:
            return 0.0, ["no_items"]
        
        if len(items) == 1:
            # Single items get maximum 60% unless they're interesting colors
            single_item = items[0]
            if single_item['temperature'] == 'neutral':
                return 0.35, ["single_neutral_item_boring"]
            else:
                return 0.55, ["single_colored_item_needs_complement"]
        
        issues = []
        score = 1.0
        
        # 1. MONOTONE/ALL NEUTRAL PENALTY
        neutral_count = sum(1 for item in items if item['temperature'] == 'neutral')
        total_items = len(items)
        
        if neutral_count == total_items:
            issues.append("completely_monotone_boring")
            score -= 0.5
        elif neutral_count >= total_items * 0.8:
            issues.append("mostly_neutral_lacks_interest")
            score -= 0.3
        
        # 2. ALL DARK PENALTY
        dark_items = sum(1 for item in items if item['hsv']['v'] < 0.4)
        if dark_items == total_items:
            issues.append("all_dark_depressing")
            score -= 0.3
        elif dark_items >= total_items * 0.8:
            issues.append("mostly_dark_gloomy")
            score -= 0.2
        
        # 3. ALL DESATURATED PENALTY
        desaturated_items = sum(1 for item in items if item['hsv']['s'] < 0.2)
        if desaturated_items == total_items:
            issues.append("all_washed_out")
            score -= 0.3
        elif desaturated_items >= total_items * 0.8:
            issues.append("mostly_washed_out")
            score -= 0.2
        
        # 4. TEMPERATURE MIXING ISSUES
        warm_count = sum(1 for item in items if item['temperature'] == 'warm')
        cool_count = sum(1 for item in items if item['temperature'] == 'cool')
        
        if warm_count > 0 and cool_count > 0:
            if abs(warm_count - cool_count) <= 1:
                issues.append("confusing_temperature_mix")
                score -= 0.35
            else:
                issues.append("minor_temperature_clash")
                score -= 0.15
        
        # 5. CONTRAST ISSUES
        values = [item['hsv']['v'] for item in items]
        contrast_range = max(values) - min(values)
        
        if contrast_range < 0.2:
            issues.append("insufficient_contrast")
            score -= 0.25
        elif contrast_range < 0.3:
            issues.append("low_contrast")
            score -= 0.15
        
        # 6. OVERSATURATION
        high_sat_items = sum(1 for item in items if item['hsv']['s'] > 0.7)
        if high_sat_items > 1:
            issues.append("too_many_bright_colors")
            score -= 0.3
        
        # 7. NO FOCAL POINT
        if total_items > 1:
            has_focal_point = any(
                item['hsv']['s'] > 0.5 or item['hsv']['v'] > 0.8 or item['temperature'] != 'neutral'
                for item in items
            )
            if not has_focal_point:
                issues.append("no_focal_point")
                score -= 0.2
        
        # Apply score compression
        score = max(0.0, min(1.0, score))
        
        if score > 0.8:
            score = 0.6 + (score - 0.8) * 0.5
        elif score > 0.6:
            score = 0.4 + (score - 0.6) * 0.75
        elif score > 0.4:
            score = 0.25 + (score - 0.4) * 0.75
        
        return score, issues

    def _generate_guaranteed_recommendations(self, items: List[Dict], issues: List[str]) -> List[Dict]:
        """Generate recommendations - ALWAYS provide at least one recommendation"""
        recommendations = []
        
        # If no issues detected, still provide improvement suggestions
        if not issues:
            if len(items) == 1:
                item = items[0]
                recommendations.append({
                    'item_to_change': 'Add complementary piece',
                    'current_color': {
                        'name': f"Current: {item['dominant_color']['name']}",
                        'hex': item['dominant_color']['hex'],
                        'rgb': item['dominant_color']['rgb']
                    },
                    'reason': 'single_item_needs_complement',
                    'suggested_colors': self._get_complementary_colors(item),
                    'priority': 'medium'
                })
            else:
                # Multiple items but no issues - suggest accent
                most_neutral = max(items, key=lambda x: 1 if x['temperature'] == 'neutral' else 0)
                recommendations.append({
                    'item_to_change': most_neutral['class_name'],
                    'current_color': {
                        'name': most_neutral['dominant_color']['name'],
                        'hex': most_neutral['dominant_color']['hex'],
                        'rgb': most_neutral['dominant_color']['rgb']
                    },
                    'reason': 'add_color_interest_for_better_style',
                    'suggested_colors': self.color_recommendations['accent_colors'][:3],
                    'priority': 'low'
                })
        else:
            # Process each issue and generate specific recommendations
            processed_items = set()
            
            for issue in issues:
                item_to_change = self._find_item_for_issue(items, issue, processed_items)
                if item_to_change:
                    suggestion = self._create_recommendation_for_issue(item_to_change, issue, items)
                    if suggestion:
                        recommendations.append(suggestion)
                        processed_items.add(item_to_change['class_name'])
            
            # If no specific item recommendations, provide general advice
            if not recommendations:
                fallback_item = self._select_fallback_item(items)
                recommendations.append({
                    'item_to_change': fallback_item['class_name'],
                    'current_color': {
                        'name': fallback_item['dominant_color']['name'],
                        'hex': fallback_item['dominant_color']['hex'],
                        'rgb': fallback_item['dominant_color']['rgb']
                    },
                    'reason': 'improve_overall_harmony',
                    'suggested_colors': self.color_recommendations['versatile'][:3],
                    'priority': 'medium'
                })
        
        return recommendations

    def _find_item_for_issue(self, items: List[Dict], issue: str, processed_items: set) -> Optional[Dict]:
        """Find the most appropriate item to change for a specific issue"""
        
        available_items = [item for item in items if item['class_name'] not in processed_items]
        if not available_items:
            return None
        
        if "monotone" in issue or "neutral" in issue:
            # Find most neutral item
            neutral_items = [item for item in available_items if item['temperature'] == 'neutral']
            if neutral_items:
                return max(neutral_items, key=lambda x: x['hsv']['v'])  # Lightest neutral
            return available_items[0]
        
        elif "dark" in issue:
            # Find darkest item
            return min(available_items, key=lambda x: x['hsv']['v'])
        
        elif "washed_out" in issue:
            # Find least saturated item
            return min(available_items, key=lambda x: x['hsv']['s'])
        
        elif "temperature" in issue:
            # Find minority temperature item
            warm_items = [item for item in available_items if item['temperature'] == 'warm']
            cool_items = [item for item in available_items if item['temperature'] == 'cool']
            
            if len(warm_items) < len(cool_items) and warm_items:
                return warm_items[0]
            elif len(cool_items) < len(warm_items) and cool_items:
                return cool_items[0]
            return available_items[0]
        
        elif "contrast" in issue:
            # Find darkest item to lighten
            return min(available_items, key=lambda x: x['hsv']['v'])
        
        elif "bright" in issue:
            # Find most saturated item
            return max(available_items, key=lambda x: x['hsv']['s'])
        
        elif "focal" in issue:
            # Find most boring item
            return min(available_items, key=lambda x: x['hsv']['s'] + x['hsv']['v'])
        
        else:
            return available_items[0]

    def _create_recommendation_for_issue(self, item: Dict, issue: str, all_items: List[Dict]) -> Dict:
        """Create a specific recommendation for an issue"""
        
        reason_map = {
            'completely_monotone_boring': 'outfit_too_monotone_needs_color',
            'mostly_neutral_lacks_interest': 'too_many_neutrals_add_color',
            'all_dark_depressing': 'outfit_too_dark_needs_brightness',
            'mostly_dark_gloomy': 'mostly_dark_add_lighter_color',
            'all_washed_out': 'colors_too_dull_need_vibrancy',
            'mostly_washed_out': 'mostly_dull_add_vibrant_color',
            'confusing_temperature_mix': 'mixed_temperatures_choose_one_palette',
            'minor_temperature_clash': 'slight_temperature_clash',
            'insufficient_contrast': 'needs_more_contrast',
            'low_contrast': 'could_use_more_contrast',
            'too_many_bright_colors': 'too_bright_tone_down',
            'no_focal_point': 'outfit_needs_focal_point',
            'single_neutral_item_boring': 'single_neutral_add_interest',
            'single_colored_item_needs_complement': 'single_item_add_complement'
        }
        
        suggestions_map = {
            'completely_monotone_boring': self.color_recommendations['accent_colors'],
            'mostly_neutral_lacks_interest': self.color_recommendations['accent_colors'],
            'all_dark_depressing': self.color_recommendations['bright_colors'],
            'mostly_dark_gloomy': self.color_recommendations['bright_colors'],
            'all_washed_out': self.color_recommendations['accent_colors'],
            'mostly_washed_out': self.color_recommendations['accent_colors'],
            'confusing_temperature_mix': self.color_recommendations['neutral_safe'],
            'minor_temperature_clash': self.color_recommendations['neutral_safe'],
            'insufficient_contrast': self.color_recommendations['bright_colors'],
            'low_contrast': self.color_recommendations['bright_colors'],
            'too_many_bright_colors': self.color_recommendations['neutral_safe'],
            'no_focal_point': self.color_recommendations['accent_colors'],
            'single_neutral_item_boring': self.color_recommendations['accent_colors'],
            'single_colored_item_needs_complement': self.color_recommendations['versatile']
        }
        
        reason = reason_map.get(issue, 'improve_color_harmony')
        suggested_colors = suggestions_map.get(issue, self.color_recommendations['versatile'])
        
        # Filter out existing colors
        existing_colors = {item['dominant_color']['name'].lower() for item in all_items}
        filtered_suggestions = [color for color in suggested_colors[:5] 
                              if color.lower() not in existing_colors]
        
        if not filtered_suggestions:
            filtered_suggestions = suggested_colors[:3]
        
        return {
            'item_to_change': item['class_name'],
            'current_color': {
                'name': item['dominant_color']['name'],
                'hex': item['dominant_color']['hex'],
                'rgb': item['dominant_color']['rgb']
            },
            'reason': reason,
            'suggested_colors': filtered_suggestions[:3],
            'priority': 'high' if 'monotone' in issue or 'dark' in issue else 'medium'
        }

    def _select_fallback_item(self, items: List[Dict]) -> Dict:
        """Select a fallback item when no specific issues are found"""
        # Prefer neutral items for color addition
        neutral_items = [item for item in items if item['temperature'] == 'neutral']
        if neutral_items:
            return neutral_items[0]
        
        # Otherwise, pick the least interesting item
        return min(items, key=lambda x: x['hsv']['s'] + x['hsv']['v'])

    def _get_complementary_colors(self, item: Dict) -> List[str]:
        """Get complementary colors for a single item"""
        temp = item['temperature']
        if temp == 'warm':
            return self.color_recommendations['warm_to_cool'][:3]
        elif temp == 'cool':
            return self.color_recommendations['cool_to_warm'][:3]
        else:
            return self.color_recommendations['accent_colors'][:3]

def analyze_fashion_colors_from_directory(directory_path: str = "./data/small_set/inference_results") -> Dict:
    """Analyze fashion colors from all JSON files with guaranteed recommendations"""
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist.")
        return {}
    
    analyzer = FixedColorHarmonyAnalyzer()
    all_results = {}
    
    # Find all JSON files in the directory
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return {}
    
    print(f"Found {len(json_files)} JSON files to analyze...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                detection_data = json.load(f)
            
            # Process each image in the JSON file
            for image_name, detections in detection_data.items():
                print(f"\n=== ANALYZING {image_name} ===")
                
                if not detections:
                    print("No clothing items detected in this image.")
                    continue
                
                analysis = analyzer.analyze_outfit_harmony(detections)
                all_results[image_name] = analysis
                
                # Print detected items with their main colors
                print("\n🔍 DETECTED CLOTHING ITEMS:")
                for item in analysis['items']:
                    color = item['dominant_color']
                    rgb = color['rgb']
                    print(f"  • {item['class_name'].replace('_', ' ').title()}: "
                          f"{color['name'].title()} (RGB: {rgb[0]}, {rgb[1]}, {rgb[2]})")
                
                # Print harmony analysis
                print(f"\n📊 HARMONY ANALYSIS:")
                print(f"  Harmony Score: {analysis['harmony_score']:.1%}")
                print(f"  Color Distribution: {analysis['color_temperatures']}")
                
                if analysis['issues']:
                    print(f"  Issues: {', '.join(analysis['issues']).replace('_', ' ').title()}")
                
                # Print recommendations
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(analysis['recommendations'], 1):
                    if 'item_to_change' in rec:
                        current_color = rec['current_color']
                        rgb = current_color['rgb']
                        print(f"  {i}. Change {rec['item_to_change'].replace('_', ' ').title()}")
                        print(f"     Current: {current_color['name'].title()} "
                              f"(RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}) {current_color['hex']}")
                        print(f"     Reason: {rec['reason'].replace('_', ' ').title()}")
                        print(f"     Suggested: {', '.join(rec['suggested_colors'])}")
                    else:
                        print(f"  {i}. {rec['message']}")
                
                print("-" * 60)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return all_results

# Main execution
if __name__ == "__main__":
    # Analyze all JSON files in the inference results directory
    results = analyze_fashion_colors_from_directory("./data/small_set/inference_results")
    
    # Save comprehensive analysis
    output_file = "./data/small_set/fixed_color_harmony_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Fixed analysis complete! Detailed results saved to: {output_file}")
    print(f"📈 Analyzed {len(results)} images total.")
    
    # Print summary statistics
    if results:
        scores = [result['harmony_score'] for result in results.values() if 'harmony_score' in result]
        recommendations_count = [len(result['recommendations']) for result in results.values() if 'recommendations' in result]
        
        if scores:
            print(f"\n📊 HARMONY SCORE STATISTICS:")
            print(f"  Average Score: {np.mean(scores):.1%}")
            print(f"  Highest Score: {max(scores):.1%}")
            print(f"  Lowest Score: {min(scores):.1%}")
            print(f"  Images with recommendations: {sum(1 for c in recommendations_count if c > 0)}/{len(recommendations_count)}")
            print(f"  Average recommendations per image: {np.mean(recommendations_count):.1f}")

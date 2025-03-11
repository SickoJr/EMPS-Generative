import torch
import cv2
import numpy as np
from torchvision import transforms
import yaml
from easydict import EasyDict
import os
import sys
import json
import base64

# Add the repository to Python path
sys.path.append('/emps/Landmark')

# Import HRNet model architecture
from Landmark.lib.models.pose_hrnet import get_pose_net

class LandmarkPredictor:
    def __init__(self, cfg_path, model_path):
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(cfg_path, 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            self.cfg = EasyDict(cfg_dict)

        self.model = self._initialize_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Define landmark ranges for each category and view (front/back)
        # Added 'mn' (middle neck) landmark with index 3 for front view
        self.gt_class_keypoints_dict = {
            # Front view (*1.jpg)
            "front": {
                1: [1, 5, 3, 14, 16, 6, 24, 8, 9, 22, 21],  # Polo with 'mn'
                2: [26, 30, 3, 43, 45, 31, 57, 35, 36, 53, 52],  # Long sleeve top (wy) with 'mn'
                3: [1, 5, 3, 14, 16, 6, 24, 8, 9, 22, 21],  # T-shirt with 'mn'
                4: [158, 159, 160, 164, 162, 163, 165, 166],  # Shorts
                5: [168, 169, 170, 176, 173, 174, 178, 179],  # Trousers
                6: [27, 29, 3, 43, 45, 31, 57, 35, 36, 53, 52],  # Zip-up with 'mn'
                7: [27, 29, 3, 43, 45, 31, 57, 35, 36, 53, 52],  # Windcoat with 'mn'
                8: [27, 29, 3, 43, 45, 31, 57, 35, 36, 53, 52],  # Jacket with 'mn'
                9: [182, 183, 184, 186, 187, 188],  # Skirt
                10: [190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218],  # Dress
            },
            # Back view (*2.jpg)
            "back": {
                1: [1, 5, 14, 16, 6, 24, 8, 9, 22, 21],  # Polo
                2: [26, 30, 43, 45, 31, 57, 35, 36, 53, 52],  # Long sleeve top (wy)
                3: [1, 5, 14, 16, 6, 24, 8, 9, 22, 21],  # T-shirt
                4: [158, 159, 160, 164, 162, 163, 165, 166],  # Shorts
                5: [168, 169, 170, 176, 173, 174, 178, 179],  # Trousers
                6: [27, 29, 43, 45, 31, 57, 35, 36, 53, 52],  # Zip-up
                7: [27, 29, 43, 45, 31, 57, 35, 36, 53, 52],  # Windcoat
                8: [27, 29, 43, 45, 31, 57, 35, 36, 53, 52],  # Jacket
                9: [182, 183, 184, 186, 187, 188],  # Skirt
                10: [190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218],  # Dress
            }
        }
        
        # Updated landmark names to include 'mn' for front view
        self.landmark_names_dict = {
            # Front view (*1.jpg)
            "front": {
                1: ['ln', 'rn', 'mn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Polo with 'mn'
                2: ['ln', 'rn', 'mn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Long sleeve top (wy) with 'mn'
                3: ['ln', 'rn', 'mn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # T-shirt with 'mn'
                4: ['lw', 'mw', 'rw', 'cc', 'llo', 'lli', 'rli', 'rlo'],  # Shorts
                5: ['lw', 'mw', 'rw', 'cc', 'llo', 'lli', 'rli', 'rlo'],  # Trousers
                6: ['ln', 'rn', 'mn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Zip-up with 'mn'
                7: ['ln', 'rn', 'mn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Windcoat with 'mn'
                8: ['ln', 'rn', 'mn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Jacket with 'mn'
                9: ['lw', 'mw', 'rw', 'll', 'ml', 'rl'],  # Skirt
                10: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Dress
            },
            # Back view (*2.jpg)
            "back": {
                1: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Polo
                2: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Long sleeve top (wy)
                3: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # T-shirt
                4: ['lw', 'mw', 'rw', 'cc', 'llo', 'lli', 'rli', 'rlo'],  # Shorts
                5: ['lw', 'mw', 'rw', 'cc', 'llo', 'lli', 'rli', 'rlo'],  # Trousers
                6: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Zip-up
                7: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Windcoat
                8: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Jacket
                9: ['lw', 'mw', 'rw', 'll', 'ml', 'rl'],  # Skirt
                10: ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],  # Dress
            }
        }

    def _initialize_model(self, model_path):
        model = get_pose_net(self.cfg, is_train=False)
        state_dict = torch.load(model_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        return model

    def preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Store the original image dimensions
        self.original_height, self.original_width = img.shape[:2]
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.cfg.MODEL.IMAGE_SIZE
        img = cv2.resize(img, (w, h))
        
        # Calculate scaling factors
        self.scale_x = self.original_width / w
        self.scale_y = self.original_height / h
        
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(self.device)

    def predict(self, image_path):
        img_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(img_tensor)
            if isinstance(output, list):
                output = output[-1]
            
            batch_heatmaps = output.cpu().numpy()
            landmarks = self._get_max_preds(batch_heatmaps)[0]

        # Denormalize landmarks to original image scale
        landmarks[:, 0] *= self.original_width  # Scale x coordinates
        landmarks[:, 1] *= self.original_height  # Scale y coordinates

        return landmarks

    def _get_max_preds(self, heatmaps):
        N, K, H, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        
        # Convert indices to x,y coordinates
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W
        
        # Normalize predictions to 0-1 range
        preds[:, :, 0] = preds[:, :, 0] / W
        preds[:, :, 1] = preds[:, :, 1] / H
        
        return preds

    def visualize_landmarks(self, image_path, landmarks, output_path, category, view):
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        # Landmarks are already in the original scale, so no need to scale again
        scaled_landmarks = landmarks.astype(np.int32)
        
        # Get landmark indices and names for the specified category and view
        landmark_indices = self.gt_class_keypoints_dict[view][category]
        landmark_names = self.landmark_names_dict[view][category]
        
        # Draw all landmarks for the category
        for idx, name in zip(landmark_indices, landmark_names):
            x, y = scaled_landmarks[idx]
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)  # Red dot
            cv2.putText(img, name, (x + 5, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green text
        
        cv2.imwrite(output_path, img)
        print(f"Landmark visualization saved to: {output_path}")

    def generate_landmark_json(self, landmarks, category, image_path, output_json_path, view):
        # Read the image to get its dimensions
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img_height, img_width = img.shape[:2]

        # Define the base structure of the JSON
        json_data = {
            "version": "4.6.0",
            "flags": {},
            "shapes": [],
            "imagePath": image_path,
            "imageData": None,  # Skip base64 encoding for now
            "imageWidth": img_width,  # Add image width
            "imageHeight": img_height  # Add image height
        }

        # Get landmark indices and names for the specified category and view
        landmark_indices = self.gt_class_keypoints_dict[view][category]
        landmark_names = self.landmark_names_dict[view][category]

        # Ensure the number of landmarks matches the expected count
        if len(landmark_indices) != len(landmark_names):
            raise ValueError(f"Landmark indices and names count mismatch for category {category} in {view} view.")

        # Add each landmark to the JSON structure
        for idx, name in zip(landmark_indices, landmark_names):
            if idx >= len(landmarks):
                raise ValueError(f"Landmark index {idx} is out of bounds for category {category} in {view} view.")
            
            x, y = landmarks[idx]
            shape = {
                "label": name,
                "points": [[float(x), float(y)]],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            }
            json_data["shapes"].append(shape)

        # Save the JSON data to a file
        with open(output_json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Landmark JSON saved to: {output_json_path}")

def main(image_path, category, view):
    try:
        current_dir = os.getcwd()
        
        cfg_path = os.path.join(current_dir, "/emps/Landmark/experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml")
        model_path = os.path.join(current_dir, "/emps/Landmark/models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth")
        
        # Output path for the visualization
        output_image_path = os.path.join(current_dir, "output_marked.jpg")
        output_json_path = os.path.join("/emps/temp", f"landmark_{view}.json")
        
        # Custom category mapping
        custom_category_mapping = {
            "1_sweater": {"id": 2, "name": "Long sleeve top"},
            "2_polo": {"id": 1, "name": "Polo"}, 
            "3_tshirt": {"id": 3, "name": "T-shirt"},
            "4_shorts": {"id": 4, "name": "Shorts"},
            "5_trousers": {"id": 5, "name": "Trousers"},
            "6_hoodie": {"id": 6, "name": "Zip-up"},
            "windcoat": {"id": 7, "name": "Windcoat"},
            "jacket": {"id": 8, "name": "Jacket"}, 
            "dress": {"id": 10, "name": "Dress"},
            "7_skirt": {"id": 9, "name": "Skirt"}
        }
        
        if category not in custom_category_mapping:
            raise ValueError(f"Invalid category. Choose from: {list(custom_category_mapping.keys())}")
        
        category_info = custom_category_mapping[category]
        category_id = category_info["id"]
        category_name = category_info["name"]
        
        if category_id is None:
            raise ValueError(f"Category '{category}' is not mapped to a DeepFashion2 category.")
        
        # Determine if the image is front view or back view based on filename
        # view = "front" if image_path.endswith("1.jpg") else "back"
        
        print(f"Using configuration file: {cfg_path}")
        print(f"Using model file: {model_path}")
        print(f"Processing image: {image_path}")
        print(f"Category: {category} ({category_name})")
        print(f"View: {view}")

        predictor = LandmarkPredictor(cfg_path, model_path)

        landmarks = predictor.predict(image_path)
        print(f"Detected {len(landmarks)} landmarks")
        
        # Save visualization of landmarks for the specified category and view
        # predictor.visualize_landmarks(image_path, landmarks, output_image_path, category_id, view)

        # Save landmarks to JSON file for the specified category and view
        predictor.generate_landmark_json(landmarks, category_id, image_path, output_json_path, view)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if all file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure all paths are correct and the input image is valid.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect landmarks in an image for a specified clothing category.")
    parser.add_argument("image_path", type=str, help="Path to the input image (front view ends with *1.jpg, back view ends with *2.jpg).")
    parser.add_argument("category", type=str, help="Clothing category (e.g., wy, tshirt, shorts, trousers, dress, skirt).")
    parser.add_argument("view", type=str, help="front or back")
    args = parser.parse_args()

    main(args.image_path, args.category, args.view)
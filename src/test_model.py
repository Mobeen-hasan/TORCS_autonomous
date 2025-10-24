import torch
import os
from train_model import CarControlNet

def test_model_loading():
    print("Current working directory:", os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    model_path = os.path.join(data_dir, 'best_car_model.pth')
    print("Looking for model at:", model_path)
    
    if os.path.exists(model_path):
        print("Model file exists!")
        try:
            print("Attempting to load model...")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            print("Model loaded successfully!")
            print("Checkpoint keys:", checkpoint.keys())
            
            # Try to create and load the model
            print("\nAttempting to create model...")
            model = CarControlNet(input_size=len(checkpoint['input_features']))
            print("Model created successfully!")
            
            print("\nAttempting to load state dict...")
            model.load_state_dict(checkpoint['model_state_dict'])
            print("State dict loaded successfully!")
            
            print("\nModel summary:")
            print("Input features:", checkpoint['input_features'])
            print("Target features:", checkpoint['target_features'])
            print("Model device:", next(model.parameters()).device)
            
        except Exception as e:
            print("Error loading model:", str(e))
            print("Error type:", type(e))
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
    else:
        print("Model file not found!")
        print("Directory contents:")
        for file in os.listdir('.'):
            print(f"- {file}")

if __name__ == '__main__':
    test_model_loading() 
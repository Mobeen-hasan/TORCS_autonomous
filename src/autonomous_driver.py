import torch
import numpy as np
from train_model import CarControlNet

class AutonomousDriver:
    def __init__(self, model_path='best_car_model.pth'):
        # Load the saved model and parameters
        checkpoint = torch.load(model_path)
        self.model = CarControlNet(input_size=len(checkpoint['input_features']))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set to evaluation mode
        
        self.scaler = checkpoint['scaler']
        self.input_features = checkpoint['input_features']
        self.target_features = checkpoint['target_features']
    
    def get_control(self, sensor_data):
        """
        Convert TORCS sensor data to model input and get control outputs
        
        Args:
            sensor_data: Dictionary containing TORCS sensor data
                {
                    'track': [...],  # Track sensor values
                    'speedX': float,
                    'speedY': float,
                    'speedZ': float,
                    'angle': float,
                    'trackPos': float,
                    'rpm': float,
                    'gear': int
                }
        
        Returns:
            Dictionary of control outputs:
                {
                    'accel': float,
                    'brake': float,
                    'gear': int,
                    'steer': float,
                    'clutch': float
                }
        """
        # Prepare input features
        input_data = []
        for feature in self.input_features:
            if feature.startswith('track_'):
                idx = int(feature.split('_')[1])
                input_data.append(sensor_data['track'][idx])
            else:
                input_data.append(sensor_data[feature])
        
        # Convert to numpy array and normalize
        input_data = np.array(input_data).reshape(1, -1)
        input_data = self.scaler.transform(input_data)
        
        # Convert to tensor and get model prediction
        input_tensor = torch.FloatTensor(input_data)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Convert outputs to dictionary
        controls = {}
        for i, feature in enumerate(self.target_features):
            value = outputs[0][i].item()
            
            # Apply appropriate constraints
            if feature == 'gear':
                value = int(round(value))  # Round to nearest integer
                value = max(-1, min(6, value))  # Limit to valid gear range
            elif feature in ['accel', 'brake', 'clutch']:
                value = max(0.0, min(1.0, value))  # Limit to [0, 1]
            elif feature == 'steer':
                value = max(-1.0, min(1.0, value))  # Limit to [-1, 1]
            
            controls[feature] = value
        
        # Add wall collision detection and emergency steering
        track_data = sensor_data['track']
        if track_data and len(track_data) >= 9:
            # Look at sensors 3-7 (center and slightly to sides)
            center_sensors = track_data[3:8]
            min_dist = min(center_sensors)
            
            # Emergency wall avoidance
            if min_dist < 20:  # Very close to wall
                # Find the direction to steer away from the wall
                sensor_idx = center_sensors.index(min_dist)
                steer_direction = (sensor_idx - 2) / 2.0  # Convert to [-1, 1] range
                
                # Strong steering away from wall
                emergency_steer = steer_direction * 1.0
                
                # Reduce speed when very close to wall
                if min_dist < 10:
                    controls['accel'] = max(0.0, controls['accel'] - 0.5)
                    controls['brake'] = min(1.0, controls['brake'] + 0.3)
                
                # Override normal steering with emergency steering
                controls['steer'] = emergency_steer
        
        # Apply automatic transmission logic
        current_speed = abs(sensor_data['speedX'])
        current_rpm = sensor_data['rpm']
        current_gear = sensor_data['gear']
        
        # Gear shifting thresholds - exactly matching the manual implementation
        GEAR_SHIFT_POINTS = {
            1: {'up_rpm': 3500, 'speed_threshold': 20},   # 1st to 2nd
            2: {'up_rpm': 4500, 'speed_threshold': 40},   # 2nd to 3rd
            3: {'up_rpm': 5500, 'speed_threshold': 60},   # 3rd to 4th
            4: {'up_rpm': 6500, 'speed_threshold': 80},   # 4th to 5th
            5: {'up_rpm': 7000, 'speed_threshold': 100}   # 5th to 6th
        }
        
        # Determine transmission state based on input - exactly matching manual implementation
        if current_speed < 5 and current_rpm != 0:
            controls['gear'] = 1  # First gear for low speeds
        elif current_rpm == 0:
            controls['gear'] = 1  # First gear if RPM is zero
        elif current_gear > 0:
            # Check for upshifting
            if (current_rpm > GEAR_SHIFT_POINTS.get(current_gear, {}).get('up_rpm', 7000) and 
                current_speed > GEAR_SHIFT_POINTS.get(current_gear, {}).get('speed_threshold', 100)):
                controls['gear'] = min(current_gear + 1, 6)
            # Check for downshifting
            elif current_rpm < 2000 and current_gear > 1:
                controls['gear'] = current_gear - 1
            else:
                controls['gear'] = current_gear
        else:
            controls['gear'] = 1  # Default to first gear
        
        return controls

# Example usage in TORCS client
def process_torcs_data(torcs_data):
    """
    Example function showing how to use the autonomous driver in your TORCS client
    """
    driver = AutonomousDriver()
    
    # Convert TORCS data to the format expected by the model
    sensor_data = {
        'track': torcs_data['track'],
        'speedX': torcs_data['speedX'],
        'speedY': torcs_data['speedY'],
        'speedZ': torcs_data['speedZ'],
        'angle': torcs_data['angle'],
        'trackPos': torcs_data['trackPos'],
        'rpm': torcs_data['rpm'],
        'gear': torcs_data['gear']
    }
    
    # Get control outputs
    controls = driver.get_control(sensor_data)
    
    # Format the control string for TORCS
    control_string = f"(accel {controls['accel']:.3f})" \
                    f"(brake {controls['brake']:.3f})" \
                    f"(gear {controls['gear']})" \
                    f"(steer {controls['steer']:.3f})" \
                    f"(clutch {controls['clutch']:.3f})" \
                    f"(focus 0)(meta 0)"
    
    return control_string 
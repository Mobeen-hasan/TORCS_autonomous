import msgParser
import carState
import carControl
import csv
import os
from datetime import datetime
import math
import torch
import numpy as np
from train_model import CarControlNet

class Driver(object):
    '''
    A driver object for the SCRC with automatic transmission and neural network control
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Connect the control to this driver instance
        self.control.set_driver(self)
        
        self.steer_lock = 0.785398
        self.max_speed = 250  # Reasonable top speed

        # Gear shifting RPM thresholds for automatic transmission
        self.GEAR_SHIFT_POINTS = {
            1: {'up_rpm': 3500, 'speed_threshold': 20},   # 1st to 2nd
            2: {'up_rpm': 4500, 'speed_threshold': 40},   # 2nd to 3rd
            3: {'up_rpm': 5500, 'speed_threshold': 60},   # 3rd to 4th
            4: {'up_rpm': 6500, 'speed_threshold': 80},   # 4th to 5th
            5: {'up_rpm': 7000, 'speed_threshold': 100}   # 5th to 6th
        }

        # Prepare telemetry logging
        self.telemetry_log = []
        self.log_filename = f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Neural Network setup
        self.autonomous_mode = False
        self.nn_model = None
        self.nn_scaler = None
        self.input_features = None
        self.target_features = None
        
        # Try to load the neural network model
        try:
            self.load_neural_network()
        except Exception as e:
            print(f"Could not load neural network model: {e}")
            print("Autonomous mode will not be available until a model is trained.")
    
    def load_neural_network(self):
        '''Load the trained neural network model'''
        try:
            # Get the absolute path to the model file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            model_path = os.path.join(data_dir, 'best_car_model.pth')
            
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {script_dir}")
            print(f"Data directory: {data_dir}")
            print(f"Looking for model at: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                try:
                    print("Found model file, attempting to load...")
                    print(f"Model file size: {os.path.getsize(model_path)} bytes")
                    
                    # Add numpy's reconstruct function to safe globals
                    import numpy as np
                    import torch.serialization
                    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
                    
                    # Load with weights_only=False since we trust our own model file
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                    print("Model file loaded successfully!")
                    print(f"Checkpoint keys: {checkpoint.keys()}")
                    
                    # Create model with correct input size
                    input_size = len(checkpoint['input_features'])
                    print(f"Creating model with input size: {input_size}")
                    self.nn_model = CarControlNet(input_size=input_size)
                    
                    # Load model state
                    print("Loading model state...")
                    self.nn_model.load_state_dict(checkpoint['model_state_dict'])
                    self.nn_model.eval()
                    print("Model state loaded!")
                    
                    # Recreate scaler from saved parameters
                    print("Recreating scaler...")
                    from sklearn.preprocessing import StandardScaler
                    self.nn_scaler = StandardScaler()
                    self.nn_scaler.mean_ = checkpoint['scaler_params']['mean_']
                    self.nn_scaler.scale_ = checkpoint['scaler_params']['scale_']
                    self.nn_scaler.var_ = checkpoint['scaler_params']['var_']
                    self.nn_scaler.n_samples_seen_ = checkpoint['scaler_params']['n_samples_seen_']
                    print("Scaler recreated!")
                    
                    # Store features
                    self.input_features = checkpoint['input_features']
                    self.target_features = checkpoint['target_features']
                    
                    print("Neural network model loaded successfully!")
                    print(f"Input features: {self.input_features}")
                    print(f"Target features: {self.target_features}")
                    print(f"Model device: {next(self.nn_model.parameters()).device}")
                    return True
                    
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    print("Full error traceback:")
                    print(traceback.format_exc())
                    # Reset model state
                    self.nn_model = None
                    self.nn_scaler = None
                    self.input_features = None
                    self.target_features = None
                    return False
            else:
                print(f"Model file not found at {model_path}")
                print("Directory contents:")
                for file in os.listdir(script_dir):
                    print(f"- {file}")
                return False
                
        except Exception as outer_e:
            print(f"Outer error in load_neural_network: {str(outer_e)}")
            print(f"Error type: {type(outer_e)}")
            import traceback
            print("Full outer error traceback:")
            print(traceback.format_exc())
            return False
    
    def toggle_autonomous_mode(self):
        '''Toggle between manual and autonomous driving'''
        print("Toggle autonomous mode called!")  # Debug message
        
        # If model is not loaded, try to load it
        if self.nn_model is None:
            print("Model not loaded, attempting to load...")
            if not self.load_neural_network():
                print("Failed to load neural network model")
                print("Cannot enable autonomous mode: No trained model available")
                print("Model state:", self.nn_model)
                return
        
        # Toggle mode only if model is loaded
        self.autonomous_mode = not self.autonomous_mode
        print(f"Autonomous mode {'enabled' if self.autonomous_mode else 'disabled'}")
        print("Model state:", self.nn_model)
    
    def get_neural_network_control(self):
        '''Get control values from the neural network'''
        if not self.nn_model:
            return None
            
        try:
            # Prepare input features
            input_data = []
            for feature in self.input_features:
                try:
                    if feature.startswith('track_'):
                        idx = int(feature.split('_')[1])
                        track_data = self.state.getTrack()
                        if track_data and idx < len(track_data):
                            # Handle invalid track readings
                            value = float(track_data[idx])
                            if value == -1:  # Invalid reading
                                value = 200.0  # Default to max range
                            input_data.append(value)
                        else:
                            input_data.append(200.0)  # Default value if track data is missing
                    elif feature == 'speed_x':
                        input_data.append(float(self.state.getSpeedX()))
                    elif feature == 'speed_y':
                        input_data.append(float(self.state.getSpeedY()))
                    elif feature == 'speed_z':
                        input_data.append(float(self.state.getSpeedZ()))
                    elif feature == 'angle':
                        input_data.append(float(self.state.getAngle()))
                    elif feature == 'track_pos':
                        track_pos = self.state.getTrackPos()
                        if isinstance(track_pos, str):
                            if track_pos.lower() == 'pos':
                                input_data.append(0.0)  # Default to center of track
                            else:
                                try:
                                    input_data.append(float(track_pos))
                                except (ValueError, TypeError):
                                    input_data.append(0.0)  # Default to center on error
                        else:
                            try:
                                input_data.append(float(track_pos))
                            except (ValueError, TypeError):
                                input_data.append(0.0)  # Default to center on error
                    elif feature == 'rpm':
                        input_data.append(float(self.state.getRpm()))
                    elif feature == 'dist_raced':
                        input_data.append(float(self.state.getDistRaced()))
                    elif feature == 'dist_from_start':
                        input_data.append(float(self.state.getDistFromStart()))
                    elif feature == 'race_pos':
                        race_pos = self.state.getRacePos()
                        if isinstance(race_pos, str):
                            if race_pos.lower() == 'pos':
                                input_data.append(1.0)  # Default to position 1
                            else:
                                try:
                                    input_data.append(float(race_pos))
                                except (ValueError, TypeError):
                                    input_data.append(1.0)  # Default to position 1 on error
                        else:
                            try:
                                input_data.append(float(race_pos))
                            except (ValueError, TypeError):
                                input_data.append(1.0)  # Default to position 1 on error
                    elif feature == 'damage':
                        input_data.append(float(self.state.getDamage()))
                    elif feature == 'fuel':
                        input_data.append(float(self.state.getFuel()))
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Error processing feature {feature}: {e}")
                    input_data.append(0.0)  # Default value on error
            
            # Ensure input data matches expected size
            if len(input_data) != len(self.input_features):
                print(f"Warning: Input data size mismatch. Expected {len(self.input_features)}, got {len(input_data)}")
                # Pad or truncate to match expected size
                if len(input_data) < len(self.input_features):
                    input_data.extend([0.0] * (len(self.input_features) - len(input_data)))
                else:
                    input_data = input_data[:len(self.input_features)]
            
            # Convert to numpy array and normalize
            input_data = np.array(input_data).reshape(1, -1)
            input_data = self.nn_scaler.transform(input_data)
            
            # Get model prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data)
                outputs = self.nn_model(input_tensor)
            
            # Extract predicted values
            next_speed_x = outputs[0][0].item()
            next_speed_y = outputs[0][1].item()
            next_angle = outputs[0][2].item()
            next_track_pos = outputs[0][3].item()
            predicted_steer = outputs[0][4].item()  # Direct steering prediction
            
            # Initialize controls dictionary
            controls = {}
            
            # Get current state
            track_data = self.state.getTrack()
            current_speed = math.sqrt(self.state.getSpeedX()**2 + self.state.getSpeedY()**2)
            current_angle = self.state.getAngle()
            track_pos = self.state.getTrackPos()
            
            # Calculate steering based on track sensors and model prediction
            if track_data and len(track_data) >= 9:
                # Get center sensors (3-7)
                center_sensors = track_data[3:8]
                min_dist = min(center_sensors)
                
                # Calculate track position correction (increased for better centering)
                track_correction = -float(track_pos) * 0.5  # Increased from 0.3
                
                # Calculate path using all sensors (balanced sensor groups)
                left_sensors = track_data[0:4]   # Left side sensors
                right_sensors = track_data[6:10]  # Right side sensors
                center_sensors = track_data[3:8]  # Overlapping center sensors for better path detection
                
                # Calculate average distances with error checking
                try:
                    # Filter out invalid readings (-1)
                    left_sensors = [x for x in left_sensors if x != -1]
                    right_sensors = [x for x in right_sensors if x != -1]
                    center_sensors = [x for x in center_sensors if x != -1]
                    
                    if not left_sensors: left_sensors = [200.0]
                    if not right_sensors: right_sensors = [200.0]
                    if not center_sensors: center_sensors = [200.0]
                    
                    left_avg = sum(float(x) for x in left_sensors) / len(left_sensors)
                    right_avg = sum(float(x) for x in right_sensors) / len(right_sensors)
                    center_avg = sum(float(x) for x in center_sensors) / len(center_sensors)
                except (ValueError, ZeroDivisionError):
                    print("Error in sensor readings")
                    left_avg = right_avg = center_avg = 100.0
                
                # Debug print sensor values
                print(f"\nSensor values - Left: {left_avg:.2f}, Right: {right_avg:.2f}, Center: {center_avg:.2f}")
                
                # Calculate path direction with balanced influence
                path_direction = (right_avg - left_avg) / 800.0  # Keep extremely reduced sensitivity
                
                # More aggressive boundary handling
                boundary_threshold = 40.0  # Increased from 30.0 to start turning earlier
                emergency_threshold = 20.0  # Increased from 15.0 for smoother transitions
                
                # Calculate boundary proximity factor (1.0 when very close, 0.0 when far)
                boundary_proximity = 1.0 - min(1.0, max(0.0, min_dist / boundary_threshold))
                
                # Determine if we're near a boundary
                near_boundary = min_dist < boundary_threshold
                emergency_boundary = min_dist < emergency_threshold
                
                # Calculate road alignment angle
                # Use the difference between left and right sensors to determine road angle
                road_angle = math.atan2(right_avg - left_avg, 100.0)  # 100.0 is approximate sensor distance
                current_angle = float(self.state.getAngle())
                angle_diff = road_angle - current_angle
                
                # Normalize angle difference to [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # Speed control based on boundary proximity and angle alignment
                if emergency_boundary:
                    controls['accel'] = 0.0
                    controls['brake'] = 0.8  # Strong braking
                elif near_boundary:
                    # Gradually reduce speed based on boundary proximity and angle misalignment
                    speed_reduction = boundary_proximity * 0.8
                    angle_factor = min(1.0, abs(angle_diff) / (math.pi / 4))  # Reduce speed more when angle is off
                    controls['accel'] = max(0.0, 1.0 - (speed_reduction + angle_factor * 0.3))
                    controls['brake'] = (speed_reduction + angle_factor * 0.3) * 0.5
                else:
                    # Normal speed control with slight reduction for angle misalignment
                    angle_factor = min(1.0, abs(angle_diff) / (math.pi / 2))
                    if speed_diff > 0:
                        controls['accel'] = min(1.0, speed_diff / 5.0) * (1.0 - angle_factor * 0.2)
                        controls['brake'] = 0.0
                    else:
                        controls['accel'] = 0.0
                        controls['brake'] = min(1.0, -speed_diff / 10.0)
                
                # Enhanced steering for boundary situations
                if emergency_boundary:
                    # Emergency steering - align with road first, then avoid boundary
                    road_alignment = angle_diff * 0.7  # Strong road alignment
                    # Calculate steer direction based on track position
                    steer_direction = -1 if track_pos > 0 else 1
                    boundary_avoidance = steer_direction * 0.3  # Less aggressive boundary avoidance
                    final_steer = road_alignment + boundary_avoidance
                    print(f"Emergency steering: {final_steer:.3f}")
                elif near_boundary:
                    # Enhanced boundary steering with road alignment
                    steering_factors = []
                    
                    # Road alignment factor
                    alignment_weight = 0.4 * (1 + boundary_proximity)
                    steering_factors.append(angle_diff * alignment_weight)
                    
                    # Path following with reduced weight
                    path_weight = 0.2 * (1 + boundary_proximity)
                    steering_factors.append(path_direction * path_weight)
                    
                    # Track position correction with reduced weight
                    track_correction_weight = 0.2 * (1 + boundary_proximity)
                    steering_factors.append(track_correction * track_correction_weight)
                    
                    # Calculate final steering with boundary influence
                    final_steer = sum(steering_factors) / len(steering_factors)
                    final_steer *= (1 + boundary_proximity * 0.3)  # Reduced boundary influence
                    
                    print(f"Boundary steering: {final_steer:.3f}")
                else:
                    # Normal steering logic with emphasis on road alignment
                    steering_factors = []
                    
                    # Primary focus on road alignment
                    alignment_weight = 0.5
                    steering_factors.append(angle_diff * alignment_weight)
                    
                    # Secondary focus on track position
                    if abs(track_pos) > 0.1:  # Increased threshold
                        track_weight = 0.3
                        steering_factors.append(track_correction * track_weight)
                    
                    # Calculate base steering
                    if steering_factors:
                        final_steer = sum(steering_factors) / len(steering_factors)
                    else:
                        final_steer = 0.0
                    
                    # Add curve steering only in extreme cases
                    if abs(path_direction) > 0.5:
                        curve_steer = path_direction * 0.1  # Reduced curve steering
                        final_steer = final_steer + curve_steer
                    
                    print(f"Normal steering: {final_steer:.3f}")
                
                # Smooth steering changes
                if hasattr(self, 'last_steer'):
                    # Allow smoother steering changes
                    max_steer_change = 0.3  # Reduced from 1.0 for smoother transitions
                    steer_diff = final_steer - self.last_steer
                    if abs(steer_diff) > max_steer_change:
                        final_steer = self.last_steer + (max_steer_change * (1 if steer_diff > 0 else -1))
                
                # Store current steering
                self.last_steer = final_steer
                
                # Apply final steering with minimal deadzone
                if abs(final_steer) < 0.001:
                    final_steer = 0.0
                controls['steer'] = max(-1.0, min(1.0, final_steer))
                
                print(f"Final steering: {controls['steer']:.3f}")
            else:
                # Fallback to model prediction if track data is missing
                controls['steer'] = max(-1.0, min(1.0, predicted_steer * 1.0))  # Increased from 0.8
            
            # Speed and acceleration control
            target_speed = math.sqrt(next_speed_x**2 + next_speed_y**2)
            speed_diff = target_speed - current_speed
            
            # Maintain consistent acceleration on straights
            if track_data and len(track_data) >= 9:
                center_sensors = track_data[3:8]
                min_dist = min(center_sensors)
                max_dist = max(center_sensors)
                is_straight = (max_dist - min_dist) < 20
                
                if is_straight and current_speed < 100:
                    controls['accel'] = 1.0
                    controls['brake'] = 0.0
                else:
                    if speed_diff > 0:
                        controls['accel'] = min(1.0, speed_diff / 5.0)
                        controls['brake'] = 0.0
                    else:
                        controls['accel'] = 0.0
                        controls['brake'] = min(1.0, -speed_diff / 10.0)
            else:
                if speed_diff > 0:
                    controls['accel'] = min(1.0, speed_diff / 5.0)
                    controls['brake'] = 0.0
                else:
                    controls['accel'] = 0.0
                    controls['brake'] = min(1.0, -speed_diff / 10.0)
            
            # Gear control
            current_rpm = self.state.getRpm()
            current_gear = self.state.getGear()
            current_speed = abs(self.state.getSpeedX())
            
            if current_speed < 5 and current_rpm != 0:
                controls['gear'] = 1
            elif current_rpm == 0:
                controls['gear'] = 1
            elif current_gear > 0:
                if (current_rpm > self.GEAR_SHIFT_POINTS.get(current_gear, {}).get('up_rpm', 7000) and 
                    current_speed > self.GEAR_SHIFT_POINTS.get(current_gear, {}).get('speed_threshold', 100)):
                    controls['gear'] = min(current_gear + 1, 6)
                elif current_rpm < 2000 and current_gear > 1:
                    controls['gear'] = current_gear - 1
                else:
                    controls['gear'] = current_gear
            else:
                controls['gear'] = 1
            
            # Clutch control
            controls['clutch'] = 0.0
            
            return controls
            
        except Exception as e:
            print(f"Error in neural network control: {e}")
            print("Stack trace:", e.__traceback__)
            return None
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        # Parse incoming sensor data
        self.state.setFromMsg(msg)
        
        # Log telemetry data
        telemetry_entry = {
            'angle': self.state.getAngle(),
            'dist_raced': self.state.getDistRaced(),
            'race_pos': self.state.getRacePos(),
            'track_pos': self.state.getTrackPos(),
            'rpm': self.state.getRpm(),
            'speed_x': self.state.getSpeedX(),
            'speed_y': self.state.getSpeedY(),
            'speed_z': self.state.getSpeedZ(),
            'gear': self.state.getGear(),
            'track': str(self.state.getTrack()),
            'opponents': str(self.state.getOpponents()),
            'fuel': self.state.getFuel(),
            'damage': self.state.getDamage(),
            'dist_from_start': self.state.getDistFromStart(),
            'wheel_spin_vel': str(self.state.getWheelSpinVel()),
            'curLapTime': self.state.getCurLapTime(),
            'z': self.state.getZ(),
            'accel': round(self.control.getAccel(), 3),
            'brake': round(self.control.getBrake(), 3),
            'steer': round(self.control.getSteer(), 3),
            'clutch': round(self.control.getClutch(), 3)
        }
        self.telemetry_log.append(telemetry_entry)
        
        if self.autonomous_mode and self.nn_model:
            # Get control values from neural network
            controls = self.get_neural_network_control()
            if controls:
                # Debug print all control values
                print("\nApplying control values:")
                print(f"Acceleration: {controls['accel']:.3f}")
                print(f"Brake: {controls['brake']:.3f}")
                print(f"Steering: {controls['steer']:.3f}")
                print(f"Gear: {controls['gear']}")
                print(f"Clutch: {controls['clutch']:.3f}")
                
                # Apply all control values
                self.control.setAccel(controls['accel'])
                self.control.setBrake(controls['brake'])
                self.control.setGear(controls['gear'])
                self.control.setSteer(controls['steer'])
                self.control.setClutch(controls['clutch'])
        else:
            # Manual control implementation
            keys_pressed = self.control.keys_pressed
            current_speed = abs(self.state.getSpeedX())
            current_gear = self.state.getGear()
            
            # Handle steering (A and D keys) - Fixed swapped controls
            if keys_pressed.get('d', False):
                self.control.setSteer(-1.0)  # Full left
            elif keys_pressed.get('a', False):
                self.control.setSteer(1.0)   # Full right
            else:
                self.control.setSteer(0.0)   # No steering
            
            # Handle acceleration and braking (W and S keys)
            if keys_pressed.get('w', False):
                # Forward acceleration
                self.control.setAccel(1.0)
                self.control.setBrake(0.0)
                # Switch to first gear if in reverse or almost stopped
                if current_gear == -1 or current_speed < 5:
                    self.control.setGear(1)
            elif keys_pressed.get('s', False):
                if current_speed < 5 or current_gear == -1:  # If almost stopped or already in reverse
                    self.control.setGear(-1)  # Ensure we're in reverse
                    self.control.setAccel(1.0)  # Accelerate in reverse
                    self.control.setBrake(0.0)  # No braking
                else:
                    self.control.setAccel(0.0)
                    self.control.setBrake(1.0)  # Full brake
            else:
                # No keys pressed - coast
                self.control.setAccel(0.0)
                self.control.setBrake(0.0)
            
            # Only manage automatic transmission if not in reverse
            if current_gear != -1:
                new_gear = self.manage_automatic_transmission()
                self.control.setGear(new_gear)
            
            # Set clutch to 0 for manual control
            self.control.setClutch(0.0)
        
        return self.control.toMsg()
    
    def manage_automatic_transmission(self):
        '''
        Implement automatic transmission logic
        '''
        # Get current state
        current_rpm = self.state.getRpm()
        current_gear = self.state.getGear()
        current_speed = abs(self.state.getSpeedX())

        # Determine transmission state based on input
        keys_pressed = self.control.keys_pressed

        # Reverse logic
        if current_speed < 5 and current_rpm != 0:
            if keys_pressed['s'] and not keys_pressed['w']:
                return -1  # Reverse gear
            elif keys_pressed['w'] and not keys_pressed['s']:
                return 1  # First gear

        # Reset to first gear if RPM is zero and acceleration is desired
        elif current_rpm == 0 and (keys_pressed['w'] or keys_pressed['s']):
            return 1  # First gear

        # Normal driving logic
        if current_gear > 0:
            # Check for upshifting
            if (current_rpm > self.GEAR_SHIFT_POINTS.get(current_gear, {}).get('up_rpm', 7000) and 
                current_speed > self.GEAR_SHIFT_POINTS.get(current_gear, {}).get('speed_threshold', 100)):
                return min(current_gear + 1, 6)
            
            # Check for downshifting
            if current_rpm < 2000 and current_gear > 1:
                return current_gear - 1
        
        # If no shift is needed, return current gear
        return current_gear
    
    def onShutDown(self):
        # Write telemetry to CSV file
        if self.telemetry_log:
            # Get the absolute path to the data directory (one level up from src)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Ask user if they want to store the data
            store_data = input("\nDo you want to store the race data? (yes/no): ").lower().strip()
            
            if store_data in ['yes', 'y']:
                filepath = os.path.join(data_dir, self.log_filename)
                
                with open(filepath, 'w', newline='') as csvfile:
                    # Determine all unique keys across all entries
                    fieldnames = set()
                    for entry in self.telemetry_log:
                        fieldnames.update(entry.keys())
                    
                    writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
                    writer.writeheader()
                    writer.writerows(self.telemetry_log)
                print(f"\nRace data saved to: {filepath}")
            else:
                print("\nRace data not saved.")
    
    def onRestart(self):
        pass
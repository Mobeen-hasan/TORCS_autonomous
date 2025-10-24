import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import os
import argparse

# Define the neural network architecture
class CarControlNet(nn.Module):
    def __init__(self, input_size):
        super(CarControlNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Output: [next_speed_x, next_speed_y, next_angle, next_track_pos, steer]
        )
    
    def forward(self, x):
        return self.network(x)

# Custom Dataset class
class CarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(data_dir=None, map_name=None):
    """
    Load and preprocess data from CSV files.
    
    Args:
        data_dir (str): Base directory for data
        map_name (str): Name of the map to load data for
    """
    # If no data_dir is provided, use the default location in src
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
    
    # Create map-specific directory if map_name is provided
    if map_name:
        data_dir = os.path.join(data_dir, map_name)
        os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n=== Loading Training Data ===")
    print(f"Base data directory: {data_dir}")
    if map_name:
        print(f"Loading data for map: {map_name}")
        print(f"Looking for CSV files in: {data_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, 'telemetry_*.csv'))
    if not csv_files:
        if map_name:
            print(f"\nNo telemetry CSV files found in {data_dir}")
            print("Please collect some data for this map first.")
            print("You can move existing CSV files to this directory:")
            print(f"  {data_dir}")
            print("\nCurrent directory structure:")
            for root, dirs, files in os.walk(os.path.join(script_dir, 'data')):
                level = root.replace(script_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f"{subindent}{f}")
        else:
            print(f"No telemetry CSV files found in {data_dir}")
            print("Please collect some data first.")
        raise FileNotFoundError(f"No telemetry CSV files found in {data_dir}")
    
    # Sort files by timestamp (newest first)
    csv_files.sort(reverse=True)
    
    print(f"\nFound {len(csv_files)} telemetry files:")
    for i, file in enumerate(csv_files, 1):
        file_size = os.path.getsize(file) / 1024  # Size in KB
        print(f"{i}. {os.path.basename(file)} ({file_size:.1f} KB)")
        
        # Print first few lines of each file
        print(f"\nFirst 2 lines of {os.path.basename(file)}:")
        with open(file, 'r') as f:
            for j, line in enumerate(f):
                if j < 2:  # Print first 2 lines
                    print(line.strip())
                else:
                    break
        print()
    
    print("\nReading and combining files...")
    
    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        print(f"Reading: {os.path.basename(file)}")
        # Read CSV with explicit header handling
        df = pd.read_csv(file, header=0)  # header=0 means first row is header
        
        # Convert numeric columns to float
        numeric_columns = ['accel', 'angle', 'brake', 'clutch', 'curLapTime', 'damage', 
                         'dist_from_start', 'dist_raced', 'fuel', 'gear', 'race_pos', 
                         'rpm', 'speed_x', 'speed_y', 'speed_z', 'steer', 'track_pos', 'z']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle track data - split into individual sensors
        # First convert to string and remove quotes if present
        track_str = df['track'].astype(str).str.replace('"', '')
        track_data = track_str.str.strip('[]').str.split(',', expand=True)
        track_data = track_data.apply(pd.to_numeric, errors='coerce')
        track_data.columns = [f'track_{i}' for i in range(len(track_data.columns))]
        
        # Handle opponents data - split into individual sensors
        opponents_str = df['opponents'].astype(str).str.replace('"', '')
        opponents_data = opponents_str.str.strip('[]').str.split(',', expand=True)
        opponents_data = opponents_data.apply(pd.to_numeric, errors='coerce')
        opponents_data.columns = [f'opponent_{i}' for i in range(len(opponents_data.columns))]
        
        # Handle wheel spin velocity data - split into individual values
        wheel_spin_str = df['wheel_spin_vel'].astype(str).str.replace('"', '')
        wheel_spin_data = wheel_spin_str.str.strip('[]').str.split(',', expand=True)
        wheel_spin_data = wheel_spin_data.apply(pd.to_numeric, errors='coerce')
        wheel_spin_data.columns = [f'wheel_spin_{i}' for i in range(len(wheel_spin_data.columns))]
        
        # Combine all data
        df = pd.concat([df, track_data, opponents_data, wheel_spin_data], axis=1)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal data points: {len(data):,}")
    print(f"Total files processed: {len(dfs)}")
    print("=== Data Loading Complete ===\n")
    
    # Add wall-related features
    # Calculate minimum distance to wall from center sensors
    data['min_wall_dist'] = data[[f'track_{i}' for i in range(3, 8)]].min(axis=1)
    
    # Calculate wall direction (which side is closer)
    center_sensors = [f'track_{i}' for i in range(3, 8)]
    data['wall_direction'] = data[center_sensors].idxmin(axis=1).apply(lambda x: (int(x.split('_')[1]) - 5) / 2.0)
    
    # Calculate wall proximity factor (1.0 when very close, 0.0 when far)
    data['wall_proximity'] = 1.0 - (data['min_wall_dist'] / 100.0).clip(0, 1)
    
    # Define input features
    input_features = [
        # Track sensors
        'track_0', 'track_1', 'track_2', 'track_3', 'track_4', 
        'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
        # Basic state
        'speed_x', 'speed_y', 'speed_z', 'angle', 'track_pos',
        'rpm', 'dist_raced', 'dist_from_start', 'race_pos',
        'damage', 'fuel',
        # Wall-related features
        'min_wall_dist', 'wall_direction', 'wall_proximity'
    ]
    
    # Target features
    target_features = ['speed_x', 'speed_y', 'angle', 'track_pos', 'steer']
    
    # Create next state values by shifting the current values
    for feature in target_features:
        if feature != 'steer':  # Don't shift steering as it's a direct action
            data[f'next_{feature}'] = data[feature].shift(-1)
    
    # Remove the last row since it won't have next state values
    data = data[:-1]
    
    # Prepare input data
    X = data[input_features].values
    y = data[[f'next_{col}' if col != 'steer' else col for col in target_features]].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, input_features, target_features

def train_model(load_previous_model=False, map_name=None):
    """
    Train the model with options to load previous model and specify map.
    
    Args:
        load_previous_model (bool): If True, loads the previous model and continues training
        map_name (str): If provided, only uses CSV files from this map
    """
    # Load and preprocess data
    X, y, scaler, input_features, target_features = load_and_preprocess_data(map_name=map_name)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    
    # Initialize model
    model = CarControlNet(input_size=len(input_features))
    
    # Load previous model if requested
    if load_previous_model:
        try:
            checkpoint = torch.load('best_car_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded previous model successfully!")
        except Exception as e:
            print(f"Could not load previous model: {e}")
            print("Starting with a new model instead.")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 200
    best_val_loss = float('inf')
    patience = 10000  # Early stopping patience
    no_improve = 0  # Counter for epochs without improvement
    
    print(f"\nStarting training with {len(dataset)} data points")
    print(f"Input features: {input_features}")
    print(f"Target features: {target_features}")
    if map_name:
        print(f"Training on map: {map_name}")
    
    # Track training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_steer_loss': [],
        'val_steer_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steer_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate losses for each output separately
            speed_loss = criterion(outputs[:, :2], targets[:, :2])  # Speed components
            angle_loss = criterion(outputs[:, 2], targets[:, 2])    # Angle
            track_pos_loss = criterion(outputs[:, 3], targets[:, 3])  # Track position
            steer_loss = criterion(outputs[:, 4], targets[:, 4])    # Steering
            
            # Get wall proximity from inputs (last feature)
            wall_proximity = inputs[:, -1]
            
            # Increase steering loss weight when close to walls
            wall_weight = 1.0 + (wall_proximity * 2.0)  # Up to 3x weight when very close to wall
            
            # Weight the losses (give more importance to steering and wall proximity)
            total_loss = speed_loss + angle_loss + track_pos_loss + (wall_weight * steer_loss).mean()
            
            train_steer_loss += steer_loss.item()
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_steer_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                
                # Calculate losses for each output separately
                speed_loss = criterion(outputs[:, :2], targets[:, :2])
                angle_loss = criterion(outputs[:, 2], targets[:, 2])
                track_pos_loss = criterion(outputs[:, 3], targets[:, 3])
                steer_loss = criterion(outputs[:, 4], targets[:, 4])
                
                # Get wall proximity from inputs (last feature)
                wall_proximity = inputs[:, -1]
                
                # Increase steering loss weight when close to walls
                wall_weight = 1.0 + (wall_proximity * 2.0)  # Up to 3x weight when very close to wall
                
                # Weight the losses (same as training)
                total_loss = speed_loss + angle_loss + track_pos_loss + (wall_weight * steer_loss).mean()
                
                val_loss += total_loss.item()
                val_steer_loss += steer_loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        avg_train_steer_loss = train_steer_loss/len(train_loader)
        avg_val_steer_loss = val_steer_loss/len(val_loader)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_steer_loss'].append(avg_train_steer_loss)
        history['val_steer_loss'].append(avg_val_steer_loss)
        
        # Print progress with more details
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f} (Steering: {avg_train_steer_loss:.4f})')
        print(f'Validation Loss: {avg_val_loss:.4f} (Steering: {avg_val_steer_loss:.4f})')
        
        # Early stopping check with more detailed output
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            improvement = ((best_val_loss - val_loss) / best_val_loss) * 100
            print(f'New best model! Improvement: {improvement:.2f}%')
            
            # Save best model
            scaler_params = {
                'mean_': scaler.mean_,
                'scale_': scaler.scale_,
                'var_': scaler.var_,
                'n_samples_seen_': scaler.n_samples_seen_
            }
            
            # Add map name to model file if provided
            model_filename = f'best_car_model_{map_name}.pth' if map_name else 'best_car_model.pth'
            
            # Create data directory if it doesn't exist
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Save model in data directory
            model_path = os.path.join(data_dir, model_filename)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_params': scaler_params,
                'input_features': input_features,
                'target_features': target_features,
                'training_history': history,
                'map_name': map_name
            }, model_path)
            print(f'Model saved as {model_path}!')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f}')
                print(f'Final training loss: {avg_train_loss:.4f}')
                print(f'Final validation loss: {avg_val_loss:.4f}')
                break
    
    print('\nTraining completed!')
    print(f'Final training loss: {history["train_loss"][-1]:.4f}')
    print(f'Final validation loss: {history["val_loss"][-1]:.4f}')
    print(f'Final training steering loss: {history["train_steer_loss"][-1]:.4f}')
    print(f'Final validation steering loss: {history["val_steer_loss"][-1]:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the car control model')
    parser.add_argument('--load', action='store_true', help='Load previous model and continue training')
    parser.add_argument('--map', type=str, help='Specify map name for training')
    args = parser.parse_args()
    
    train_model(load_previous_model=args.load, map_name=args.map) 
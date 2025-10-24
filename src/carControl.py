import msgParser
from pynput import keyboard

class CarControl:
    '''
    An object holding all the control parameters of the car
    '''
    def __init__(self):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        
        self.actions = None
        
        # Initialize control attributes
        self.accel = 0.0
        self.brake = 0.0
        self.gear = 1
        self.steer = 0.0
        self.clutch = 0.0
        self.focus = [-1, -1, -1, -1, -1]
        self.meta = 0
        
        self.driver = None
        self.keys_pressed = {
            'w': False,  # Forward
            's': False,  # Backward
            'a': False,  # Left
            'd': False,  # Right
            'up': False,    # Gear up
            'down': False,  # Gear down
            'space': False,  # Brake
            't': False     # Toggle autonomous mode
        }
        
        # Setup keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.listener.start()
    
    def set_driver(self, driver):
        '''Set the driver instance'''
        self.driver = driver
    
    def on_key_press(self, key):
        '''Handle key press events'''
        try:
            k = key.char.lower()
            if k in self.keys_pressed:
                self.keys_pressed[k] = True
                if k == 't' and self.driver:
                    self.driver.toggle_autonomous_mode()
        except AttributeError:
            if key == keyboard.Key.up:
                self.keys_pressed['up'] = True
            elif key == keyboard.Key.down:
                self.keys_pressed['down'] = True
            elif key == keyboard.Key.space:
                self.keys_pressed['space'] = True
    
    def on_key_release(self, key):
        '''Handle key release events'''
        try:
            k = key.char.lower()
            if k in self.keys_pressed:
                self.keys_pressed[k] = False
        except AttributeError:
            if key == keyboard.Key.up:
                self.keys_pressed['up'] = False
            elif key == keyboard.Key.down:
                self.keys_pressed['down'] = False
            elif key == keyboard.Key.space:
                self.keys_pressed['space'] = False
    
    def toMsg(self):
        self.actions = {}
        
        self.actions['accel'] = [self.accel]
        self.actions['brake'] = [self.brake]
        self.actions['gear'] = [self.gear]
        self.actions['steer'] = [self.steer]
        self.actions['clutch'] = [self.clutch]
        self.actions['focus'] = [self.focus]
        self.actions['meta'] = [self.meta]
        
        return self.parser.stringify(self.actions)
    
    # Existing getter and setter methods remain the same with bounds checking
    def setAccel(self, accel):
        self.accel = max(0.0, min(1.0, accel))
    
    def getAccel(self):
        return self.accel
    
    def setBrake(self, brake):
        self.brake = max(0.0, min(1.0, brake))
    
    def getBrake(self):
        return self.brake
    
    def setGear(self, gear):
        # Allow -1 (reverse), 0 (neutral), and 1-6 (drive gears)
        self.gear = max(-1, min(6, gear))
    
    def getGear(self):
        return self.gear
    
    def setSteer(self, steer):
        self.steer = max(-1.0, min(1.0, steer))
    
    def getSteer(self):
        return self.steer
    
    def setClutch(self, clutch):
        self.clutch = max(0.0, min(1.0, clutch))
    
    def getClutch(self):
        return self.clutch
    
    def setMeta(self, meta):
        self.meta = meta
    
    def getMeta(self):
        return self.meta
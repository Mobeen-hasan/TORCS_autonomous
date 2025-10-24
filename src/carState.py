from msgParser import MsgParser

class CarState:
    '''
    Class that holds all the car state variables
    '''

    def __init__(self):
        '''Constructor'''
        self.parser = MsgParser()
        self.sensors = None
        self.angle = None
        self.curLapTime = None
        self.damage = None
        self.distFromStart = None
        self.distRaced = None
        self.focus = None
        self.fuel = None
        self.gear = None
        self.lastLapTime = None
        self.opponents = None
        self.racePos = None
        self.rpm = None
        self.speedX = None
        self.speedY = None
        self.speedZ = None
        self.track = None
        self.trackPos = None
        self.wheelSpinVel = None
        self.z = None
    
    def setFromMsg(self, str_sensors: str):
        '''Parse a UDP message and update car state variables'''
        # First, handle race position specially
        if 'racePos pos' in str_sensors:
            str_sensors = str_sensors.replace('racePos pos', 'racePos 1')
        
        self.sensors = self.parser.parse(str_sensors)
        
        if self.sensors is None:
            return  # Stop if parsing failed
        
        self.setAngleD()
        self.setCurLapTimeD()
        self.setDamageD()
        self.setDistFromStartD()
        self.setDistRacedD()
        self.setFocusD()
        self.setFuelD()
        self.setGearD()
        self.setLastLapTimeD()
        self.setOpponentsD()
        self.setRacePosD()
        self.setRpmD()
        self.setSpeedXD()
        self.setSpeedYD()
        self.setSpeedZD()
        self.setTrackD()
        self.setTrackPosD()
        self.setWheelSpinVelD()
        self.setZD()
    
    def toMsg(self) -> str:
        '''Convert car state to a UDP message'''
        self.sensors = {
            'angle': [self.angle],
            'curLapTime': [self.curLapTime],
            'damage': [self.damage],
            'distFromStart': [self.distFromStart],
            'distRaced': [self.distRaced],
            'focus': self.focus,
            'fuel': [self.fuel],
            'gear': [self.gear],
            'lastLapTime': [self.lastLapTime],
            'opponents': self.opponents,
            'racePos': [self.racePos],
            'rpm': [self.rpm],
            'speedX': [self.speedX],
            'speedY': [self.speedY],
            'speedZ': [self.speedZ],
            'track': self.track,
            'trackPos': [self.trackPos],
            'wheelSpinVel': self.wheelSpinVel,
            'z': [self.z],
        }
        return self.parser.stringify(self.sensors)
    
    def getFloatD(self, name: str) -> float:
        '''Retrieve a float value from sensors dictionary'''
        val = self.sensors.get(name)
        return float(val[0]) if val else None
    
    def getFloatListD(self, name: str) -> list:
        '''Retrieve a list of float values from sensors dictionary'''
        val = self.sensors.get(name)
        return [float(v) for v in val] if val else None
    
    def getIntD(self, name: str) -> int:
        '''Retrieve an integer value from sensors dictionary'''
        val = self.sensors.get(name)
        if not val:
            return None
        try:
            # Special handling for race position
            if name == 'racePos':
                if isinstance(val[0], str):
                    if val[0].lower() == 'pos':
                        return 1  # Default to position 1 when we get 'pos'
                    try:
                        return int(val[0])
                    except ValueError:
                        return 1  # Default to position 1 if parsing fails
            return int(val[0])
        except (ValueError, TypeError):
            # If conversion fails, return None
            return None
    
    # Getter & Setter methods for each sensor variable
    def setAngle(self, angle: float): self.angle = angle
    def setAngleD(self): self.angle = self.getFloatD('angle')
    def getAngle(self) -> float: return self.angle
    
    def setCurLapTime(self, curLapTime: float): self.curLapTime = curLapTime
    def setCurLapTimeD(self): self.curLapTime = self.getFloatD('curLapTime')
    def getCurLapTime(self) -> float: return self.curLapTime
    
    def setDamage(self, damage: float): self.damage = damage
    def setDamageD(self): self.damage = self.getFloatD('damage')
    def getDamage(self) -> float: return self.damage
    
    def setDistFromStart(self, distFromStart: float): self.distFromStart = distFromStart
    def setDistFromStartD(self): self.distFromStart = self.getFloatD('distFromStart')
    def getDistFromStart(self) -> float: return self.distFromStart
    
    def setDistRaced(self, distRaced: float): self.distRaced = distRaced
    def setDistRacedD(self): self.distRaced = self.getFloatD('distRaced')
    def getDistRaced(self) -> float: return self.distRaced
    
    def setFocus(self, focus: list): self.focus = focus
    def setFocusD(self): self.focus = self.getFloatListD('focus')
    
    def setFuel(self, fuel: float): self.fuel = fuel
    def setFuelD(self): self.fuel = self.getFloatD('fuel')
    def getFuel(self) -> float: return self.fuel
    
    def setGear(self, gear: int): self.gear = gear
    def setGearD(self): self.gear = self.getIntD('gear')
    def getGear(self) -> int: return self.gear
    
    def setLastLapTime(self, lastLapTime: float): self.lastLapTime = lastLapTime
    def setLastLapTimeD(self): self.lastLapTime = self.getFloatD('lastLapTime')
    
    def setOpponents(self, opponents: list): self.opponents = opponents
    def setOpponentsD(self): self.opponents = self.getFloatListD('opponents')
    def getOpponents(self) -> list: return self.opponents
    
    def setRacePos(self, racePos: int): self.racePos = racePos
    def setRacePosD(self):
        '''Set race position from sensors'''
        try:
            val = self.sensors.get('racePos')
            if not val:
                self.racePos = 1  # Default to position 1 if not set
                return
                
            if isinstance(val[0], str):
                if val[0].lower() == 'pos':
                    self.racePos = 1  # Default to position 1 when we get 'pos'
                else:
                    try:
                        self.racePos = int(val[0])
                    except ValueError:
                        self.racePos = 1  # Default to position 1 if parsing fails
            else:
                self.racePos = int(val[0])
        except (ValueError, TypeError):
            self.racePos = 1  # Default to position 1 if any error occurs
    def getRacePos(self) -> int:
        '''Get race position with safe fallback'''
        try:
            if self.racePos is None:
                return 1  # Default to position 1 if not set
            if isinstance(self.racePos, str):
                if self.racePos.lower() == 'pos':
                    return 1
                try:
                    return int(self.racePos)
                except ValueError:
                    return 1
            return int(self.racePos)
        except (ValueError, TypeError):
            return 1  # Default to position 1 if any error occurs
    
    def setRpm(self, rpm: float): self.rpm = rpm
    def setRpmD(self): self.rpm = self.getFloatD('rpm')
    def getRpm(self) -> float: return self.rpm
    
    def setSpeedX(self, speedX: float): self.speedX = speedX
    def setSpeedXD(self): self.speedX = self.getFloatD('speedX')
    def getSpeedX(self) -> float: return self.speedX
    
    def setSpeedY(self, speedY: float): self.speedY = speedY
    def setSpeedYD(self): self.speedY = self.getFloatD('speedY')
    def getSpeedY(self) -> float: return self.speedY
    
    def setSpeedZ(self, speedZ: float): self.speedZ = speedZ
    def setSpeedZD(self): self.speedZ = self.getFloatD('speedZ')
    def getSpeedZ(self) -> float: return self.speedZ
    
    def setTrack(self, track: list): self.track = track
    def setTrackD(self): self.track = self.getFloatListD('track')
    def getTrack(self) -> list: return self.track
    
    def setTrackPos(self, trackPos: float): self.trackPos = trackPos
    def setTrackPosD(self): self.trackPos = self.getFloatD('trackPos')
    def getTrackPos(self) -> float: return self.trackPos
    
    def setWheelSpinVel(self, wheelSpinVel: list): self.wheelSpinVel = wheelSpinVel
    def setWheelSpinVelD(self): self.wheelSpinVel = self.getFloatListD('wheelSpinVel')
    def getWheelSpinVel(self) -> list: return self.wheelSpinVel
    
    def setZ(self, z: float): self.z = z
    def setZD(self): self.z = self.getFloatD('z')
    def getZ(self) -> float: return self.z
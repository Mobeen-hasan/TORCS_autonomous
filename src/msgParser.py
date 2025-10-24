class MsgParser:
    '''
    A parser for received UDP messages and building UDP messages
    '''
    def __init__(self):
        '''Constructor'''
        pass
    
    def parse(self, str_sensors: str) -> dict:
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        b_open = str_sensors.find('(')
        
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close >= 0:
                substr = str_sensors[b_open + 1: b_close]
                items = substr.split()
                if len(items) < 2:
                    print("Problem parsing substring:", substr)
                else:
                    # Handle all values
                    value = []
                    for i in range(1, len(items)):
                        try:
                            # Try to convert to float first (for decimal numbers)
                            val = float(items[i])
                            # If it's a whole number, convert to int
                            if val.is_integer():
                                val = int(val)
                        except ValueError:
                            # If conversion fails, keep as string
                            val = items[i]
                        value.append(val)
                    sensors[items[0]] = value
                b_open = str_sensors.find('(', b_close)
            else:
                print("Problem parsing sensor string:", str_sensors)
                return None
        
        return sensors
    
    def stringify(self, dictionary: dict) -> str:
        '''Build a UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value and value[0] is not None:
                msg += f'({key}'
                for val in value:
                    msg += f' {str(val)}'
                msg += ')'
        
        return msg
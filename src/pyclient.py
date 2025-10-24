import sys
import argparse
import socket
import driver
import os
import time

def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR',
                        help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=2,
                        help='Maximum number of learning episodes (default: 2)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                        help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default=None,
                        help='Name of the track')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
    parser.add_argument('--customInput', action='store', dest='custom_input', default=None,
                        help='Path to CSV file with custom input')

    arguments = parser.parse_args()

    # Enhanced debug logging
    print('=== TORCS CLIENT CONFIGURATION ===')
    print('Host IP:', arguments.host_ip)
    print('Host Port:', arguments.host_port)
    print('Bot ID:', arguments.id)
    print('Maximum Episodes:', arguments.max_episodes)
    print('Maximum Steps:', arguments.max_steps)
    print('Track:', arguments.track)
    print('Stage:', arguments.stage)
    print('Custom Input:', arguments.custom_input)
    print('==================================')

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Increase timeout and add socket options
        sock.settimeout(5.0)  # Increased timeout to 5 seconds
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except socket.error as msg:
        print('FATAL ERROR: Could not create socket.', msg)
        sys.exit(-1)

    shutdownClient = False
    curEpisode = 0

    verbose = True  # Enable verbose logging for debugging

    d = driver.Driver(arguments.stage)

    # Load custom input if provided
    if arguments.custom_input and os.path.exists(arguments.custom_input):
        d.control.load_custom_input(arguments.custom_input)

    while not shutdownClient:
        try:
            # Robust connection initialization
            connection_attempts = 0
            while connection_attempts < 3:
                print(f'Connection Attempt {connection_attempts + 1}')
                
                # Send initialization
                buf = arguments.id + d.init()
                print('Sending init string:', buf)
                
                try:
                    sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
                except socket.error as send_msg:
                    print(f"ERROR: Failed to send initialization. {send_msg}")
                    connection_attempts += 1
                    time.sleep(1)
                    continue

                try:
                    buf, addr = sock.recvfrom(1000)
                    buf = buf.decode()
                    print(f'Received from {addr}: {buf}')
                except socket.timeout:
                    print("Connection timeout. Server not responding.")
                    connection_attempts += 1
                    time.sleep(1)
                    continue
                except socket.error as recv_msg:
                    print(f"ERROR receiving data: {recv_msg}")
                    connection_attempts += 1
                    time.sleep(1)
                    continue

                if '***identified***' in buf:
                    print('Server Identified Successfully!')
                    break
            
            if connection_attempts >= 3:
                print("FATAL: Could not establish connection after 3 attempts.")
                break

            currentStep = 0
            
            while True:
                try:
                    buf, addr = sock.recvfrom(1000)
                    buf = buf.decode()
                except socket.timeout:
                    print("Socket receive timeout. Attempting to continue.")
                    continue
                except socket.error as msg:
                    print(f"Socket error during receive: {msg}")
                    break

                if verbose:
                    print('Received:', buf)
                
                if '***shutdown***' in buf:
                    d.onShutDown()
                    shutdownClient = True
                    print('Client Shutdown')
                    break
                
                if '***restart***' in buf:
                    d.onRestart()
                    print('Client Restart')
                    break
                
                currentStep += 1
                if currentStep != arguments.max_steps:
                    if buf:
                        buf = d.drive(buf)
                else:
                    buf = '(meta 1)'
                
                if verbose:
                    print('Sending:', buf)
                
                if buf:
                    try:
                        sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
                    except socket.error as msg:
                        print(f"Failed to send data: {msg}")
                        break
        
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            break

        curEpisode += 1
        
        if curEpisode == arguments.max_episodes:
            shutdownClient = True

    sock.close()
    print("Client Terminated.")

if __name__ == '__main__':
    main()
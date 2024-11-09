import os
import numpy as np
from dnslib.server import DNSServer, BaseResolver, DNSLogger
from dnslib import RR, TXT, QTYPE, RCODE
import time
import zlib
import base64
import traceback

# Import ViZDoom
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, Button

# Import OpenCV
import cv2

# Initialize DoomGame
def initialize_game():
    game = DoomGame()
    
    # Set the path to your original Doom WAD file
    game.set_doom_game_path("DOOM.WAD")  # Replace with the actual path
    
    game.set_window_visible(True)  # Set to True to display the game window
    game.set_mode(Mode.PLAYER)
    
    # Add all necessary buttons for full gameplay
    buttons = [
        Button.MOVE_FORWARD,     # Index 0
        Button.MOVE_BACKWARD,    # Index 1
        Button.TURN_LEFT,        # Index 2
        Button.TURN_RIGHT,       # Index 3
        Button.MOVE_LEFT,        # Index 4
        Button.MOVE_RIGHT,       # Index 5
        Button.ATTACK,           # Index 6
        Button.USE,              # Index 7
        Button.JUMP,             # Index 8
        Button.CROUCH,           # Index 9
        Button.SELECT_WEAPON1,   # Index 10
        Button.SELECT_WEAPON2,   # Index 11
        Button.SELECT_WEAPON3,   # Index 12
        Button.SELECT_WEAPON4,   # Index 13
        Button.SELECT_WEAPON5,   # Index 14
        Button.SELECT_WEAPON6,   # Index 15
        Button.SELECT_WEAPON7,   # Index 16
        Button.SELECT_WEAPON8,   # Index 17
        Button.RELOAD,           # Index 18
        Button.ZOOM,             # Index 19
        Button.SPEED,            # Index 20
    ]
    for button in buttons:
        game.add_available_button(button)
    
    # Set screen format and resolution
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)  # Increased resolution
    
    # Start at the first level
    game.set_doom_map("MAP01")  # For Doom II
    # For Ultimate Doom, use game.set_doom_map("E1M1")
    
    game.init()
    return game

# Map input commands to Doom actions
def map_input_to_action(input_command):
    # Initialize all buttons to 0
    action = [0] * 21  # Number of buttons added
    # Map the input command to the action
    command_map = {
        'w': 0,              # MOVE_FORWARD
        's': 1,              # MOVE_BACKWARD
        'left': 2,           # TURN_LEFT
        'right': 3,          # TURN_RIGHT
        'a': 4,              # MOVE_LEFT
        'd': 5,              # MOVE_RIGHT
        'mouse1': 6,         # ATTACK
        'k': 6,              # ATTACK (alternative key)
        'use': 7,            # USE
        'u': 7,              # USE (alternative key)
        'space': 8,          # JUMP
        'ctrl': 9,           # CROUCH
        'c': 9,              # CROUCH (alternative key)
        '1': 10,             # SELECT_WEAPON1
        '2': 11,             # SELECT_WEAPON2
        '3': 12,             # SELECT_WEAPON3
        '4': 13,             # SELECT_WEAPON4
        '5': 14,             # SELECT_WEAPON5
        '6': 15,             # SELECT_WEAPON6
        '7': 16,             # SELECT_WEAPON7
        '8': 17,             # SELECT_WEAPON8
        'r': 18,             # RELOAD
        'zoom': 19,          # ZOOM
        'shift': 20,         # SPEED
    }
    if input_command in command_map:
        index = command_map[input_command]
        action[index] = 1
        return action
    else:
        return [0] * 21  # Return no action if command is 'idle' or unrecognized

# Improved frame to ASCII conversion
def frame_to_ascii(frame, cols=160, rows=90):
    # Convert frame to grayscale
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()

    # Resize the image
    resized_frame = cv2.resize(gray_frame, (cols, rows), interpolation=cv2.INTER_AREA)

    # Normalize the resized frame to [0,1]
    normalized_frame = resized_frame / 255.0

    # Apply Gaussian blurs with different sigmas
    sigma1 = 1.0
    sigma2 = sigma1 * 1.6  # Sigma scale from the shader
    blur1 = cv2.GaussianBlur(normalized_frame, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    blur2 = cv2.GaussianBlur(normalized_frame, (0, 0), sigmaX=sigma2, sigmaY=sigma2)

    # Compute Difference of Gaussians (DoG)
    tau = 0.98  # Detail level (adjustable)
    dog = blur1 - tau * blur2

    # Threshold the DoG to detect edges
    threshold = 0.005  # Edge threshold (adjustable)
    edge_map = (dog >= threshold).astype(np.float32)

    # Apply Sobel operator to get edge gradients
    sobelx = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(sobelx, sobely)
    angle = np.arctan2(sobely, sobelx)  # Angle in radians between -π and π

    # Normalize angle to [0,1] for quantization
    abs_theta = np.abs(angle) / np.pi  # Normalize absolute angle to [0,1]

    # Quantize edge directions
    direction = np.full_like(angle, -1)
    # Vertical edges
    condition1 = (abs_theta >= 0.0) & (abs_theta < 0.05)
    condition2 = (abs_theta > 0.95) & (abs_theta <= 1.0)
    direction[np.where(condition1 | condition2)] = 0
    # Horizontal edges
    condition3 = (abs_theta > 0.45) & (abs_theta < 0.55)
    direction[np.where(condition3)] = 1
    # Diagonal edges
    condition4 = (abs_theta > 0.05) & (abs_theta < 0.45)
    condition5 = (abs_theta > 0.55) & (abs_theta < 0.95)
    sign_theta = np.sign(angle)
    direction[np.where(condition4 & (sign_theta > 0))] = 2  # Diagonal /
    direction[np.where(condition4 & (sign_theta < 0))] = 3  # Diagonal \
    direction[np.where(condition5 & (sign_theta > 0))] = 3  # Diagonal \
    direction[np.where(condition5 & (sign_theta < 0))] = 2  # Diagonal /

    # Edge ASCII characters based on direction
    edge_ascii_chars = ['|', '-', '/', '\\']  # Adjust or expand as needed

    # Luminance adjustment for fill ASCII
    exposure = 1.0  # Adjust exposure
    attenuation = 1.0  # Adjust attenuation
    luminance = np.power(normalized_frame * exposure, attenuation)
    luminance = np.clip(luminance, 0, 1)

    # Luminance ASCII characters from darkest to lightest
    luminance_ascii_chars = " .:-=+*#%@"  # Adjust or expand as needed
    num_lum_chars = len(luminance_ascii_chars)

    # Map luminance to ASCII characters
    lum_indices = (luminance * (num_lum_chars - 1)).astype(int)
    lum_ascii = np.array([luminance_ascii_chars[idx] for idx in lum_indices.flatten()]).reshape(rows, cols)

    # Create the final ASCII image
    ascii_image = []
    for i in range(rows):
        ascii_row = ""
        for j in range(cols):
            if edge_map[i, j] > 0:
                dir_value = direction[i, j]
                if dir_value >= 0 and dir_value <= 3:
                    ascii_char = edge_ascii_chars[int(dir_value)]
                else:
                    ascii_char = ' '  # Fallback for invalid directions
            else:
                ascii_char = lum_ascii[i, j]
            ascii_row += ascii_char
        ascii_image.append(ascii_row)
    return ascii_image

# DNS Resolver Class

class DoomResolver(BaseResolver):
    def __init__(self, game, domain):
        self.game = game
        self.domain = domain
        self.frame_id = 0  # Unique frame identifier
                            
    def resolve(self, request, handler):
        print("resolve method called", flush=True)
        try:
            reply = request.reply()
            qname = request.q.qname
            labels = qname.label  # Use the 'label' attribute
            labels = [label.decode('utf-8') for label in labels]

            # Adjust for trailing empty label if present
            if labels and labels[-1] == '':
                labels = labels[:-1]
            domain_labels = self.domain.split('.')
            if domain_labels and domain_labels[-1] == '':
                domain_labels = domain_labels[:-1]

            if len(labels) > len(domain_labels):
                command_labels = labels[:-(len(domain_labels))]
            else:
                command_labels = labels

            print(f"Labels: {labels}", flush=True)
            print(f"Command Labels: {command_labels}", flush=True)
            print(f"Domain Labels: {domain_labels}", flush=True)

            # Extract input_command
            if len(command_labels) >= 1:
                input_command = command_labels[0]
            else:
                input_command = 'idle'

            print(f"Input Command: {input_command}", flush=True)

            # Generate new frame
            self.game.make_action(map_input_to_action(input_command))
            if self.game.is_episode_finished():
                self.game.new_episode()
            state = self.game.get_state()
            if state is not None:
                frame = state.screen_buffer
                # Convert frame to ASCII
                ascii_lines = frame_to_ascii(frame)
                ascii_data = '\n'.join(ascii_lines)
                # Compress the ASCII data
                compressed_data = zlib.compress(ascii_data.encode('utf-8'), level=1)  # Use level=1 for faster compression
                encoded_data = base64.b64encode(compressed_data).decode('utf-8')
                # Split the encoded data into chunks of up to 255 bytes
                txt_chunks = [encoded_data[i:i+255] for i in range(0, len(encoded_data), 255)]
                # Send the chunks in a single TXT record
                reply.add_answer(RR(
                    rname=request.q.qname,
                    rtype=QTYPE.TXT,
                    rclass=1,
                    ttl=0,
                    rdata=TXT(txt_chunks)
                ))
                reply.add_answer(RR(
                    rname=request.q.qname,
                    rtype=QTYPE.TXT,
                    rclass=1,
                    ttl=0,
                    rdata=TXT(f"FRAME_ID:{self.frame_id}")
                ))
                self.frame_id += 1
            else:
                reply.add_answer(RR(
                    rname=request.q.qname,
                    rtype=QTYPE.TXT,
                    rclass=1,
                    ttl=0,
                    rdata=TXT("No frame available.")
                ))

            # Ensure reply has a NOERROR response code
            reply.header.rcode = RCODE.NOERROR
            return reply
        except Exception as e:
            print(f"Exception in resolve method: {e}", flush=True)
            traceback.print_exc()
            # Return an empty reply with SERVFAIL to prevent NoneType error
            reply = request.reply()
            reply.header.rcode = RCODE.SERVFAIL
            return reply

if __name__ == "__main__":
    # Initialize Doom game
    game = initialize_game()
    domain = "dnsroleplay.club"  # Replace with your domain
    # Start DNS server with TCP support
    resolver = DoomResolver(game, domain)
    logger = DNSLogger(prefix=False)
    server = DNSServer(resolver, port=9353, address="0.0.0.0", logger=logger, tcp=True)
    print("Starting DNS server on port 9353 (TCP and UDP)...")
    server.start_thread()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        game.close()
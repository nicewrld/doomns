import os
import numpy as np
from dnslib.server import DNSServer, BaseResolver, DNSLogger
from dnslib import RR, TXT, QTYPE, RCODE
import time
import zlib
import base64
import traceback
from numba import njit, prange

# Import ViZDoom
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, Button

# Import OpenCV
import cv2

import time
import functools


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[DEBUG] {func.__name__} took {elapsed:.4f} seconds")
        return result

    return wrapper


# Initialize DoomGame
def initialize_game():
    game = DoomGame()

    # Set the path to your original Doom WAD file
    game.set_doom_game_path("DOOM.WAD")  # Replace with the actual path

    game.set_window_visible(True)  # Set to True to display the game window
    game.set_mode(Mode.PLAYER)

    # Add all necessary buttons for full gameplay
    buttons = [
        Button.MOVE_FORWARD,  # Index 0
        Button.MOVE_BACKWARD,  # Index 1
        Button.TURN_LEFT,  # Index 2
        Button.TURN_RIGHT,  # Index 3
        Button.MOVE_LEFT,  # Index 4
        Button.MOVE_RIGHT,  # Index 5
        Button.ATTACK,  # Index 6
        Button.USE,  # Index 7
        Button.JUMP,  # Index 8
        Button.CROUCH,  # Index 9
        Button.SELECT_WEAPON1,  # Index 10
        Button.SELECT_WEAPON2,  # Index 11
        Button.SELECT_WEAPON3,  # Index 12
        Button.SELECT_WEAPON4,  # Index 13
        Button.SELECT_WEAPON5,  # Index 14
        Button.SELECT_WEAPON6,  # Index 15
        Button.SELECT_WEAPON7,  # Index 16
        Button.SELECT_WEAPON8,  # Index 17
        Button.RELOAD,  # Index 18
        Button.ZOOM,  # Index 19
        Button.SPEED,  # Index 20
    ]
    for button in buttons:
        game.add_available_button(button)

    # Set screen format and resolution
    game.set_screen_format(ScreenFormat.BGR24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)  # Increased resolution

    # Start at the first level
    game.set_doom_map("MAP01")  # For Doom II
    # For Ultimate Doom, use game.set_doom_map("E1M1")

    game.init()
    return game


# Map input commands to Doom actions
@timing_decorator
def map_input_to_action(input_command):
    # Initialize all buttons to 0
    action = [0] * 21  # Number of buttons added
    # Map the input command to the action
    command_map = {
        "w": 0,  # MOVE_FORWARD
        "s": 1,  # MOVE_BACKWARD
        "left": 2,  # TURN_LEFT
        "right": 3,  # TURN_RIGHT
        "a": 4,  # MOVE_LEFT
        "d": 5,  # MOVE_RIGHT
        "mouse1": 6,  # ATTACK
        "k": 6,  # ATTACK (alternative key)
        "use": 7,  # USE
        "u": 7,  # USE (alternative key)
        "space": 8,  # JUMP
        "ctrl": 9,  # CROUCH
        "c": 9,  # CROUCH (alternative key)
        "1": 10,  # SELECT_WEAPON1
        "2": 11,  # SELECT_WEAPON2
        "3": 12,  # SELECT_WEAPON3
        "4": 13,  # SELECT_WEAPON4
        "5": 14,  # SELECT_WEAPON5
        "6": 15,  # SELECT_WEAPON6
        "7": 16,  # SELECT_WEAPON7
        "8": 17,  # SELECT_WEAPON8
        "r": 18,  # RELOAD
        "zoom": 19,  # ZOOM
        "shift": 20,  # SPEED
    }
    if input_command in command_map:
        index = command_map[input_command]
        action[index] = 1
        return action
    else:
        return [0] * 21  # Return no action if command is 'idle' or unrecognized


# Improved frame to ASCII conversion
@timing_decorator
def frame_to_ascii(frame, cols=140, rows=60):
    # Resize the frame
    resized_frame = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)

    # Ensure the frame has three channels (RGB)
    if len(resized_frame.shape) == 2 or resized_frame.shape[2] == 1:
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)

    # Convert to float32 and normalize
    resized_frame = resized_frame.astype(np.float32) / 255.0

    # Split channels
    b_channel, g_channel, r_channel = cv2.split(resized_frame)

    # Compute luminance
    luminance = 0.2126 * r_channel + 0.7152 * g_channel + 0.0722 * b_channel

    # Apply Difference of Gaussians (DoG)
    sigma1 = 1.0
    sigma2 = sigma1 * 1.6
    blur1 = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma1)
    blur2 = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma2)
    dog = blur1 - blur2

    # Edge detection threshold
    threshold = 0.005
    edge_map = dog >= threshold

    # Compute gradient magnitude and angle using OpenCV functions
    # Convert edge_map to uint8 for Sobel function
    edge_map_uint8 = (edge_map * 255).astype(np.uint8)
    sobelx = cv2.Sobel(edge_map_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(edge_map_uint8, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely, angleInDegrees=False)

    # Determine edge directions
    direction = np.full_like(angle, 4)  # Default to space character index
    abs_theta = angle / np.pi
    # Direction mapping based on angle
    direction[(abs_theta < 0.05) | (abs_theta > 0.95)] = 0  # '|'
    direction[(abs_theta > 0.45) & (abs_theta < 0.55)] = 1  # '-'
    condition1 = (abs_theta > 0.05) & (abs_theta < 0.45)
    condition2 = (abs_theta > 0.55) & (abs_theta < 0.95)
    direction[condition1 & (sobelx * sobely > 0)] = 2  # '/'
    direction[condition1 & (sobelx * sobely < 0)] = 3  # '\\'
    direction[condition2 & (sobelx * sobely > 0)] = 3  # '\\'
    direction[condition2 & (sobelx * sobely < 0)] = 2  # '/'

    # Edge ASCII characters
    edge_ascii_chars = np.array(["|", "-", "/", "\\", " "])

    # Map luminance to ASCII characters
    luminance_ascii_chars = np.array(list(" .:-=+*#%@"))
    num_lum_chars = len(luminance_ascii_chars)
    lum_indices = np.clip(
        (luminance * (num_lum_chars - 1)).astype(int), 0, num_lum_chars - 1
    )
    lum_ascii = luminance_ascii_chars[lum_indices]

    # Prepare color codes using vectorized operations
    r_vals = (r_channel * 255).astype(np.uint8)
    g_vals = (g_channel * 255).astype(np.uint8)
    b_vals = (b_channel * 255).astype(np.uint8)
    color_codes = np.char.add(
        np.char.add(
            np.char.add("\033[38;2;", r_vals.astype(str)),
            np.char.add(";", g_vals.astype(str)),
        ),
        np.char.add(";", np.char.add(b_vals.astype(str), "m")),
    )

    # Build the ASCII image without loops
    ascii_chars = np.where(edge_map, edge_ascii_chars[direction.astype(int)], lum_ascii)
    colored_ascii = np.char.add(color_codes, ascii_chars)

    # Convert each row to a single string
    ascii_image = ["".join(row) + "\033[0m" for row in colored_ascii]

    return ascii_image


# DNS Resolver Class


class DoomResolver(BaseResolver):
    def __init__(self, game, domain):
        self.game = game
        self.domain = domain
        self.frame_id = 0  # Unique frame identifier

    @timing_decorator
    def resolve(self, request, handler):
        print("resolve method called", flush=True)
        try:
            reply = request.reply()
            qname = request.q.qname
            labels = qname.label  # Use the 'label' attribute
            labels = [label.decode("utf-8") for label in labels]

            # Adjust for trailing empty label if present
            if labels and labels[-1] == "":
                labels = labels[:-1]
            domain_labels = self.domain.split(".")
            if domain_labels and domain_labels[-1] == "":
                domain_labels = domain_labels[:-1]

            if len(labels) > len(domain_labels):
                command_labels = labels[: -(len(domain_labels))]
            else:
                command_labels = labels

            print(f"Labels: {labels}", flush=True)
            print(f"Command Labels: {command_labels}", flush=True)
            print(f"Domain Labels: {domain_labels}", flush=True)

            # Extract input_command
            if len(command_labels) >= 1:
                input_command = command_labels[0]
            else:
                input_command = "idle"

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
                ascii_data = "\n".join(ascii_lines)
                # Compress the ASCII data
                compressed_data = zlib.compress(
                    ascii_data.encode("utf-8"), level=1
                )  # Use level=1 for faster compression
                encoded_data = base64.b64encode(compressed_data).decode("utf-8")
                # Split the encoded data into chunks of up to 255 bytes
                txt_chunks = [
                    encoded_data[i : i + 255] for i in range(0, len(encoded_data), 255)
                ]
                # Send the chunks in a single TXT record
                reply.add_answer(
                    RR(
                        rname=request.q.qname,
                        rtype=QTYPE.TXT,
                        rclass=1,
                        ttl=0,
                        rdata=TXT(txt_chunks),
                    )
                )
                reply.add_answer(
                    RR(
                        rname=request.q.qname,
                        rtype=QTYPE.TXT,
                        rclass=1,
                        ttl=0,
                        rdata=TXT(f"FRAME_ID:{self.frame_id}"),
                    )
                )
                self.frame_id += 1
            else:
                reply.add_answer(
                    RR(
                        rname=request.q.qname,
                        rtype=QTYPE.TXT,
                        rclass=1,
                        ttl=0,
                        rdata=TXT("No frame available."),
                    )
                )

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

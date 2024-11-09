#!/bin/bash

# Configuration
SERVER_IP="127.0.0.1"                  # Replace with your DNS server IP
SERVER_PORT="9353"                     # Port where your DNS server is running
DOMAIN="dnsroleplay.club"              # Your domain

# Initialize variables
command="idle"                         # Default command when no input is detected

# Hide the cursor for a better visual experience
tput civis

# Save current terminal settings
OLD_STTY_SETTINGS=$(stty -g)
# Configure terminal for non-blocking input
stty -echo -icanon time 0 min 0

# Function to restore cursor visibility and terminal settings on exit
function on_exit {
    tput cnorm
    stty "$OLD_STTY_SETTINGS"
}
trap on_exit EXIT

while true; do
    # Read user input without blocking
    input=$(dd bs=1 count=1 2>/dev/null)
    if [ -n "$input" ]; then
        # Update command based on input
        case "$input" in
            w)      command="w" ;;
            a)      command="a" ;;
            s)      command="s" ;;
            d)      command="d" ;;
            j)      command="left" ;;
            l)      command="right" ;;
            k)      command="mouse1" ;;
            u)      command="use" ;;
            c)      command="ctrl" ;;
            ' ')    command="space" ;;
            r)      command="r" ;;
            1|2|3|4|5|6|7|8) command="$input" ;;
            *)      command="idle" ;;
        esac
    else
        # No input detected; set to 'idle'
        command="idle"
    fi

    # Send the DNS query using dig and capture the output
    output=$(dig +short +bufsize=4096 -p $SERVER_PORT TXT $command.$DOMAIN @$SERVER_IP)

    # Extract the TXT records and combine them
    ascii_art=$(echo "$output" | sed 's/^"//;s/"$//' | tr -d '\r')

    # Clear the terminal
    clear

    # Display the ASCII art
    echo -e "$ascii_art"

    # Sleep for a short time before the next iteration
done
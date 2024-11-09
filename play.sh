#!/bin/bash

# Configuration
SERVER_IP="127.0.0.1"                  # Replace with your DNS server IP
SERVER_PORT="9353"                     # Port where your DNS server is running
DOMAIN="dnsroleplay.club"              # Your domain
SLEEP_TIME="0.01"                      # Time between frames in seconds (adjust as needed)

# Initialize variables
command="idle"                         # Default command when no input is detected
ascii_art=""                           # Initialize ascii_art

# Hide the cursor for a better visual experience and switch to alternate screen buffer
tput civis
tput smcup

# Function to restore cursor visibility and terminal settings on exit
function on_exit {
    tput rmcup
    tput cnorm
    stty "$STTY_ORIG"
}
trap on_exit EXIT

# Save original terminal settings and set terminal to non-blocking mode
STTY_ORIG=$(stty -g)
stty -icanon -echo min 0 time 0

# Function to read user input without blocking
read_input() {
    # Attempt to read a single character without blocking
    if IFS= read -rsn1 input; then
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
            [1-8])  command="$input" ;;
            *)      command="idle" ;;  # Unrecognized input defaults to idle
        esac
    else
        # No input detected; set to 'idle'
        command="idle"
    fi
}

while true; do
    # Read user input without blocking
    read_input

    # Build the query name
    qname="$command.$DOMAIN"

    # Make the DNS query
    output=$(dig +tcp +short -t TXT +bufsize=65535 -p "$SERVER_PORT" "$qname" @"$SERVER_IP")

    # Parse the output
    ascii_data=""
    frame_id=0
    while IFS= read -r line; do
        # Remove leading and trailing quotes
        # Also, split the line into multiple quoted strings
        IFS='"' read -ra parts <<< "$line"
        for part in "${parts[@]}"; do
            # Skip empty strings and commas
            if [[ -z "$part" || "$part" == " " || "$part" == "," ]]; then
                continue
            fi
            if [[ "$part" == FRAME_ID:* ]]; then
                frame_id=${part#FRAME_ID:}
            else
                ascii_data+="$part"
            fi
        done
    done <<< "$output"

    # Decode and decompress the ASCII data
    ascii_art_new=$(echo "$ascii_data" | base64 -D 2>/dev/null | python3 -c "import sys, zlib; print(zlib.decompress(sys.stdin.buffer.read()).decode('utf-8'))" 2>/dev/null)

    # Check if ascii_art_new is not empty
    if [ -n "$ascii_art_new" ]; then
        ascii_art="$ascii_art_new"
    fi

    # Clear the terminal and display the ASCII art
    clear
    if [ -n "$ascii_art" ]; then
        echo -e "$ascii_art"
    else
        echo "Failed to retrieve frame."
    fi

    # Sleep for a short time before the next iteration
    sleep "$SLEEP_TIME"
done
#!/bin/bash

# Configuration
SERVER_IP="127.0.0.1"                  # Replace with your DNS server IP
SERVER_PORT="9353"                     # Port where your DNS server is running
DOMAIN="dnsroleplay.club"              # Your domain
SLEEP_TIME="0.02"                       # Time between frames (adjust as needed)

# Initialize variables
command="idle"                         # Default command when no input is detected

# Hide the cursor for a better visual experience
tput civis

# Function to restore cursor visibility on exit
function on_exit {
    tput cnorm
}
trap on_exit EXIT

# Function to read user input without blocking
read_input() {
    # Use 'read' with a timeout of 0 for non-blocking behavior
    if read -rsn1 -t 1 input; then
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
}

while true; do
    # Read user input without blocking
    read_input

    # Initialize frame assembly variables
    ascii_data=""
    part=0
    total_parts=1
    frame_id=0

    while [ $part -lt $total_parts ]; do
    if [ $part -eq 0 ]; then
        qname="$command.$DOMAIN"
    else
        qname="$command.$((part)).$frame_id.$DOMAIN"
    fi

    # Adjust 'part' to start from 1 for the server
    adjusted_part=$((part + 1))

    if [ $part -eq 0 ]; then
        qname="$command.$DOMAIN"
    else
        qname="$command.$adjusted_part.$frame_id.$DOMAIN"
    fi

    # Rest of the code remains the same...

        output=$(dig +tcp +short +bufsize=65535 -p $SERVER_PORT TXT $qname @$SERVER_IP)

        while IFS= read -r line; do
            line=$(echo "$line" | sed 's/^"//;s/"$//')
            if [[ "$line" == FRAME_ID:* ]]; then
                frame_id=${line#FRAME_ID:}
            elif [[ "$line" == TOTAL_PARTS:* ]]; then
                total_parts=${line#TOTAL_PARTS:}
            else
                ascii_data+="$line"
            fi
        done <<< "$output"

        part=$((part + 1))
    done

    # Decode and decompress the ASCII data
    ascii_art=$(echo "$ascii_data" | base64 --decode | python3 -c "import sys, zlib; print(zlib.decompress(sys.stdin.buffer.read()).decode('utf-8'))")

    # Clear the terminal
    clear

    # Display the ASCII art
    echo -e "$ascii_art"

    # Sleep for a short time before the next iteration
    sleep $SLEEP_TIME
done
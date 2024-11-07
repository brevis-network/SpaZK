import sys
import re

text = sys.stdin.read()

# Find all occurrences of \d+s or \d+ms
times = re.findall(r'\b\d+\.?\d*(?:s|ms|µs|ns)\b', text)

# Print results in table format with 15 records per line
for i in range(0, len(times), 15):
    print(','.join(times[i:i+15]))
import sys

if len(sys.argv) < 2:
    print("Usage: python convert_indent.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

with open(filename, "r") as f:
    lines = f.readlines()

with open(filename, "w") as f:
    for line in lines:
        leading_spaces = len(line) - len(line.lstrip(' '))
        # convert every 4 spaces to 2 spaces
        new_leading = ' ' * (leading_spaces // 2)
        f.write(new_leading + line.lstrip(' '))

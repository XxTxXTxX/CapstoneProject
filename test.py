import math

def parse_pdb(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                print(list(line.split()))
                x = float(list(line.split())[6])  
                y = float(list(line.split())[7]) 
                z = float(list(line.split())[8])
                coordinates.append((x, y, z))
    return coordinates

def calculate_distance(coord1, coord2):
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) / 3
    return distance

def compare_pdb_files(file1, file2):
    coords_file1 = parse_pdb(file1)
    coords_file2 = parse_pdb(file2)

    differences = []
    for i, (coord1, coord2) in enumerate(zip(coords_file1, coords_file2)):
        distance = calculate_distance(coord1, coord2)
        differences.append((i + 1, distance)) 

    return differences

file1 = "origin.pdb"
file2 = "output.pdb" 

differences = compare_pdb_files(file1, file2)
print("Line\tcoordinate average difference")
for line_number, distance in differences:
    print(f"{line_number}\t{distance:.4f}")
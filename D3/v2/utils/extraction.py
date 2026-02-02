import json


file_path = '/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/force_displacement_results.json'

with open(file_path, 'r') as f:
    data = json.load(f)


displacements = [x[1] for x in data]

for d in displacements:
    print(d)





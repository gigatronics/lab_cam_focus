import json
import csv

dir='/Users/geewiz/python/lab_cam_focus/'
#filename='data.json'
filename2='data.csv'

angles = []

with open(dir+filename2,'r')as f:
#rects=json.loads(f)
#    print(dir+filename2)
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
#    rows = [header] + [[row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])] for row in reader]
    for row in reader:
        angle = row[0]
        angles.append(angle)
    print(angle)



#forrectinrects:
#print(rects[0])



#rects=json.load(json_data)
#forrectinrects:
#print(rects)

import csv

with open('D:\MajorProject\extracted_features_set_shuffled.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    #print(stripped)
    lines = (line.split(" ") for line in stripped if line)
    #print(lines[0])
    #print(lines[0])
    with open('D:\MajorProject\knn_features.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('contrast', 'dissimilarity','homogeneity','energy','correlation','label'))
        writer.writerows(lines)
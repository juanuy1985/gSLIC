from scipy.misc import imsave
from skimage import io, color
from ClusterCPU import Cluster

import math
import numpy as np
import sys
import time

def distance(lab, x1, y1, x2, y2, m, S):
    dlab = np.linalg.norm(lab[x1][y1] - lab[x2][y2])
    dxy = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return dlab + m / S * dxy

def isInCluster(labels, x, y, indexCluster, width, height):
    x0 = -1 if x > 0 else 0
    x1 = 2 if x < width-1 else 1
    y0 = -1 if y > 0 else 0
    y1 = 2 if y < height-1 else 1
    for i in range(x0, x1):
        for j in range(y0, y1):
            if labels[x+i][y+j] == indexCluster:
                return True
    return False

def enforceConnectivity(centers, labels, width, height):
    newLabels = np.zeros((width, height), dtype=int) - 1
    for index in range(len(centers)):
        center = centers[index]
        x, y = center.x, center.y
        newLabels[x][y] = index
        side = 0
        connectivity = True
        while connectivity:
            connectivity = False
            x += 1
            y += 1
            side += 2
            for cont in range(side):
                y -= 1
                if x >= width-1 or y >= height-1:
                    break
                if y > 0 and newLabels[x][y] == -1 and labels[x][y] == index and isInCluster(newLabels, x, y, index, width, height):
                    newLabels[x][y] = index
                    connectivity = True

            for cont in range(side):
                x -= 1
                if x >= width-1 or y >= height-1:
                    break
                if x > 0 and newLabels[x][y] == -1 and labels[x][y] == index and isInCluster(newLabels, x, y, index, width, height):
                    newLabels[x][y] = index
                    connectivity = True

            for cont in range(side):
                y += 1
                if x >= width-1 or y >= height-1:
                    break
                if y < height-1 and newLabels[x][y] == -1 and labels[x][y] == index and isInCluster(newLabels, x, y, index, width, height):
                    newLabels[x][y] = index
                    connectivity = True

            for cont in range(side):
                x += 1
                if x >= width-1 or y >= height-1:
                    break
                if x < width-1 and newLabels[x][y] == -1 and labels[x][y] == index and isInCluster(newLabels, x, y, index, width, height):
                    newLabels[x][y] = index
                    connectivity = True

    for i in range(1, width):
        for j in range(1, height):
            if newLabels[i][j] == -1:
                newLabels[i][j] = newLabels[i-1][j]
    return newLabels
        
def SLIC(filename, m, S, threshold):
    rgb = io.imread(filename)
    #TODO: Convert to lab using CUDA
    lab = color.rgb2lab(rgb)

    start = time.time()
    
    width, height = len(lab), len(lab[0])
    distances = np.full((width, height), np.inf)
    labels = np.zeros((width, height), dtype=int)
    E = np.inf

    # 1. Initialize cluster centers
    centers = [Cluster(int(x*S+S/2),int(y*S+S/2)) for x in range(int(width/S)) for y in range(int(height/S))]

    while E > threshold:
        # 3. Assign the best matching pixels from a 2S × 2S square neighborhood around the cluster center
        for index in range(len(centers)):
            center = centers[index]
            x0 = center.x - S if center.x >= S else 0
            y0 = center.y - S if center.y >= S else 0
            x1 = center.x + S if width >= center.x + S else width
            y1 = center.y + S if height >= center.y + S else height
            center.xTmp = center.yTmp = center.num = 0

            for i in range(x0, x1):
                for j in range(y0, y1):
                    dist = distance(lab, i, j, center.x, center.y, m, S)
                    if distances[i][j] > dist:
                        distances[i][j] = dist
                        labels[i][j] = int(index)

        #4. Compute new cluster centers and residual error E
        for i in range(width):
            for j in range(height):
                center = centers[labels[i][j]]
                center.xTmp += i
                center.yTmp += j
                center.num += 1

        E = 0
        for index in range(len(centers)):
            center = centers[index]
            newCenterX = int(center.xTmp / center.num)
            newCenterY = int(center.yTmp / center.num)
            E += abs(center.x - newCenterX) + abs(center.y - newCenterY)
            center.x = newCenterX
            center.y = newCenterY

        E = E/len(centers)

        print("Error es:", E)

    end = time.time()
    print("CPU time: \t" , end - start, "\t", distances.shape)

    labels = enforceConnectivity(centers, labels, width, height)
    
    newLab = np.copy(lab)
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if  labels[i][j] != labels[i-1][j] or labels[i][j] != labels[i+1][j] or labels[i][j] != labels[i][j-1] or labels[i][j] != labels[i][j+1] :
                newLab[i][j] = [0, 0, 0]
    
    newRGB = color.lab2rgb(newLab)
    imsave('processed.jpg', newRGB)

def main(argv):
	if len(argv) > 4:
		path = argv[1]
		m = int(argv[2])
		S = int(argv[3])
		threshold = float(argv[4])
		SLIC(argv[1], m, S, threshold)

if __name__ == "__main__":
    main(sys.argv)

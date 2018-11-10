from scipy.misc import imsave
from skimage import io, color
from Cluster import Cluster
from numba import cuda
import math
import numpy as np
import sys
import os
import PIL
import sys
import time
from os.path import join


@cuda.jit(device = True)
def getDistance(lab, x1, y1, x2, y2, m, S):
	lab1 = lab[x1][y1]
	lab2 = lab[x2][y2]
	dlab = (math.pow(lab1[0] - lab2[0], 2) + math.pow(lab1[1] - lab2[1], 2) + math.pow(lab1[2] - lab2[2], 2)) 
	dxy = (math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
	return dlab + m / S * dxy

@cuda.jit
def kernelClustering(lab, centers, labels, m, S, w, h):
	x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	distance = np.inf
	indexCenter = -1
	if x < w and y < h:
		for index in range(len(centers)):
			center = centers[index]
			if abs(x - center[0]) <= S and abs(y - center[1]) <= S:
				tmpDistance = getDistance(lab, x, y, center[0], center[1], m, S)
				if tmpDistance < distance:
					indexCenter = index
					distance = tmpDistance
		labels[x][y] = indexCenter

@cuda.jit
def kernelRecalculateCenters(centers, labels, S, w, h):
	index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if index < len(centers):
		center = centers[index]
		x = center[0]
		y = center[1]
		x0 = 0 if x - S < 0 else x - S
		x1 = w if x + S > w else x + S
		y0 = 0 if y - S < 0 else y - S
		y1 = h if y + S > h else y + S
		center[2] = center[3] = center[4] = 0		
		for i in range(x0, x1):
			for j in range(y0, y1):
				if labels[i][j] == index:
					center[2] += i
					center[3] += j
					center[4] += 1
		center[2] /= center[4] 
		center[3] /= center[4] 
		center[5] = abs(center[0]-center[2]) + abs(center[1]-center[3])
		center[0] = center[2]
		center[1] = center[3]
		 

def SLIC_CUDA(rgb, lab, m, S, threshold, start):
   
	width, height = len(lab), len(lab[0])
	distances = np.full((width, height), np.inf)
	labels = np.zeros((width, height), dtype=int)
	E = np.inf

	# 1. Initialize cluster centers
	C =[Cluster(int(x*S+S/2),int(y*S+S/2)) for x in range(int(width/S)) for y in range(int(height/S))]
	C = np.asarray([[c.x, c.y, c.xTmp, c.yTmp, c.num, c.E] for c in C])
	labD = cuda.to_device(lab)

	blocksInX = math.ceil(lab.shape[0] / 32.0)
	blocksInY = math.ceil(lab.shape[1] / 32.0)
	blocksPerGrid = (blocksInX, blocksInY)
	threadsPerBlock = (32, 32)

	blocks = (math.ceil(len(C) / 1024.0), 1)
	threads = (1024 ,1)

	while E > threshold:
        	# 3. Assign the best matching pixels from a 2S × 2S neighborhood around the cluster
		CD = cuda.to_device(C)
		labelsD = cuda.to_device(labels)	
		kernelClustering[blocksPerGrid,threadsPerBlock](labD, CD, labelsD, m, S, width, height)
		kernelRecalculateCenters[blocks, threads](CD, labelsD, S, width, height)
		C = CD.copy_to_host()
		E = 0
		for c in C:
			E = E + c[5]
			c[5] = 0
		E = E / len(C)
		print("Error es:", E)
	
	end = time.time()
	print("CUDA\t" , end - start, "\t", distances.shape)

	labels = labelsD.copy_to_host()
	labels = enforceConnectivity([Cluster(c[0], c[1]) for c in C], labels, width, height)
	newLab = lab	
	for i in range(1, width - 1):
		for j in range(1, height - 1):
			if  labels[i][j] != labels[i-1][j] or labels[i][j] != labels[i+1][j] or labels[i][j] != labels[i][j-1] or labels[i][j] != labels[i][j+1] :
				newLab[i][j] = [0, 0, 0]
    
	return newLab

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

def getImages(path):
	return [join(path, x) for x in os.listdir(path)]

def main(argv):
	if len(argv) > 4:
		path = argv[1]
		m = int(argv[2])
		S = int(argv[3])
		threshold = float(argv[4])
		rgb = io.imread(path)
		lab = color.rgb2lab(rgb)
		
		start = time.time()
		result = SLIC_CUDA(rgb, lab, m, S, threshold, start)
		newRGB = color.lab2rgb(result)
		imsave('processed_CUDA.jpg', newRGB)

def main1(argv):
	print(argv[1])
	paths = getImages(argv[1])
	cont = 1
	for path in paths:
		rgb = io.imread(path)
		lab = color.rgb2lab(rgb)
		# ---------------------------------------------------------
		start = time.time()
		result = SLIC_CUDA(rgb, lab, 20, 100, 5, start)
		newRGB = color.lab2rgb(result)
		imsave('Pruebas/Salida/' + "{0}.jpg".format(cont) , newRGB)

		cont = cont + 1

if __name__ == "__main__":
    main(sys.argv)

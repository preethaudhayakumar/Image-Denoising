import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
from scipy.stats import multivariate_normal
from scipy.special import expit as sigmoid
import tqdm

def read_data(filename, is_RGB, visualize=False, save=False, save_name=None):
# read the text data file
#   data, image = read_data(filename, is_RGB) read the data file named 
#   filename. Return the data matrix with same shape as data in the file. 
#   If is_RGB is False, the data will be regarded as Lab and convert to  
#   RGB format to visualise and save.
#
#   data, image = read_data(filename, is_RGB, visualize)  
#   If visualize is True, the data will be shown. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save)  
#   If save is True, the image will be saved in an jpg image with same name
#   as the text filename. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save, save_name)  
#   The image filename.
#
#   Example: data, image = read_data("1_noise.txt", True)
#   Example: data, image = read_data("cow.txt", False, True, True, "segmented_cow.jpg")

	with open(filename, "r") as f:
		lines = f.readlines()

	data = []

	for line in lines:
		data.append(list(map(float, line.split(" "))))
		#print(line,data)

	data = np.asarray(data).astype(np.float32)

	N, D = data.shape

	cols = int(data[-1, 0] + 1)
	rows = int(data[-1, 1] + 1)
	channels = D - 2
	img_data = data[:, 2:]
	#print (img_data)
	#print(cols,rows)
	#print(N,D)

	# In numpy, transforming 1d array to 2d is in row-major order, which is different from the way image data is organized.
	image = np.reshape(img_data, [cols, rows, channels]).transpose((1, 0, 2))

	if visualize:
                if channels == 1:
                        # for visualizing grayscale image
                        cv2.imshow("", image)
                else:
                        # for visualizing RGB image
                        cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_Lab2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

	if save:
		if save_name is None:
			save_name = filename[:-4] + ".jpg"
		assert save_name.endswith(".jpg") or save_name.endswith(".png"), "Please specify the file type in suffix in 'save_name'!"

		if channels == 1:
			# for saving grayscale image
			cv2.imwrite(save_name, image)
		else:
			# for saving RGB image
			cv2.imwrite(save_name, (cv2.cvtColor(image, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))

	return data, image

def write_data(data, filename):
# write the matrix into a text file
#   write_data(data, filename) write 2d matrix data into a text file named
#   filename.
#
#   Example: write_data(data, "cow.txt")

	lines = []
	for i in range(data.shape[0]):
                for j in range( data.shape[1]):
                        lines.append(" ".join([str(i), str(j)] + ["%.6f" % data[i, j]]) + "\n")

	with open(filename, "w") as f:
		f.writelines(lines)
	
def find_neighbours(x,y,mu,M,N):
        neighbours=np.empty((0))
        if y!=0:
                neighbours=np.append(neighbours,mu[x,y-1])
        if x!=0:
                neighbours=np.append(neighbours,mu[x-1,y])
        if y!=N-1:
                neighbours=np.append(neighbours,mu[x,y+1])
        if x!=M-1:
                neighbours=np.append(neighbours,mu[x+1,y])
        return neighbours
   
def main():
        data,image=read_data('a1/1_noise.txt',True,True)
        n=np.empty([image.shape[1],image.shape[0]],dtype=int)
        print(data.shape[0])
        for i in range(data.shape[0]):
                n[int(data[i][0])][int(data[i][1])]=int(data[i][2])
        img_binary = +1*(n>1) + -1*(n<1)
        print(img_binary)

        W = 1 
        mean=1
        sigma=2
        [M, N] = img_binary.shape

        Lpos = multivariate_normal.logpdf(img_binary.flatten(), +mean, cov=sigma**2)
        Lneg = multivariate_normal.logpdf(img_binary.flatten(), -mean, cov=sigma**2) 
        L=Lpos - Lneg
        L = np.reshape(L, (M, N))
        
        mu=np.empty([M,N])
        munew=mu.copy()
        diff=0
        count=0
        
        while((((np.absolute((np.sum(np.absolute(mu-munew)>0.88)/(M*N))-diff))>=0.001) and ((np.sum(np.absolute(mu-munew)>0.88)/(M*N))!=0))or count<3):
                diff=(np.sum(np.absolute(mu-munew)>0.9)/(M*N))
                mu=munew.copy()
                
                for x in range(M):
                        for y in range(N):
                                neighbours=find_neighbours(x,y,mu,M,N)
                                munew[x,y]=np.tanh(W*np.sum(neighbours)+0.5*L[x,y])
                count+=1
        
        mu[mu >= 0] = 255
        mu[mu < 0] = 0        
        write_data(mu,"a1/1_denoise.txt")
        read_data("a1/1_denoise.txt",False,True,True,"a1/1_denoise.jpg")

main()

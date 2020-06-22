import sys
import random
import numpy as np
from functions.io_data import read_data
from functions.io_data import write_data


class gibbs:

    def __init__(self, image, l1_norm, beta):
        self.image = image
        self.row = image.shape[0]
        self.column = image.shape[1]
        self.l1_norm =  l1_norm
        self.beta = beta

    def neighbours(self, i, j):

#checking for the neighbours (includes the boundary cases of first row,column and last row,column
        list = []
        if i == 0:
            list.append((self.row-1, j))
        else:
            list.append((i-1, j))

        if i == self.row-1:
            list.append((0, j))
        else:
            list.append((i+1, j))

        if j == 0:
            list.append((i, self.column-1))
        else:
            list.append((i, j-1))

        if j == self.column-1:
            list.append((i, 0))
        else:
            list.append((i, j+1))

        return list

    def energy(self, i, j):
        return self.l1_norm[i,j] + sum(self.image[ii,jj] for (ii, jj) in self.neighbours(i, j)) 

    def gibbs_sample(self, i, j):
        pixel_p = 1 / (1 + np.exp(-2 * self.beta * self.energy(i,j))) 

        if random.uniform(0, 1) <= pixel_p: 
            self.image[i, j] = 1
        else:
            self.image[i, j] = -1


#-------------------------------------Main function-------------------------------------
def main(filename):

        data, image = read_data(filename, True)      
        image = image.transpose(1,0,2)
#Converting gray to binary (-1,1)

        image[image == 0] = -1
        image[image == 255] = 1
      
        burn_in_itr = 10
        itr = 30 
        q = 0.73

        l1_norm = 0.5 * np.log(q / (1-q))
        denoise_sub = gibbs(image, l1_norm*image, 3)
        
        avg = np.zeros_like(image).astype(np.float32)

        for i in range(burn_in_itr + itr):
            print("Iteration - " + str(i))
            for j in range(image.shape[0]):
                for k in range(image.shape[1]):
                    if(random.uniform(0, 1) <= 0.73):
                        denoise_sub.gibbs_sample(j, k)
            if(i > burn_in_itr):
                avg += denoise_sub.image

        avg / itr

#converting back to gray scale
        avg[avg >= 0] = 255
        avg[avg < 0] = 0

        row = avg.shape[0]
        column = avg.shape[1]
        cnt = 0
        print(avg.shape)
        for i in range(0,row):
            for j in range(0,column):
                data[cnt][2] = avg[i][j][0]
                cnt = cnt + 1

        write_data(data, filename+"_denoise.txt")
        read_data(filename+"_denoise.txt", True, save=True, save_name=filename+"_denoise.jpg")
        print("Iterations completed - Please check your folder for output -" + filename+"_denoise.jpg")

#-------------------------------------End of Main function-------------------------------------

if __name__ == "__main__":
    if len(sys.argv)!=2:
        print("Usage: python3 Gibbs.py <filename.txt>")
        sys.exit(-1)
    main(sys.argv[1])

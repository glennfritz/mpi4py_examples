from skimage import data, img_as_float
import skimage.io
import os.path
import time
from mpi4py import MPI

from skimage.restoration import denoise_bilateral

curPath = os.path.abspath(os.path.curdir)
noisyDir = os.path.join(curPath,'noisy')
denoisedDir = os.path.join(curPath,'denoised')


def loop(imgFiles):
    for f in imgFiles:
        img = img_as_float(data.load(os.path.join(noisyDir,f)))
        startTime = time.time()
        img = denoise_bilateral(img, sigma_color=0.1, sigma_spatial=3, multichannel=True)
        skimage.io.imsave(os.path.join(denoisedDir,f),img)
        print("Took %f seconds for %s" %(time.time() - startTime, f))

def serial():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    numImages = 100
    numImagesPerRank = numImages/size
    total_start_time = time.time()
    imgFiles = ["%.3d.jpeg"%x for x in range(rank * numImagesPerRank+1,(rank + 1) * numImagesPerRank)]
    loop(imgFiles)

    if rank == 0:
        print("Total time %f seconds" % (time.time() - total_start_time))

if __name__=='__main__':
    serial()
# Segmenting Homogeneous Regions in Images using Variance Wells 

This repository contains a 2D vessel segmentation algorithm developed for analyzing arteries from TOF-MRA brain scans. This version of the algorithm works on 2D slices only. Our current work is focused on applying it on 3D images.

## Description

This code is written as a supplement to the CMU RI Technical Report 24-33 *Segmenting Homogeneous Regions in Images using Variance Wells.* by Bhargava et al. Readers seeking a more in-depth understanding of the theory behind this code are encouraged to refer to the paper found [here](https://www.ri.cmu.edu/publications/segmenting-homogeneous-regions-in-images-using-variance-wells/). This demonstration version of the code is condensed into one script for ease of use and distribution. Ongoing projects may choose to separate functions and classes based on personal preference or as needed. 

## Getting Started

### Dependencies

* Python v3.12 was last used to test this code
* All other libraries used are common Python libraries that should be easily accesible.

### Installing

* Please download the main script *vWellAlgorithm_2D.py* and image *X216DoubleCroppedSlice.nii* to the same working directory
* By default, the program for look for the image in its own directory.
* However, this can be changed by replacing the path in the following lines (around 552 - 554)
```
# Load the image in as a NumPy array
img = nib.load("C:\\Users\\name\\Documents\\PycharmProjects\\etc")
loaded_image = img.get_fdata()
```
### Executing program

* Run the main script *vWellAlgorithm_2D.py* from a command window/terminal or through any IDE

### Alterable Settings

There are a few settings you can change to alter the behavior of the algorithm. All of them are found around lines 544 - 547 here:
```
if __name__ == "__main__":
    # Settings
    kernel_radius = 1
    resampling_factor = 2
    heuristic_weight = 0.4
```
#### Description of Each Setting
* **kernel_radius**: Affects the window size of the the kernel that iterates through the image to calculate the variance at each pixel. For example, a kernel_radius of 1 means a 3x3 kernel iterates through the input image, calculating the variance for 9 pixels at a time. A kernel_radius of 2 would mean a 5x5 kernel does the same, calculating variance for 25 pixels at a time.
* **resampling_factor**: Refers to the factor by which the input image is linearly interpolated. By default, it is set to *2* to achieve higher resolution from the input image. This also allows for the detection of single-pixel wide vessels, for which the theory is detailed in the technical report.
* **heuristic_weight**: Is an multiplier used to guide the Dijkstra's shortest path algorithm, similar to how the A* algorithm works. Our implementation of this *heuristic weight* has a range of 0 to infinity, where a weight of 0 means the direction of the selected points played no role in the determination of which node to go to, while an infinitely high heurisitc weight would be the t-test between neighboring vWells would no longer contribute to the Dijkstra's pathfinding.

## Authors

#### Code written by: Satyaj Bhargava
#### Contributing authors: John Lorence, Ben Cohen, Minjie Wu, Howard Aizenstein, & George Stetten

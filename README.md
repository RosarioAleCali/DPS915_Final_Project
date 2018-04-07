# DPS915 Final Project

## Authors:
* <a href="https://github.com/RosarioAleCali" target="_blank">Rosario A. Cali</a>
* <a href="https://github.com/mfainshtein4" target="_blank">Max Fainshtein</a>
* <a href="https://github.com/jpildush" target="_blank">Joseph Pildush</a>
* <a href="https://github.com/Mithilan16" target="_blank">Mithilan Sivanesan</a>

## Introduction
This is the home repository for our final project in the <a href="https://scs.senecac.on.ca/~gpu610/" target="_blank">DPS915 course</a> (Introduction to Parallel Programming).<br>
This readme file will primarily give a short description to each sub-project and instructions on how to successfully compile and run each project.
For more details about the project please consult our <a href="https://wiki.cdot.senecacollege.ca/wiki/Sirius" target="_blank">wiki</a>, source codes, test runs, and presentation.<br>
Lastly, this project was divided into three assignments, each of them having different goals.

## Assignment 1
In Assignment 1, each team member took a problem and implemented a serial solution for it. After each implementation was finished, the solution was then profiled to check what were the execution times and how we could improve our solutions using CUDA parallel programming. Only one solution will be parallelized and optimized by the team.

### Algorithms
<a href="https://github.com/jpildush" target="_blank">Joseph Pildush</a> decided to implement some algorithms such as std::sort, saxpy, and Prefix Sum.<br>
To compile his project, simply use the following command:
 ```
g++ serialAlgorithms.cpp -std=c++11 -o serialAlgorithms
 ```
 To run his project, simply use the following command:
 ```
./serialAlgorithms 100000000
 ```
The test results were quite interesting. Using 100000000 elements resulted in about 13420 milliseconds for execution time. Today, that is a relatively high number, but other projects have turned out to be a lot more computer intensive.

### Box Filter
<a href="https://github.com/mfainshtein4" target="_blank">Max Fainshtein</a> decided to implement a Box Filter algorithm which uses OpenCV to open images but the actual implementation of the algorithm is done manually.
To compile and run his project you will need to create a project within Visual Studio 2015. Once inside VS 2015, simply use the nuget package manager to install OpenCV 3.0. You will need to install the opencv.win.native.redist package provided by Harry Y.<br>
The test results here were quite interesting because it took about 305 seconds to process a 4k image.

### Data Compression
<a href="https://github.com/Mithilan16" target="_blank">Mithilan Sivanesan</a> decided to implement the Lempel–Ziv–Welch (LZW) Algorithm to compress and decompress large text files.
To compile his project, simply use the following command:
 ```
gcc lzw.c -o lzw
 ```
 To run his project, simply use the following command:
 ```
./lzw myText.txt
 ```
Testing this implementation on a 5 MB file resulted in about 3 seconds for execution time.

 ### Vehicle Detection
<a href="https://github.com/RosarioAleCali" target="_blank">Rosario A. Cali</a> decided to implement a Vehicle Detection program. He used FFMPEG to decode a video and Dlib to analyze each frame.
To compile his project, simply use the following command:
```
g++ -std=c++11 -O3 -I <path>/dlib-19.9/  <path>/dlib-19.9/dlib/all/source.cpp -lpthread -lX11 vehicleDetection.cpp -lpng -DDLIB_PNG_SUPPORT -o vehicleDetector
```
To run his project, simply use the following command:
```
./vehicleDetector <path-to-video>
```
To make sure the program compiles and execute successfully, you need to ensure you have installed the following libraries:
* FFmpeg
* Dlib
* libpng-dev

The execution of this program on a 10 seconds video took about 21 minutes. No, this is not a joke.

## Assignment 2
Afterwards, we had to decide which project to convert to CUDA. Surprisingly enough, we decided to move one with the Box Filter as supposed to the Vehicle Detection program. These, are some of the motivations that pushed us to make that decision:
* More chances to succeed
* Easier to Implement
* Great Improvements

Compiling and executing this project is very simple and similar to Assignment 1. In addition, you just need to have the NVIDIA CUDA Toolkit installed and create a CUDA project instead of a regular C++ empty project.<br>
Just by creating a simple kernel, we have achieved great improvements like processing a 4K image in just 3 seconds.

## Assignment 3
In this iteration, we took what we already had, and we enhanced the performance by increasing the kernel concurrency and reducing processing times to about 1.5 seconds.
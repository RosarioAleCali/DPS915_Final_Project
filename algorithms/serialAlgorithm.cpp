//Serial Algorithms - Assignment 1
//serialAlgorithm.cpp
//The first command argument will be used to maintain the size of the vector

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace std;
using namespace std::chrono;

//Populate array with random numbers
void setRandArr(vector<int>& array,int size) {
	srand(time(NULL));
	for (int i = 0;i < size;i++)
		array.push_back(rand() % 10000 + 1);
}

//Method used to calculate timing
void printTiming(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<milliseconds>(span);
	std::cout << msg << " - took - " << ms.count() << " millisecs" << std::endl;
}

//std::sort Algorithm
void stdSort(vector<int>& array,int arrSize,steady_clock::time_point ts,steady_clock::time_point te){
	cout << "--==Execution Time of std::sort Alogirthm==--" << endl;
	/*std::sort Algorithm*/
	//Time the fill of 1 vector
	ts = steady_clock::now();
	//Fill array with random numbers
	setRandArr(array, arrSize);
	te = steady_clock::now();
	printTiming("std::sort Vector (1) Initialize", te - ts);
	//Start timing of std::sort
	ts = steady_clock::now();
	//Use std::sort to sort vector array1
	sort(array.begin(),array.end());
	//End timing std::sort
	te = steady_clock::now();
	//Print Results
	printTiming("std::sort algorithm", te - ts);
}

//saxpy Algorithm
void saxpyAlg(int arrSize,steady_clock::time_point ts,steady_clock::time_point te){
	cout << endl << "--==Execution Time of saxpy Alogirthm==--" << endl;
	/*saxpy Algorithm*/
	vector<int> saxpyX,saxpyY;
	int saxpyA = 15;
	//Time the fill of 2 vectors
	ts = steady_clock::now();
	setRandArr(saxpyX, arrSize);
	setRandArr(saxpyY, arrSize);
	te = steady_clock::now();
	printTiming("saxpy Vectors (2) Initialize", te - ts);
	//Start timing of saxpy
	ts = steady_clock::now();
	for (int i = 0;i < arrSize;++i)
		saxpyY[i] = saxpyA*saxpyX[i] + saxpyY[i];
	//End timing of saxpy
	te = steady_clock::now();
	printTiming("saxpy Algorithm", te - ts);
}

//Prefix Sum Algorithm
void prefixSum(vector<int>& array,int arrSize,steady_clock::time_point ts,steady_clock::time_point te){
	cout << endl << "--==Execution Time of Prefix-Sum Alogirthm==--" << endl;
	/*Prefix-Sum Algorithm*/
	vector<int> psSum;
	array.clear();
	//Time the fill of 1 vector
	ts = steady_clock::now();
	//Fill array with random numbers
	setRandArr(array, arrSize);
	te = steady_clock::now();
	printTiming("Prefix-Sum Vector (1) Initialize", te - ts);
	//Start timing of Prefix-Sum
	ts = steady_clock::now();
	psSum.push_back(array[0]);
	for (int i = 1;i < arrSize;++i)
		psSum.push_back(psSum[i - 1] + array[i]);
	//End timing of Prefix-Sum
	te = steady_clock::now();
	printTiming("Prefix-Sum Algorithm", te - ts);
}

//Main program will execute the following algorithms:
//		std::sort, saxpy, reduction, prefix sum
int main(int argc,char** argv){
	if (argc != 2)
		cout << "Error: Enter size in command line argument" << endl;
	else{
		//Setting the size of the array based on command line argument n
		size_t arrSize = std::atoi(argv[1]);
		//Creating array of n size
		vector<int> array1;
		//Creating clock variables for timing
		steady_clock::time_point ts, te;
		//Calling std::sort function
		stdSort(array1,arrSize,ts,te);
		//Calling saxpy algorithm
		saxpyAlg(arrSize,ts,te);
		//Calling Prefix-sum algorithm
		prefixSum(array1,arrSize,ts,te);
		//Pause before the program ends
		while(true){
			cout << endl << "Press any key to exit..." << endl;
			cin.get();
			break;
		}
	}
}

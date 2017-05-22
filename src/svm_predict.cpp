// Include libraries
#include <opencv2/opencv.hpp>
#include <ctype.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "svm.h"

// Namespace definitions
using namespace std;
using namespace cv;

// Allocate memory block
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

// SVM global varibles
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;

// Histogram Oriented Gradients (HOG) variables
vector<float> descriptorsValues;    	// hog descriptors of features
vector<Point> locations;            	// hog locations of features

/**********************************
******* AUXILIAR FUNCTIONS ********
**********************************/

void computeHOGDescriptors(Mat img)
{
    // Define variables to define HOG descriptor        
    Size winSize = Size(32, 16);
    Size blockSize = Size(8 ,8); 
    Size blockStride = Size(4, 4);
    Size cellSize = Size(4, 4);
    int nbins = 9;

    // Define hog descriptor
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);

    // Define HOG variables to compute features
    Size winStride = Size(0, 0);
    Size padding = Size(0, 0);
   
    // Compute hog descriptors extracting features and locations
    hog.compute(img, descriptorsValues, winStride, padding, locations);
}

vector<vector<float>> generateData(int problemSize, int featureNum)
{
	vector<vector<float>> data;
	data.push_back(descriptorsValues);
	return data;
}


/**********************************
************** MAIN ***************
**********************************/

int main(int argc, char **argv)
{
	if(argc < 2)
	{cout << "Missing image argument. Usage: svm_predict <image directory>" << endl;
	}else{

		// Load model
		cout << "Loading model..." << endl;
		svm_model *model = svm_load_model("model");
		cout << "Loading model: DONE" << endl;
		struct svm_node *x_space;

		// Load image
		string dir = string(argv[1]);
		Mat image = imread(dir, 0);	
		cout << "Image: " << dir <<  endl;

		// Extract features
		computeHOGDescriptors(image);

		int sizeOfProblem = 1;				// number of lines with labels
		int elements = descriptorsValues.size();	// number of features for each data vector
		vector<vector<float>> data = generateData(sizeOfProblem, elements);

		// initialize the size of the problem with just an int	
		prob.l = sizeOfProblem;
	
		// here we need to give some memory to our structures
		// @param prob.l = number of labels
		// @param elements = number of features for each label
		prob.y = Malloc(double, prob.l); 				// space for prob.l doubles
		prob.x = Malloc(struct svm_node *, prob.l); 			// space for prob.l pointers to struct svm_node
		x_space = Malloc(struct svm_node, (elements+1) * prob.l); 	// memory for pairs of index/value

		// initialize the svm_node vector with input data array as follows:
		int j=0; 					// counter to traverse x_space[i];
		for (int i=0;i < prob.l; i++)
		{
			// set i-th element of prob.x to the address of x_space[j]. 
			// elements from x_space[j] to x_space[j+data[i].size] get filled right after next line
			prob.x[i] = &x_space[j];
			for (int k=0; k<data[i].size(); ++k, ++j)
			{
				x_space[j].index=k+1; 		// index of value
				x_space[j].value=data[i][k]; 	// value
				// cout<<"x_space["<<j<<"].index = "<<x_space[j].index<<endl;
				// cout<<"x_space["<<j<<"].value = "<<x_space[j].value<<endl;
			}
			x_space[j].index=-1;			// state the end of data vector
			x_space[j].value=0;
			// cout<<"x_space["<<j<<"].index = "<<x_space[j].index<<endl;
			// cout<<"x_space["<<j<<"].value = "<<x_space[j].value<<endl;		
			j++;
		}

		// double svm_predict(const struct svm_model *model, const struct svm_node *x);
		cout << "Making predictions..." << endl;
		double retval = svm_predict(model, x_space);
		cout << "Making predictions: DONE" << endl;
		printf("PROBABILITY CLASS: %f\n",retval);
	}

	return 0;
}



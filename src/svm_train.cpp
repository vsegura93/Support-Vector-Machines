// Include libraries
#include <opencv2/opencv.hpp>
#include <sys/dir.h>
#include <cmath>
#include <ctype.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdio.h>
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
char model_file_name[1024] = "model";

// Histogram Oriented Gradients (HOG) variables
vector<vector<float>> vector_descriptorsValues; 	// vector of hog descriptors of features
vector<vector<Point>> vector_locations;    		// vector of hog locations of feature
vector<float> descriptorsValues;    			// hog descriptors of features
vector<Point> locations;            			// hog locations of features


/**********************************
******* AUXILIAR FUNCTIONS ********
**********************************/

vector<vector<float>> generateData(int problemSize, int featureNum)
{
	vector<vector<float>> data;
	for (int i=0; i<problemSize; i++)
	{ data.push_back(vector_descriptorsValues[i]); }
	return data;
}

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

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


bool is_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}

bool is_image(string name)
{
    bool find = false;
    if ((name.find("jpg") != std::string::npos) || (name.find("png") != std::string::npos) )
    { find = true; }
    return find;
}


/**********************************
************** MAIN ***************
**********************************/

int main()
{
	Mat image;
 	int total_images = 0;
	vector<string> classes_names;
	vector<vector<string>> images_names;
	vector<int> labels;
	
	// List folders of dataset
	cout << "Loading data..." << endl;
	string dir = string("dataset/");
	vector<string> files = vector<string>();
	getdir(dir, files);
	for (unsigned int i = 0;i < files.size();i++) {
		if(is_number(files[i]))
		{ classes_names.push_back(files[i]); }
	}

	for (int i=0; i<classes_names.size(); i++)
	{
		// List images of sub folder
		string sub_dir = dir+classes_names[i]+"/";
		cout << "List image of folder " << sub_dir << endl;
		vector<string> sub_files = vector<string>();
		getdir(sub_dir, sub_files);

		for (unsigned int j = 0; j<sub_files.size(); j++) {
			if (is_image(sub_files[j]))
			{ 	
				cout << sub_dir+sub_files[j] << endl;
				image = imread( sub_dir+sub_files[j], 0);

				computeHOGDescriptors(image);
				vector_descriptorsValues.push_back(descriptorsValues);
				vector_locations.push_back(locations);

				labels.push_back(stoi(classes_names[i]));
				total_images = total_images + 1;
			}
		}
	
	}
	cout << "Loading data: DONE" << endl;

	cout<<"Training system..." << endl;

	int sizeOfProblem = total_images;				// number of lines with labels
	int elements = vector_descriptorsValues[0].size();		// number of features for each data vector
	vector<vector<float> > data = generateData(sizeOfProblem, elements);

	// initialize the size of the problem with just an int	
	prob.l = sizeOfProblem;
	
	// here we need to give some memory to our structures
	// @param prob.l = number of labels
	// @param elements = number of features for each label
	prob.y = Malloc(double, prob.l); 				// space for prob.l doubles
	prob.x = Malloc(struct svm_node *, prob.l); 			// space for prob.l pointers to struct svm_node
	x_space = Malloc(struct svm_node, (elements+1) * prob.l); 	// memory for pairs of index/value

	// here we are going to initialize it all a bit
	// initialize the different lables with an array of labels
	for (int i=0; i < prob.l; i++)
	{ prob.y[i] = labels[i]; }

	// initialize the svm_node vector with input data array as follows:
	int j=0; 	// counter to traverse x_space[i];
	for (int i=0;i < prob.l; i++)
	{
		// set i-th element of prob.x to the address of x_space[j]. 
		// elements from x_space[j] to x_space[j+data[i].size] get filled right after next line
		prob.x[i] = &x_space[j];
		for (int k=0; k<data[i].size(); ++k, ++j)
		{
			x_space[j].index=k+1; 		// index of value
			x_space[j].value=data[i][k];	// value
		}
		x_space[j].index=-1;			// state the end of data vector
		x_space[j].value=0;
		j++;
	}

	/*
	-s svm_type : set type of SVM (default 0)
	0 -- C-SVC
	1 -- nu-SVC
	2 -- one-class SVM
	3 -- epsilon-SVR
	4 -- nu-SVR
	-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	*/
	//set all default parameters for param struct
	param.svm_type = 4;				
	param.kernel_type = 0;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	// Train SVM
	model = svm_train(&prob, &param);
	cout<<"Training system: DONE" << endl;

	// export model
	cout<<"Exporting model..." << endl;
	svm_save_model(model_file_name, model);
	cout<<"Exporting model: DONE" << endl;

	return 0;
}



cmake_minimum_required(VERSION 2.6)
project(svm_train)

# The library prefix
SET(LIB_PREFIX _svm_train)
SET(CMAKE_CXX_FLAGS "-g -Wall")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${svm_train_SOURCE_DIR}/bin)

# Add packages
FIND_PACKAGE(OpenCV REQUIRED)

# Add library and executables
ADD_LIBRARY(
	svm_lib
	svm.h svm.cpp
)
ADD_EXECUTABLE(svm_train svm_train.cpp)
ADD_EXECUTABLE(svm_predict svm_predict.cpp)

# Set target properties
SET_TARGET_PROPERTIES(svm_lib PROPERTIES OUTPUT_NAME ${LIB_PREFIX}_svm_lib)
SET_TARGET_PROPERTIES(svm_train PROPERTIES OUTPUT_NAME svm_train)
SET_TARGET_PROPERTIES(svm_predict PROPERTIES OUTPUT_NAME svm_predict)

# link the library to the executable
TARGET_LINK_LIBRARIES(svm_train svm_lib ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(svm_predict svm_lib ${OpenCV_LIBS})

CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT("CUDA RIBARDIÈRE WALCAK")

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_C_STANDARD 20)
ENABLE_LANGUAGE(CUDA)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(common/include)
FILE(GLOB COMMON
        "common/src/*.cpp")

###############
#  Grayscale  #
###############
INCLUDE_DIRECTORIES(grayscale/include)
FILE(GLOB GRAYSCALE_CPU_SOURCES
        "grayscale/src/*.cpp")
ADD_EXECUTABLE(grayscale_cpu ${GRAYSCALE_CPU_SOURCES} ${COMMON})
TARGET_LINK_LIBRARIES(grayscale_cpu ${OpenCV_LIBS})
FILE(GLOB GRAYSCALE_GPU_SOURCES
        "grayscale/src/*.cu")
ADD_EXECUTABLE(grayscale_gpu ${GRAYSCALE_GPU_SOURCES} ${COMMON})
TARGET_LINK_LIBRARIES(grayscale_gpu ${OpenCV_LIBS})

###############
#  ASCII Art  #
###############
INCLUDE_DIRECTORIES(asciiart/include)
FILE(GLOB ASCII_CPU_SOURCES
        "asciiart/src/*.cpp")
ADD_EXECUTABLE(ascii_cpu ${ASCII_CPU_SOURCES} ${COMMON})
TARGET_LINK_LIBRARIES(ascii_cpu ${OpenCV_LIBS})
FILE(GLOB ASCII_GPU_SOURCES
        "asciiart/src/*.cu")
ADD_EXECUTABLE(ascii_gpu ${ASCII_GPU_SOURCES} ${COMMON})
TARGET_LINK_LIBRARIES(ascii_gpu ${OpenCV_LIBS})

#################
#  Convolution  #
#################
INCLUDE_DIRECTORIES(convolution/include)
FILE(GLOB CONVOLUTION_CPU_SOURCES
        "convolution/src/*.cpp")
ADD_EXECUTABLE(convolution_cpu ${CONVOLUTION_CPU_SOURCES} ${COMMON})
TARGET_LINK_LIBRARIES(convolution_cpu ${OpenCV_LIBS})
FILE(GLOB CONVOLUTION_GPU_SOURCES
        "convolution/src/*.cu")
ADD_EXECUTABLE(convolution_gpu ${CONVOLUTION_GPU_SOURCES} ${COMMON})
TARGET_LINK_LIBRARIES(convolution_gpu ${OpenCV_LIBS})

cmake_minimum_required (VERSION 2.8)
project(MSSD)
add_definitions(-std=c++11)
set(OpenCV_DIR "/home/gjx/opencv/opencv-3.4.4/build")   
find_package(OpenCV REQUIRED)


set( TENGINE_DIR /home/gjx/visual-struct/Tengine-master ) 

set( INSTALL_DIR ${TENGINE_DIR}/install/)
set( TENGINE_LIBS tengine hclcpu)
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm32_lib/libprotobuf.so)
endif()

if(NOT WIN32)
  ADD_DEFINITIONS("-std=c++11")
endif(NOT WIN32)


if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()



if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm64_lib/libprotobuf.so)
endif()

if( BLAS_DIR)
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         set( BLAS_LIB ${BLAS_DIR}/arm32/lib/libopenblas.so)
    endif()
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
         set( BLAS_LIB ${BLAS_DIR}/arm64/lib/libopenblas.so)
    endif()
endif()

if ( NOT (ARM OR ANDROID))
set (BLAS_LIB openblas)
endif()

set( CODE_SRCS landmark.cpp yu_facedetect.cpp pose_estimate.cpp ../common/common.cpp)

#opencv
find_package(OpenCV REQUIRED)

#include
include_directories(${INSTALL_DIR}/include 
                    ${TENGINE_DIR}/examples/common )

#lib
link_directories( ${INSTALL_DIR}/lib)

#exe
add_executable(MSSD ${CODE_SRCS})
target_link_libraries(MSSD ${TENGINE_LIBS} ${PROTOBUF_LIB} ${OpenCV_LIBS} ${BLAS_LIB})

# if( ANDROID)
#    target_link_libraries(MSSD ${TENGINE_LIBS} ${PROTOBUF_LIB} ${OpenCV_LIBS} ${BLAS_LIB})
# else()
#    target_link_libraries(MSSD ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
# endif()




# Link your application with OpenCV libraries
# target_link_libraries(opencv_example ${OpenCV_LIBS})


#!/bin/bash
EN_TEST=0
if [ ${EN_TEST} -eq 1]; then
EXE_FILE="mytests"
CPP_FILE="mytests.cpp"

CMAKE_FILE="CMakeLists.txt"
echo "cmake_minimum_required(VERSION 3.10.2) # version can be different">${CMAKE_FILE}
echo "project(my_cpp_project) #name of your project">>${CMAKE_FILE}

# include openCV
echo "find_package( OpenCV REQUIRED )">>${CMAKE_FILE}
echo "include_directories( ${OpenCV_INCLUDE_DIRS} )">>${CMAKE_FILE}

echo "add_subdirectory(googletest) # add googletest subdirectory">>${CMAKE_FILE}
echo "include_directories(googletest/include) # this is so we can #include <gtest/gtest.h>">>${CMAKE_FILE}
echo "add_executable(${EXE_FILE} ${CPP_FILE}) # add this executable">>${CMAKE_FILE}
echo "target_link_libraries(${EXE_FILE} PRIVATE gtest) # link google test to this executable">>${CMAKE_FILE}

# link to openCV
echo "target_link_libraries( ${EXE_FILE} ${OpenCV_LIBS} )">>${CMAKE_FILE}

# https://stackoverflow.com/questions/75379251/why-do-i-get-an-error-about-a-dummy-variable-when-building-googletest-v1-10-with
echo "target_compile_options(gtest PRIVATE "-w")">>${CMAKE_FILE}

rm -rf build
mkdir -p build
cd build
cmake ..
make
./${EXE_FILE}

cd ..
fi
ENABLE_MAKE_CLEAN=1
ENABLE_MAKE=1
GET_ASM=1
ENABLE_RUN=1

CC="g++"
EXE_FILE="main.adx"
C_FILE="test.cpp"
MAP_FILE="main.map"
LFLAGS="-Wl,-Map=${MAP_FILE},--cref "
LFLAGS="-lfftw3 -lm ${LFLAGS}"
LFLAGS="`pkg-config --cflags --libs opencv4` ${LFLAGS}"
#LFLAGS="-lopencv_core452 -lopencv_imgcodecs452 -lopencv_imgproc452 -lopencv_calib3d452 -lopencv_dnn452 -lopencv_features2d452 -lopencv_flann452 -lopencv_gapi452 -lopencv_highgui452 -lopencv_ml452 -lopencv_objdetect452 -lopencv_photo452 -lopencv_stitching452 -lopencv_video452 -lopencv_videoio452 ${LFLAGS}"
LFLAGS="-Wl,--gc-sections ${LFLAGS}"
CFLAGS="-W -Wall -O0 -g3 ${LFLAGS}"
CFLAGS="-ffunction-sections -fdata-sections ${CFLAGS}"
CFLAGS=" ${CFLAGS}"

#generate ENV_FILE
ENV_FILE="NLIB_CROSS"

echo "CC=${CC}">${ENV_FILE}
echo "EXE_FILE=${EXE_FILE}">>${ENV_FILE}
echo "C_FILE=${C_FILE}">>${ENV_FILE}
echo "CFLAGS=${CFLAGS}">>${ENV_FILE}

if [ ${ENABLE_MAKE_CLEAN} -eq 1 ]; then
	make clean
fi

chmod 700 ${C_FILE}

if [ ${ENABLE_MAKE} -eq 1 ]; then
	make 1>make.log 2>make_warning.log
fi

#rep -inIr "\.text          " ${MAP_FILE}
#grep -inIr "\.text          " ${MAP_FILE} | awk '$0 ~/tmp/ {print}'
#grep -inIr "\.text          " ${MAP_FILE} | awk '{ printf "text =  %s\n",$3, $4} '


if [ ${GET_ASM} -eq 1 ]; then
	objdump -S ${EXE_FILE} > asm.log
	objdump -t ${EXE_FILE} > symbol.log
fi

if [ ${ENABLE_RUN} -eq 1 ]; then
	echo "below is result of ${EXE_FILE} :"
	./${EXE_FILE}
fi

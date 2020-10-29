cmake_minimum_required(VERSION 3.1)
project(face)
	
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/yu_facedetect.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O3 -fPIC -std=c++11")

if(ENABLE_NEON)
	add_definitions(-D_ENABLE_NEON)
	message("ENABLE_NEON")
elseif(NOT ENABLE_NEON)
	set(OpenCV_DIR  ../thirdParty/opencv-3.4.3/share/OpenCV)
	set(Tengine_DIR /work/Test/thirdParty/tenginelite-v1.0/x86-linux)
	find_package(OpenCV 3 REQUIRED)
endif()
set(TENGINE_LIBS tengine-lite)
link_directories(${Tengine_DIR}/lib)
message("TENGINE_LIBS :" ${Tengine_DIR}/lib)

include_directories(
	${Tengine_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(face ${SRC_FILES})
target_link_libraries(face ${TENGINE_LIBS} -Wl,--whole-archive ${OpenCV_LIBS} -Wl,--no-whole-archive)



project(ribtagclassfy)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/")

set(RIBRMAPI_SRC Classify.cpp)
add_library(${PROJECT_NAME} SHARED ${RIBRMAPI_SRC})

include_directories(
    /usr/local/include/opencv2/imgcodecs)

 target_link_libraries(${PROJECT_NAME}
   /usr/local/lib/libopencv_imgcodecs.so
   /usr/local/lib/libopencv_ml.so
   /usr/local/lib/libopencv_objdetect.so)

set(CMAKE_DEBUG_POSTFIX "d")
set_target_properties(${PROJECT_NAME} PROPERTIES
    DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

add_compile_options(-rdynamic)

install(TARGETS ${PROJECT_NAME} 
        LIBRARY
        DESTINATION ${CMAKE_INSTALL_PREFIX}/)

#set(CXXFLAGS "${CXXFLAGS} -Wl,-rpath=.")

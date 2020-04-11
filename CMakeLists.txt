cmake_minimum_required(VERSION 3.14)
project(template)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#find_package(PCL REQUIRED COMPONENTS COMMON IO OCTREE SEARCH FEATURES FILTERS SURFACE)
#find_package(Eigen3 REQUIRED)
#find_package(Boost REQUIRED COMPONENTS system)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ChoUtil REQUIRED PATHS "$ENV{HOME}/Repos/Ravel/ChoUtils/install")
# find_package(nanoflann REQUIRED)


set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    )

set(HDR_FILES
    "")


#add_library(${PROJECT_NAME} SHARED src/lib.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES} ${HDR_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${ChoUtil_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
    )
target_link_libraries(${PROJECT_NAME} fmt ChoUtil::core ChoUtil::util ${OpenCV_LIBRARIES})# nanoflann::nanoflann)

# Copy scripts.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
    add_custom_target(symlink_scripts ALL
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
        "${CMAKE_CURRENT_BINARY_DIR}/scripts")
endif()

# Debugging - print all variables.
if(FALSE)
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()

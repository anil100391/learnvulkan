cmake_minimum_required(VERSION 3.15)
project(vulkan)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# for intellisense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glfw3 CONFIG REQUIRED)

find_package(Vulkan REQUIRED)

find_package(glm CONFIG REQUIRED)

add_executable(vulkan main.cpp)

target_link_libraries(vulkan PRIVATE glfw Vulkan::Vulkan glm::glm-header-only)

project(tools LANGUAGES CXX)
find_package(Threads REQUIRED)

# Score test generator
add_executable(score_test_generator score_test_generator.cpp)

target_include_directories(score_test_generator
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

set(TOOLS_LINK_LIBS Threads::Threads)

target_link_libraries(score_test_generator
    PRIVATE
    ${TOOLS_LINK_LIBS})
# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/r320hyeon/broiron/ADAS_Project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/r320hyeon/broiron/ADAS_Project/build

# Include any dependencies generated for this target.
include lib/CMakeFiles/laneDetect.dir/depend.make

# Include the progress variables for this target.
include lib/CMakeFiles/laneDetect.dir/progress.make

# Include the compile flags for this target's objects.
include lib/CMakeFiles/laneDetect.dir/flags.make

lib/CMakeFiles/laneDetect.dir/laneDetect.cpp.o: lib/CMakeFiles/laneDetect.dir/flags.make
lib/CMakeFiles/laneDetect.dir/laneDetect.cpp.o: ../lib/laneDetect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/r320hyeon/broiron/ADAS_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/CMakeFiles/laneDetect.dir/laneDetect.cpp.o"
	cd /home/r320hyeon/broiron/ADAS_Project/build/lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/laneDetect.dir/laneDetect.cpp.o -c /home/r320hyeon/broiron/ADAS_Project/lib/laneDetect.cpp

lib/CMakeFiles/laneDetect.dir/laneDetect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/laneDetect.dir/laneDetect.cpp.i"
	cd /home/r320hyeon/broiron/ADAS_Project/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/r320hyeon/broiron/ADAS_Project/lib/laneDetect.cpp > CMakeFiles/laneDetect.dir/laneDetect.cpp.i

lib/CMakeFiles/laneDetect.dir/laneDetect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/laneDetect.dir/laneDetect.cpp.s"
	cd /home/r320hyeon/broiron/ADAS_Project/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/r320hyeon/broiron/ADAS_Project/lib/laneDetect.cpp -o CMakeFiles/laneDetect.dir/laneDetect.cpp.s

# Object files for target laneDetect
laneDetect_OBJECTS = \
"CMakeFiles/laneDetect.dir/laneDetect.cpp.o"

# External object files for target laneDetect
laneDetect_EXTERNAL_OBJECTS =

lib/liblaneDetect.a: lib/CMakeFiles/laneDetect.dir/laneDetect.cpp.o
lib/liblaneDetect.a: lib/CMakeFiles/laneDetect.dir/build.make
lib/liblaneDetect.a: lib/CMakeFiles/laneDetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/r320hyeon/broiron/ADAS_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library liblaneDetect.a"
	cd /home/r320hyeon/broiron/ADAS_Project/build/lib && $(CMAKE_COMMAND) -P CMakeFiles/laneDetect.dir/cmake_clean_target.cmake
	cd /home/r320hyeon/broiron/ADAS_Project/build/lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/laneDetect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/CMakeFiles/laneDetect.dir/build: lib/liblaneDetect.a

.PHONY : lib/CMakeFiles/laneDetect.dir/build

lib/CMakeFiles/laneDetect.dir/clean:
	cd /home/r320hyeon/broiron/ADAS_Project/build/lib && $(CMAKE_COMMAND) -P CMakeFiles/laneDetect.dir/cmake_clean.cmake
.PHONY : lib/CMakeFiles/laneDetect.dir/clean

lib/CMakeFiles/laneDetect.dir/depend:
	cd /home/r320hyeon/broiron/ADAS_Project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/r320hyeon/broiron/ADAS_Project /home/r320hyeon/broiron/ADAS_Project/lib /home/r320hyeon/broiron/ADAS_Project/build /home/r320hyeon/broiron/ADAS_Project/build/lib /home/r320hyeon/broiron/ADAS_Project/build/lib/CMakeFiles/laneDetect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/CMakeFiles/laneDetect.dir/depend


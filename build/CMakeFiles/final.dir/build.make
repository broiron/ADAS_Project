# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/r320/broiron/final

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/r320/broiron/final/build

# Include any dependencies generated for this target.
include CMakeFiles/final.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/final.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/final.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/final.dir/flags.make

CMakeFiles/final.dir/src/main.cpp.o: CMakeFiles/final.dir/flags.make
CMakeFiles/final.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/final.dir/src/main.cpp.o: CMakeFiles/final.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/r320/broiron/final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/final.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/final.dir/src/main.cpp.o -MF CMakeFiles/final.dir/src/main.cpp.o.d -o CMakeFiles/final.dir/src/main.cpp.o -c /home/r320/broiron/final/src/main.cpp

CMakeFiles/final.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/final.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/r320/broiron/final/src/main.cpp > CMakeFiles/final.dir/src/main.cpp.i

CMakeFiles/final.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/final.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/r320/broiron/final/src/main.cpp -o CMakeFiles/final.dir/src/main.cpp.s

# Object files for target final
final_OBJECTS = \
"CMakeFiles/final.dir/src/main.cpp.o"

# External object files for target final
final_EXTERNAL_OBJECTS =

final: CMakeFiles/final.dir/src/main.cpp.o
final: CMakeFiles/final.dir/build.make
final: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_barcode.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudabgsegm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudafeatures2d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaobjdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudastereo.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_sfm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_wechat_qrcode.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_xfeatures2d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.5.5
final: lib/libyoloDetect.a
final: lib/liblaneDetect.a
final: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudacodec.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaoptflow.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudalegacy.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudawarping.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaimgproc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudafilters.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaarithm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudev.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_barcode.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudabgsegm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudafeatures2d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaobjdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudastereo.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_sfm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudacodec.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaoptflow.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudalegacy.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudawarping.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_wechat_qrcode.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_xfeatures2d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaimgproc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudafilters.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudaarithm.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.5
final: /usr/lib/aarch64-linux-gnu/libopencv_cudev.so.4.5.5
final: CMakeFiles/final.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/r320/broiron/final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable final"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/final.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/final.dir/build: final
.PHONY : CMakeFiles/final.dir/build

CMakeFiles/final.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/final.dir/cmake_clean.cmake
.PHONY : CMakeFiles/final.dir/clean

CMakeFiles/final.dir/depend:
	cd /home/r320/broiron/final/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/r320/broiron/final /home/r320/broiron/final /home/r320/broiron/final/build /home/r320/broiron/final/build /home/r320/broiron/final/build/CMakeFiles/final.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/final.dir/depend


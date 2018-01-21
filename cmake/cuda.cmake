option(BUILD_TNT_WITH_GPU_SUPPORT "Accelerate TNT using CUDA." OFF)
if (BUILD_TNT_WITH_GPU_SUPPORT)
    find_package(CUDA 7.5)

    if (CUDA_FOUND               AND
        MSVC                     AND NOT
        CUDA_CUBLAS_LIBRARIES    AND
        "${CMAKE_SIZEOF_VOID_P}" EQUAL "4")
        message(WARNING "You have CUDA installed, but we can't use it unless you put visual studio in 64bit mode.")
        set(CUDA_FOUND OFF)
    endif()

    if (NOT CUDA_FOUND)
        message(FATAL_ERROR "*** Neurotic can't find CUDA! ***")
    endif()

    mark_as_advanced(CUDA_HOST_COMPILER CUDA_SDK_ROOT_DIR CUDA_TOOLKIT_ROOT_DIR CUDA_USE_STATIC_CUDA_RUNTIME)

    # Include the CUDA headers
    include_directories(${CUDA_INCLUDE_DIRS})

    # There is a bug on OSX that messes up propagation of the -std=c++11 flag so
    # we have to it manually
    if (APPLE)
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    
        # Get all the -D flags from CMAKE_CXX_FLAGS and add them manually to the cuda flags
        string(REGEX MATCHALL "-D[^ ]*" FLAGS_FOR_NVCC ${CMAKE_CXX_FLAGS})
    endif()

    set(CUDA_HOST_COMPILATION_CPP ON)

    # Note that we add __STRICT_ANSI__ to avoid freaking out nvcc with gcc specific
    # magic in the standard C++ header files (since nvcc uses gcc headers on
    # linux).
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES;${FLAGS_FOR_NVCC}")

    # Add debug flags to NVCC if we are in debug mode
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-G;-g")
    endif()

    add_definitions(-DBUILD_WITH_GPU_SUPPORT)

    set(TNT_THIRDPARTY_LIBS ${TNT_THIRDPARTY_LIBS} ${CUDA_CUBLAS_LIBRARIES}
                                                   ${CUDA_curand_LIBRARY})
endif()

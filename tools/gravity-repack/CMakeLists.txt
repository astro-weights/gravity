set(TARGET llama-gravity-repack)
add_executable(${TARGET} gravity-repack.cpp)
target_link_libraries(${TARGET} PRIVATE gravity-stream)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

llama_add_compile_flags()

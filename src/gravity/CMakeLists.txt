# gravity streaming groundwork (Phase 1): standalone pieces, not called by libllama yet.
# Design ideas credited to Strata (github.com/Niko1221/Strata, MIT); no Strata code is used.
add_library(gravity-stream STATIC
    mem-budget.cpp
    io-mode.cpp
    read-pool.cpp
    swap-policy.cpp
    slot-ring.cpp
    scheduler.cpp
    residency.cpp
)
target_include_directories(gravity-stream PUBLIC .)
target_compile_features(gravity-stream PUBLIC cxx_std_17)
set_target_properties(gravity-stream PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(gravity-stream PUBLIC Threads::Threads)

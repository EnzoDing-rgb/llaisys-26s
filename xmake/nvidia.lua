-- 这份文件 = NVIDIA 版的「怎么编」；对照 xmake/cpu.lua 看结构几乎一一对应。
-- 当你执行: xmake f --nv-gpu=y 时，根目录 xmake.lua 会 includes 本文件，
-- 同时 #define ENABLE_NVIDIA_API，C++ 里 #ifdef 的 NVIDIA 分支会进编译。
-- 环境备忘：登录节点有 /usr/local/cuda（能 nvcc），真正 GPU 在 lcpu-infra（RTX 5090 → sm_120）。

-- target(...)：声明一个「编译产物」。名字 llaisys-device-nvidia 会被 xmake.lua 里
-- add_deps("llaisys-device-nvidia") 引用 —— 依赖名必须和这里完全一致。
target("llaisys-device-nvidia")
    -- static = 打成 .a 静态库；最后由共享库 libllaisys.so 再链进来。
    -- 对照 cpu.lua：那边是 llaisys-device-cpu，也是 static。
    set_kind("static")

    -- xmake 看到 .cu 文件会自动走 nvcc
    set_languages("cxx17")
    set_warnings("all", "error")

    -- 依赖 utils（ASSERT / TO_BE_IMPLEMENTED 等宏所在库），保证链接顺序正确。
    add_deps("llaisys-utils")

    if not is_plat("windows") then
        -- -fPIC：生成「位置无关代码」，静态库以后要链进 .so 时 Linux 上几乎必须开。
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        -- .cu 是 nvcc 编译的；主机侧 PIC 要通过 -Xcompiler 转交给底层 g++/clang。
        -- {force = true}：让 xmake 保留这条 flag。
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
    end

    -- xmake 默认 cuda.rdc=true（-rdc=true）。本 target 目前只有 host 侧 Runtime 封装，
    -- 设为 false：编出来的 .o 可直接被 g++ 链进 libllaisys.so。
    -- 以后加 __global__ kernel 时改回默认，并 set_policy("build.cuda.devlink", true)。
    set_values("cuda.rdc", false)

    -- 只编 device 后端（Runtime API + Resource），路径相对本 lua 文件所在目录。
    -- *.cu = CUDA 源；对照 cpu.lua 的 ../src/device/cpu/*.cpp。
    add_files("../src/device/nvidia/*.cu")

    -- 告诉 nvcc「为哪种 GPU 架构生成机器码」。
    -- sm_XX = 针对该架构的真实 SASS；compute_XX = 可 JIT 的中间码（PTX）。
    -- 本机群 RTX 5090 的 compute capability = 12.0 → 写成 120（中间没有点）。
    add_cugencodes("sm_120")
    add_cugencodes("compute_120")

    -- cudart = CUDA Runtime 库（cudaMalloc / cudaMemcpy 等符号都在这里）。
    -- 系统库用 add_syslinks；工程内自己的库用 add_links。
    add_syslinks("cudart")
    -- 让依赖本 target 的代码也能找到 cuda_runtime.h；public = 依赖方一并继承。
    add_includedirs("/usr/local/cuda/include", {public = true})
    -- 链接时去这个目录找 libcudart.so。
    add_linkdirs("/usr/local/cuda/lib64")

    -- 真正对外安装的是最终的 libllaisys.so（见根 xmake.lua）。
    on_install(function (target) end)
target_end()

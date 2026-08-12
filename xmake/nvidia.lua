-- CUDA / CUDA-compat GPU build (NVIDIA or Iluvatar COREX).
-- Detect Iluvatar via COREX_HOME or /usr/local/corex; otherwise NVIDIA CUDA.

includes("iluvatar.lua")

local function detect_gpu_stack()
    local corex = os.getenv("COREX_HOME") or "/usr/local/corex"
    if os.isdir(corex) and os.isfile(path.join(corex, "include", "cuda_runtime.h")) then
        return "iluvatar", corex, path.join(corex, "lib")
    end

    local cuda = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME") or "/usr/local/cuda"
    if os.isdir(cuda) then
        local lib64 = path.join(cuda, "lib64")
        if not os.isdir(lib64) then
            lib64 = path.join(cuda, "lib")
        end
        return "nvidia", cuda, lib64
    end

    raise("Neither Iluvatar COREX (/usr/local/corex) nor NVIDIA CUDA (/usr/local/cuda) found")
end

local stack, cuda_root, cuda_lib = detect_gpu_stack()
print(string.format("[llaisys] GPU stack=%s root=%s lib=%s", stack, cuda_root, cuda_lib))

local function configure_cuda_target()
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_values("cuda.rdc", false)

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end

    add_syslinks("cudart")
    add_includedirs(path.join(cuda_root, "include"), {public = true})
    add_linkdirs(cuda_lib)

    if stack == "iluvatar" then
        -- COREX: .cu must be compiled as ivcore (not NVIDIA cubin/ptx)
        set_toolchains("iluvatar.toolchain")
        add_rules("iluvatar.env")
        -- COREX clang defaults to pre-C++17 for .cu unless we force the dialect.
        add_cuflags("-x", "ivcore", "-fPIC", "-std=c++17", {force = true})
        add_culdflags("-x", "ivcore", {force = true})
        add_cxxflags("-std=c++17", {force = true})
    else
        local arch = os.getenv("XMAKE_CUDA_ARCH") or "sm_120"
        add_cugencodes(arch)
        if not is_plat("windows") then
            add_cuflags("-Xcompiler", "-fPIC", {force = true})
        end
    end

    on_install(function (target) end)
end

target("llaisys-device-nvidia")
    configure_cuda_target()
    add_deps("llaisys-utils")
    add_files("../src/device/nvidia/*.cu")
target_end()

target("llaisys-ops-nvidia")
    configure_cuda_target()
    add_deps("llaisys-tensor")
    add_files("../src/ops/*/nvidia/*.cu")
target_end()

add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    -- 天数（Iluvatar COREX）：探测到 COREX 时额外打开一等设备宏。
    -- 天数复用同一套 CUDA runtime/kernel（COREX 提供 CUDA 兼容接口），
    -- 只是多一个 LLAISYS_DEVICE_ILUVATAR 设备类型 + iluvatar::getRuntimeAPI 别名。
    local corex = os.getenv("COREX_HOME") or "/usr/local/corex"
    if os.isdir(corex) and os.isfile(path.join(corex, "include", "cuda_runtime.h")) then
        add_defines("ENABLE_ILUVATAR_API")
    end
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    if has_config("nv-gpu") then
        add_syslinks("cudart")
        -- Prefer Iluvatar COREX when present; otherwise NVIDIA CUDA.
        local corex = os.getenv("COREX_HOME") or "/usr/local/corex"
        local cuda = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME") or "/usr/local/cuda"
        if os.isdir(corex) then
            add_linkdirs(path.join(corex, "lib"))
            add_rpathdirs(path.join(corex, "lib"))
        elseif os.isdir(path.join(cuda, "lib64")) then
            add_linkdirs(path.join(cuda, "lib64"))
            add_rpathdirs(path.join(cuda, "lib64"))
        elseif os.isdir(path.join(cuda, "lib")) then
            add_linkdirs(path.join(cuda, "lib"))
            add_rpathdirs(path.join(cuda, "lib"))
        end
    end
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
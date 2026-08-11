-- 这份文件 = NVIDIA 版的「怎么编」；对照 xmake/cpu.lua 看结构几乎一一对应。
-- xmake f --nv-gpu=y 时由根 xmake.lua includes，并定义 ENABLE_NVIDIA_API。
-- GPU：lcpu-infra / RTX 5090 → sm_120；toolkit：/usr/local/cuda。

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_deps("llaisys-utils")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
    end

    -- host-only Runtime 封装：关 RDC，便于 g++ 直接链进 libllaisys.so
    set_values("cuda.rdc", false)

    add_files("../src/device/nvidia/*.cu")

    add_cugencodes("sm_120")
    add_cugencodes("compute_120")
    add_syslinks("cudart")
    add_includedirs("/usr/local/cuda/include", {public = true})
    add_linkdirs("/usr/local/cuda/lib64")

    on_install(function (target) end)
target_end()

-- 对称于 cpu.lua 的 llaisys-ops-cpu；根 xmake.lua 里 llaisys-ops 在 nv-gpu 时 add_deps 本 target
target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
    end

    set_values("cuda.rdc", false)

    add_files("../src/ops/*/nvidia/*.cu")

    add_cugencodes("sm_120")
    add_cugencodes("compute_120")
    add_syslinks("cudart")
    add_includedirs("/usr/local/cuda/include", {public = true})
    add_linkdirs("/usr/local/cuda/lib64")

    on_install(function (target) end)
target_end()

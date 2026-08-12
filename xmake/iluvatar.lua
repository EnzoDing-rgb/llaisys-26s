-- Iluvatar (天数智芯) COREX toolchain + CUDA-compat build helpers.
-- Official sample: LearningInfiniTensor/.github/server/iluvatar/xmake-sample

toolchain("iluvatar.toolchain")
    set_toolset("cc", "clang")
    set_toolset("cxx", "clang++")
    set_toolset("cu", "clang++")
    set_toolset("culd", "clang++")
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
toolchain_end()

rule("iluvatar.env")
    add_deps("cuda.env")
    add_orders("cuda.env", "iluvatar.env")
    after_load(function (target)
        local old = target:get("syslinks") or {}
        local new = {}
        for _, link in ipairs(old) do
            if link ~= "cudadevrt" then
                table.insert(new, link)
            end
        end
        if #old > #new then
            target:set("syslinks", new)
        end
    end)
rule_end()

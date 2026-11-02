// almond-engine: one executable for every model.  The arguments say which path runs: GLM-5.3-Flash's when they name
// its pack (--glm-pack, src/glm), Qwen3.8-Flash-Next's when they name --qwen (src/qwen, which reads the rest),
// DeepSeek-V4.1-Flash's otherwise (src/ds41).  Each path keeps its own command line and its own serve protocol lines,
// as when they were separate programs (docs/ENGINE_PLAN.md, step 3).
#include <cstring>

int ds41_main(int argc, char ** argv);
int glm_main(int argc, char ** argv);
int qwen_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--qwen") == 0) {     // taken out: Qwen's path reads the rest as its own
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            argv[argc - 1] = nullptr;
            return qwen_main(argc - 1, argv);
        }
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--glm-pack") == 0) return glm_main(argc, argv);
    return ds41_main(argc, argv);
}

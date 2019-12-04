#include <vector>
#include <string>


namespace MySpace {
    struct CmdArgs {
        std::string dictionary_path;
        std::string bigrams_path;
        std::string logits_path;
    };


    CmdArgs parse_args(int argc, char *argv[]);
}

#include "argparser.h"


MySpace::CmdArgs MySpace::parse_args(int argc, char *argv[]) {
    std::string res_s = argv[0];

    std::string dictionary_path = argv[1];
    std::string bigrams_path = argv[2];
    std::string pmf_path = argv[3];

    CmdArgs res;
    res.dictionary_path = dictionary_path;
    res.bigrams_path = bigrams_path;
    res.logits_path = pmf_path;
    return res;
}

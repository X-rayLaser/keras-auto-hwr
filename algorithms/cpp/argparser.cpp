#include "argparser.h"
#include <iostream>
#include <fstream>
 


MySpace::CmdArgs MySpace::parse_args(int argc, char *argv[]) {
    std::string res_s = argv[0];

    std::string dictionary_path = argv[1];
    std::string bigrams_path = argv[2];
    std::string pmf_path = argv[3];

    std::vector<std::vector<double>> distributions;

    std::ifstream ifs(pmf_path);

    int seq_len;
    int num_classes;

    if (ifs.fail()) {
        std::cout << "Failed to open a file\n";

    }

    ifs >> seq_len >> num_classes;

    for (int i = 0; i < seq_len; i++) {
        std::vector<double> v;
        distributions.push_back(v);
    }

    for (int i = 0; i < seq_len; i ++){
        std::vector<float> pmf;

        for (int j = 0; j < num_classes; j++) {
            double p;
            ifs >> p;
            distributions[i].push_back(p);
            std::string s = std::to_string(distributions[i][j]);
        }
    }

    CmdArgs res;
    res.dictionary_path = dictionary_path;
    res.bigrams_path = bigrams_path;
    res.distributions = distributions;
    return res;
}

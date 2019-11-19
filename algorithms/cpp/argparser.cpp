#include "argparser.h"
 
MySpace::CmdArgs MySpace::parse_args(int argc, char *argv[]) {
    std::string res_s = argv[0];

    std::string dictionary_path = argv[1];
    std::string bigrams_path = argv[2];

    double seq_len = std::atof(argv[3]);
    double num_classes = std::atof(argv[4]);

    argv++; //skip program's name
    argv++; //skip dictionary path
    argv++; //skip bigrams_path
    argv++; //skip seq_len arg
    argv++; //skip num_classes arg

    int counter = 0;

    std::vector<std::vector<float>> distributions;

    for (int i = 0; i < seq_len; i++) {
        std::vector<float> v;
        distributions.push_back(v);
    }

    for (int i = 0; i < seq_len; i ++){
        std::vector<float> pmf;

        for (int j = 0; j < num_classes; j++) {
            double p = std::atof(argv[counter]);
            distributions[i].push_back(p);
            std::string s = std::to_string(distributions[i][j]);
            counter++;
        }
    }

    CmdArgs res;
    res.dictionary_path = dictionary_path;
    res.bigrams_path = bigrams_path;
    res.distributions = distributions;
    return res;
}

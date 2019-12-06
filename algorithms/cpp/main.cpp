#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include "argparser.h"
#include "io_utils.h"
#include "token_passing.h"

using std::cout;
using std::string;
using std::vector;


vector<vector<double>> build_pmf(std::string pmf_path) {
    vector<vector<double>> distributions;

    std::ifstream ifs(pmf_path);

    int seq_len;
    int num_classes;

    if (ifs.fail()) {
        cout << "Failed to open a file\n";

    }

    ifs >> seq_len >> num_classes;

    for (int i = 0; i < seq_len; i++) {
        vector<double> v;
        distributions.push_back(v);
    }

    for (int i = 0; i < seq_len; i ++){
        vector<float> pmf;

        for (int j = 0; j < num_classes; j++) {
            double p;
            ifs >> p;
            distributions[i].push_back(p);
            string s = std::to_string(distributions[i][j]);
        }
    }

    return distributions;
}


string run_algorithm(string dict_path, string bigrams_path, vector<vector<double>>& pmf) {

    auto dictionary = get_dictionary(dict_path);

    std::map<int, TransitionRoot> transitions = get_transitions(bigrams_path);

    auto result = token_passing(dictionary, transitions, pmf);

    string s;
    for (auto word_index : result) {
        s += std::to_string(word_index) + " ";
    }

    return s;
}


vector<vector<double>> make_matrix(double* flatten, int h, int w) {
    vector<vector<double>> rows;
    int count = 0;
    for (int i = 0; i < h; i++) {
        vector<double> v;
        for (int j = 0; j < w; j++) {
            v.push_back(flatten[count]);
            count += 1;
        }
        rows.push_back(v);
    }

    return rows;
}


extern "C" {
    char* token_passing_js(char* dictionary_path, char* bigrams_path, int steps, int num_classes, double* logits) {
        string dict_path(dictionary_path);
        string bi_path(bigrams_path);

        vector<vector<double>> pmf = make_matrix(logits, steps, num_classes);

        string s = run_algorithm(dict_path, bi_path, pmf);

        const char* cstr = s.c_str();

        char* str = (char*)malloc(sizeof(char)*s.size());
        strcpy(str, cstr);
        return str;
    }

    void test(char* dictionary_path) {
        string dict_path(dictionary_path);
        get_dictionary(dict_path);
    }
}


int main(int argc, char *argv[])
{
    auto args = MySpace::parse_args(argc, argv);

    auto distributions = build_pmf(args.logits_path);

    auto res = run_algorithm(args.dictionary_path, args.bigrams_path, distributions);

    cout << res;

    return 0;
}

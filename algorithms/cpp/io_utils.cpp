#include "io_utils.h"
#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>

WordRepresentation::WordRepresentation(std::vector<int> v, double p) : m_v(v), m_p(p) {}

std::vector<int> WordRepresentation::as_vector() {
    return m_v;
}


double WordRepresentation::probability() {
    return m_p;
}


TransitionRoot::TransitionRoot(int src, std::map<int, double> transitions):
    m_from(src), m_transitions(transitions) {}

TransitionRoot::TransitionRoot(const TransitionRoot& root) : m_from(root.m_from), m_transitions(root.m_transitions) {}

double TransitionRoot::probability(int to) {
    auto search = m_transitions.find(to);
    if (search != m_transitions.end()) {
        return search->second;
    } else {
        return 0.0;
    }
}


std::vector<std::pair<int, double>> TransitionRoot::children() {
    std::vector<std::pair<int, double>> v;
    
    for (auto kv_pair : m_transitions) {
        v.push_back(kv_pair);
    }

    return v;
}


std::vector<WordRepresentation> get_dictionary(std::string file_path) {
    std::ifstream ifs(file_path);
    std::string s;

    std::vector<WordRepresentation> res;

    while (std::getline(ifs, s))
    {
        std::istringstream iss{ s };

        double p;
        iss >> p;
   
        std::vector<int> data{std::istream_iterator<int>(iss), std::istream_iterator<int>()};

        WordRepresentation wr(data, p);
        res.push_back(wr);
    }

    return res;
}


std::map<int, TransitionRoot> get_transitions(std::string file_path) {
    std::ifstream ifs(file_path);

    int num_sources;
    ifs >> num_sources;

    std::map<int, TransitionRoot> res;

    for (int i = 0; i < num_sources; i++) {
        int source_index;
        int num_connections;
        ifs >> source_index >> num_connections;
        
        std::map<int, double> connections;

        for (int j = 0; j < num_connections; j++) {
            int destination;
            double probability;
            ifs >> destination >> probability;
            
            connections[destination] = probability;
        }

        TransitionRoot root(source_index, connections);

        res.emplace(source_index, root);
    }

    return res;
}

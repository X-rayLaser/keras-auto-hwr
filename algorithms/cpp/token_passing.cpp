#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "argparser.h"
#include "io_utils.h"

using namespace std;


struct Candidate {
    double score;
    int model_id;
    int node_index;
};


struct TransitLink {
    int model_id;
    int node_index;
};


struct Node {
    int node_id;
    int model_id;
    int code;
    double score;
    Candidate candidate;

    vector<TransitLink> linked_nodes;
    bool is_input;
    bool is_output;

    vector<TransitLink> history;
};

vector<Node> create_model(vector<int> codes, int model_id) {
    vector<Node> v;

    for (auto code : codes) {
        Node node;
        node.node_id = v.size();
        node.model_id = model_id;
        node.code = code;
        node.score = -1;
        node.candidate.score = -1;
        node.is_input = false;
        node.is_output = false;
        v.push_back(node);
    }

    for (int i = 0; i < v.size(); i++) {
        vector<TransitLink> linked;

        TransitLink transit_link;
        transit_link.model_id = model_id;
        transit_link.node_index = v[i].node_id;
        linked.push_back(transit_link);

        int next_index = i + 1;
        if (next_index < v.size()) {
            TransitLink transit_link;
            transit_link.model_id = model_id;
            transit_link.node_index = v[next_index].node_id;
            linked.push_back(transit_link);
        }

        v[i].linked_nodes = linked;
    }

    v[0].is_input = true;
    v[v.size() - 1].is_output = true;
    return v;
}


void run_iteration(vector<vector<Node>>& graph,
                   const vector<float>& pmf,
                   const std::map<int, TransitionRoot>& p_bigram) {
    for (auto& word_model : graph) {
        for (auto& node : word_model) {
            for (auto transition_link : node.linked_nodes) {                
                Node* linked_ptr = &graph[transition_link.model_id][transition_link.node_index];

                double new_score = node.score - log(pmf[linked_ptr->code]);
                if (linked_ptr->is_input && node.is_output) {
                    auto it = p_bigram.find(transition_link.model_id);
                    double p = 0;

                    if (it == p_bigram.end()) {
                        // verify this!
                        TransitionRoot root = p_bigram.find(transition_link.model_id)->second;

                        double p = root.probability(transition_link.node_index);
                    }
                    
                    double transition_cost = - log(p);
                    new_score = node.score + transition_cost;
                }

                if (new_score > linked_ptr->candidate.score) {
                    linked_ptr->candidate.score = new_score;
                    TransitLink node_identifier;
                    node_identifier.model_id = node.model_id;
                    node_identifier.node_index = node.node_id;

                    linked_ptr->candidate.model_id = node.model_id;
                    linked_ptr->candidate.node_index = node.node_id;
                }
            }
        }
    }
}


void commit(vector<vector<Node>>& graph) {
    for (auto& word_model : graph) {
        for (auto& node : word_model) {
            node.score = node.candidate.score;

            TransitLink transit_link;
            transit_link.node_index = node.candidate.node_index;
            transit_link.model_id = node.candidate.model_id;
            node.history.push_back(transit_link);
        }
    }
}


Node top_rated_node(const vector<vector<Node>>& graph) {
    double max_score = -1;
    Node res = graph[0][0];
    for (auto word_model : graph) {
        for (auto node : word_model) {
            if (node.score > max_score) {
                max_score = node.score;
                res = node;
            }
        }
    }

    return res;
}


vector<vector<Node>> create_graph(const std::vector<WordRepresentation>& dictionary) {
    vector<vector<Node>> graph;

    for (int i = 0; i < dictionary.size(); i++) {
        auto word = dictionary[i];
        auto codes = word.as_vector();
        auto model = create_model({1, 2, 3, 4}, i);
        graph.push_back(model);
    }

    for (auto& model_a : graph) {
        for (auto model_b : graph) {
            TransitLink word_link;
            word_link.model_id = model_b[0].model_id;
            word_link.node_index = 0;
            model_a[model_a.size() - 1].linked_nodes.push_back(word_link);
        }
    }

    return graph;
}


list<int> get_optimal_path(const vector<vector<Node>>& graph) {
    auto node = top_rated_node(graph);

    list<int> result;
    result.push_front(node.code);

    int num_iterations = node.history.size();

    for (int t = num_iterations - 1; t >= 0; t--) {
        auto model_id = node.history[t].model_id;
        auto node_index = node.history[t].node_index;
        Node node = graph[model_id][node_index];
        result.push_front(node.code);
    }

    return result;
}


list<int> token_passing(const std::vector<WordRepresentation>& dictionary,
                        const std::map<int, TransitionRoot>& transitions,
                        const std::vector<std::vector<float>>& pmfs) {
    auto graph = create_graph(dictionary);

    int num_iterations = pmfs.size();

    for (int t = 0; t < num_iterations; t++) {
        run_iteration(graph, pmfs[t], transitions);
        commit(graph);
    }

    list<int> result = get_optimal_path(graph);

    return result;
}


int main(int argc, char *argv[])
{
    auto args = MySpace::parse_args(argc, argv);

    auto dictionary = get_dictionary(args.dictionary_path);
    std::map<int, TransitionRoot> transitions = get_transitions(args.bigrams_path);

    auto result = token_passing(dictionary, transitions, args.distributions);

    cout << "\n";
    for (auto code : result) {
        cout << code << " ";
    }

    return 0;
}

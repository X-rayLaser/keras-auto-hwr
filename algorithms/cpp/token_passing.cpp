#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <limits>

#include "argparser.h"
#include "io_utils.h"
#include "token_passing.h"

using namespace std;


inline double compute_cost(double p) {
    return - log(p);
}


Candidate::Candidate() {
    this->score = std::numeric_limits<double>::infinity();
    this->model_id = -1;
    this->node_index = -1;
}

Candidate::Candidate(double score, NodeId node_id) {
    this->score = score;
    this->model_id = node_id.model_id;
    this->node_index = node_id.node_index;
}


Node::Node(int node_id, int model_id, int code):
        m_node_id(node_id), m_model_id(model_id),
        m_code(code), m_score(std::numeric_limits<double>::infinity())
{
}


InternalTransition::InternalTransition(Node* src, Node* dest):
    m_src(src), m_dest(dest)
{
}


void InternalTransition::pass_token(const vector<double>& emission_costs) {
    int code = m_dest->code();
    double cost = emission_costs[code];
    float new_score = m_src->score() + cost;

    Candidate candidate(new_score, m_src->node_id());
    m_dest->pass_token(candidate);
}

CrossTransition::CrossTransition(Node* src, Node* dest, double transition_cost):
    m_src(src), m_dest(dest), m_cost(transition_cost)
{
}

void CrossTransition::pass_token(const vector<double>& emission_costs) {
    int code = m_dest->code();
    double new_score = m_src->score() + emission_costs[m_dest->code()] + m_cost;

    Candidate candidate(new_score, m_src->node_id());
    m_dest->pass_token(candidate);
}


InitialTransition::InitialTransition(Node* dest, double transition_cost):
    m_dest(dest), m_cost(transition_cost), m_done(false)
{
}


void InitialTransition::pass_token(const vector<double>& emission_costs) {
    // remove duplication
    if (!m_done) {
        int code = m_dest->code();

        Candidate candidate;
        candidate.score = emission_costs[m_dest->code()] + m_cost;
        m_dest->pass_token(candidate);
        m_done = true;
    }
}

 
Graph::Graph(const std::vector<WordRepresentation>& dictionary,
             const std::map<int, TransitionRoot>& p_bigram,
             const std::vector<std::vector<double>>& pmfs)
{
    precompute_costs(pmfs);
    init_models(dictionary);
    make_word_level_transitions(pmfs);
    make_initial_transitions();
    make_cross_transitions(p_bigram);
}

void Graph::run_search() {
    for (auto& transition : m_initial) {
        transition.pass_token(m_emission_costs[0]);
    }

    for (int t = 0; t < m_emission_costs.size(); t++) {
        iterate(t);
    }
}

list<int> Graph::optimal_path() {
    auto node = top_rated_node();

    list<NodeId> ids;
    ids.push_front(node.node_id());

    int num_iterations = node.num_iterations();

    for (int t = num_iterations - 1; t > 0; t--) {
        NodeId node_id = node.back_link(t);
        node = m_words[node_id.model_id][node_id.node_index];
        ids.push_front(node_id);
    }

    list<int> result;

    NodeId prev;
    prev.model_id = -1;
    prev.node_index = -1;

    for (auto node_id : ids) {
        if ((node_id.model_id != prev.model_id) || (node_id.node_index != prev.node_index) ) {
            if (node_id.node_index == 0) {
                result.push_back(node_id.model_id);
            }

            prev = node_id;
        }
    }

    return result;
}

void Graph::precompute_costs(const std::vector<std::vector<double>>& pmfs) {
    for (const auto& pmf : pmfs) {
        std::vector<double> costs;

        for (auto p : pmf) {
            costs.push_back(compute_cost(p));
        }

        m_emission_costs.push_back(costs);
    }
}

void Graph::init_models(const std::vector<WordRepresentation>& dictionary) {
    for (int i = 0; i < dictionary.size(); i++) {
        auto word = dictionary[i];
        auto codes = word.as_vector();
        m_word_probabilities.push_back(word.probability());

        vector<Node> model;

        for (auto code : codes) {
            Node node(model.size(), i, code);
            model.push_back(node);
        }

        m_words.push_back(model);
    }
}

void Graph::make_initial_transitions() {
    for (auto& model : m_words) {
        Node* node = &model[0];
        double p = m_word_probabilities[node->node_id().model_id];
        double transition_cost = compute_cost(p);
        m_initial.push_back(InitialTransition(node, transition_cost));
    }
}

void Graph::make_word_level_transitions(const std::vector<std::vector<double>>& pmfs) {
    for (int i = 0; i < m_words.size(); i++) {
        for (int j = 0; j < m_words[i].size(); j++) {
            Node* src = &m_words[i][j];
            Node* dest = src;
            m_internal.push_back(InternalTransition(src, src));

            int next_index = j + 1;
            if (next_index < m_words[i].size()) {
                Node* dest = &m_words[i][next_index];
                m_internal.push_back(InternalTransition(src, dest));
            }
        }
    }
}

void Graph::make_cross_transitions(const std::map<int, TransitionRoot>& p_bigram) {
    for (auto [index_from, transition_root] : p_bigram) {
        auto children = transition_root.children();
        for (auto [index_to, p] : children) {
            auto word_src = m_words[index_from];

            Node* node_src = &m_words[index_from][word_src.size() - 1];
            Node* node_dest = &m_words[index_to][0];

            double transition_cost = compute_cost(p);
            if (transition_cost != - log(p)) {
                std::terminate();
            }
            m_crossing.push_back(CrossTransition(node_src, node_dest, transition_cost));
        }
    }
}

void Graph::iterate(int t) {
    for (auto& transition : m_internal) {
        transition.pass_token(m_emission_costs[t]);
    }

    for (auto& transition : m_crossing) {
        transition.pass_token(m_emission_costs[t]);
    }

    commit();
}

void Graph::commit() {
    for (auto& word_model : m_words) {
        for (auto& node : word_model) {
            node.commit();
        }
    }
}

Node Graph::top_rated_node() {
    double min_score = std::numeric_limits<double>::infinity();
    Node res = m_words[0][0];
    for (auto word_model : m_words) {
        for (auto node : word_model) {
            if (node.score() < min_score) {
                min_score = node.score();
                res = node;
            }
        }
    }

    return res;
}


std::vector<std::vector<double>> debug_pmf() {
    std::vector<std::vector<double>> p;
    
    std::vector<double> v;
    for (int i = 0; i < 100; i++) {
        v.push_back(0.5);
    }

    for (int i = 0; i < 100; i++) {
        p.push_back(v);
    }

    return p;
}


list<int> token_passing(const std::vector<WordRepresentation>& dictionary,
                        const std::map<int, TransitionRoot>& transitions,
                        const std::vector<std::vector<double>>& pmfs) {
    Graph graph(dictionary, transitions, pmfs);
    graph.run_search();
    return graph.optimal_path();
}

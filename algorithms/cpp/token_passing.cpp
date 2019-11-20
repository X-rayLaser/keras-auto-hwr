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

using namespace std;


struct Candidate {
    double score;
    int model_id;
    int node_index;

    Candidate() {
        this->score = std::numeric_limits<double>::infinity();
        this->model_id = -1;
        this->node_index = -1;
    }

    Candidate(double score, double model_id, double node_index) {
        this->score = score;
        this->model_id = model_id;
        this->node_index = node_index;
    }
};


struct NodeId {
    int model_id;
    int node_index;
};


class Node {
    public:
        Node(int node_id, int model_id, int code):
            m_node_id(node_id), m_model_id(model_id),
            m_code(code), m_score(std::numeric_limits<double>::infinity())
        {
        }

        void pass_token(Candidate candidate) {
            if (candidate.score < m_candidate.score) {
                m_candidate = candidate;
            }
        }

        void commit() {
            m_score = m_candidate.score;

            NodeId node_id;
            node_id.node_index = m_candidate.node_index;
            node_id.model_id = m_candidate.model_id;
            m_history.push_back(node_id);
            reset();
        }

        NodeId back_link(int t) {
            return m_history[t];
        }

        int node_id() {
            return m_node_id;
        }

        int model_id() {
            return m_model_id;
        }

        int code() {
            return m_code;
        }

        double score() {
            return m_score;
        }

        int num_iterations() {
            return m_history.size();
        }
    private:
        int m_node_id;
        int m_model_id;
        int m_code;
        double m_score;
        Candidate m_candidate;
        vector<NodeId> m_history;

        void reset() {
            m_candidate = Candidate();
        }
};


class Transition {
    public:
        virtual void pass_token(const vector<double>& emission_pmf) = 0;

        double emission_cost(double p) {
            return - log(p);
        }
};


class InternalTransition : public Transition {
    public:
        InternalTransition(Node* src, Node* dest):
            m_src(src), m_dest(dest)
        {
        }

        void pass_token(const vector<double>& emission_pmf) override {
            int code = m_dest->code();
            double p = emission_pmf[code];
            float new_score = m_src->score() + emission_cost(p);

            Candidate candidate(new_score, m_src->model_id(), m_src->node_id());
            m_dest->pass_token(candidate);
        }
    private:
        Node* m_src;
        Node* m_dest;
};


class CrossTransition : public Transition {
    public:
        CrossTransition(Node* src, Node* dest, double p_transition):
            m_src(src), m_dest(dest), m_p_transition(p_transition), m_cost(- log(m_p_transition))
        {
        }
        void pass_token(const vector<double>& emission_pmf) override {
            int code = m_dest->code();
            double p = emission_pmf[m_dest->code()];
            double new_score = m_src->score() + emission_cost(p) + m_cost;

            Candidate candidate(new_score, m_src->model_id(), m_src->node_id());
            m_dest->pass_token(candidate);
        }
    private:
        Node* m_src;
        Node* m_dest;
        double m_p_transition;
        double m_cost;
};


class InitialTransition : public Transition {
    public:
        InitialTransition(Node* dest, double p_transition):
            m_dest(dest), m_p_transition(p_transition), m_done(false)
        {
        }

        void pass_token(const vector<double>& emission_pmf) override {
            // remove duplication
            if (!m_done) {
                int code = m_dest->code();
                double p = emission_pmf[m_dest->code()];
    
                Candidate candidate;
                candidate.score = emission_cost(p) - log(m_p_transition);
                m_dest->pass_token(candidate);
                m_done = true;
            }
        }
    private:
        Node* m_dest;
        bool m_done;
        double m_p_transition;
};


class Graph {
    public:
        Graph(const std::vector<WordRepresentation>& dictionary,
              const std::map<int, TransitionRoot>& p_bigram,
              const std::vector<std::vector<double>>& pmfs): m_pmfs(pmfs)
        {
            init_models(dictionary);
            make_word_level_transitions(pmfs);
            make_initial_transitions();
            make_cross_transitions(p_bigram);
        }

        void run_search() {
            for (auto& transition : m_initial) {
                transition.pass_token(m_pmfs[0]);
            }

            for (int t = 0; t < m_pmfs.size(); t++) {
                iterate(t);
            }
        }

        list<int> optimal_path() {
            auto node = top_rated_node();

            list<int> result;
            result.push_front(node.code());

            int num_iterations = node.num_iterations();

            //NodeId node_id = node.back_link(num_iterations - 1);
            //Node node = m_words[node_id.model_id][node_id.node_index];

            for (int t = num_iterations - 1; t >= 0; t--) {
                NodeId node_id = node.back_link(t);
                node = m_words[node_id.model_id][node_id.node_index];
                result.push_front(node.code());
            }

            return result;
        }
    private:
        const std::vector<std::vector<double>>& m_pmfs;
        vector<vector<Node>> m_words;
        vector<InternalTransition> m_internal;
        vector<CrossTransition> m_crossing;
        vector<InitialTransition> m_initial;

        void init_models(const std::vector<WordRepresentation>& dictionary) {
            for (int i = 0; i < dictionary.size(); i++) {
                auto word = dictionary[i];
                auto codes = word.as_vector();

                vector<Node> model;

                for (auto code : codes) {
                    Node node(model.size(), i, code);
                    model.push_back(node);
                }

                m_words.push_back(model);
            }
        }

        void make_initial_transitions() {
            for (auto& model : m_words) {
                Node* node = &model[0];
                m_initial.push_back(InitialTransition(node, 1 / 4000.0));
            }
        }

        void make_word_level_transitions(const std::vector<std::vector<double>>& pmfs) {
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

        void make_cross_transitions(const std::map<int, TransitionRoot>& p_bigram) {
            for (auto [index_from, transition_root] : p_bigram) {
                auto children = transition_root.children();
                for (auto [index_to, p] : children) {
                    auto word_src = m_words[index_from];

                    Node* node_src = &m_words[index_from][word_src.size() - 1];
                    Node* node_dest = &m_words[index_to][0];

                    m_crossing.push_back(CrossTransition(node_src, node_dest, p));
                }
            }
        }

        void iterate(int t) {
            auto pmf = m_pmfs[t];
            
            for (auto& transition : m_internal) {
                transition.pass_token(pmf);
            }

            for (auto& transition : m_crossing) {
                transition.pass_token(pmf);
            }

            commit();
        }

        void commit() {
            for (auto& word_model : m_words) {
                for (auto& node : word_model) {
                    node.commit();
                }
            }
        }

        Node top_rated_node() {
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
};


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


int main(int argc, char *argv[])
{
    //todo: too long argument list error, try to pass arguments via socket or file
    //todo: find and fix performance bottleneck
    auto args = MySpace::parse_args(argc, argv);

    auto dictionary = get_dictionary(args.dictionary_path);

    std::map<int, TransitionRoot> transitions = get_transitions(args.bigrams_path);

    auto result = token_passing(dictionary, transitions, args.distributions);

    for (auto code : result) {
        cout << code << " ";
    }

    return 0;
}

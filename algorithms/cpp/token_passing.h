#include <vector>
#include <list>

using std::vector;
using std::list;


inline double compute_cost(double p);


struct NodeId {
    int model_id;
    int node_index;
};


struct Candidate {
    double score;
    int model_id;
    int node_index;

    Candidate();
    Candidate(double score, NodeId node_id);
};


class Node {
    public:
        Node(int node_id, int model_id, int code);

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

        NodeId node_id() {
            NodeId nid;
            nid.node_index = m_node_id;
            nid.model_id = m_model_id;
            return nid;
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
        virtual void pass_token(const vector<double>& emission_costs) = 0;
};


class InternalTransition : public Transition {
    public:
        InternalTransition(Node* src, Node* dest);

        void pass_token(const vector<double>& emission_costs) override;
    private:
        Node* m_src;
        Node* m_dest;
};


class CrossTransition : public Transition {
    public:
        CrossTransition(Node* src, Node* dest, double transition_cost);
        void pass_token(const vector<double>& emission_costs) override;
    private:
        Node* m_src;
        Node* m_dest;
        double m_cost;
};


class InitialTransition : public Transition {
    public:
        InitialTransition(Node* dest, double transition_cost);

        void pass_token(const vector<double>& emission_costs) override;
    private:
        Node* m_dest;
        bool m_done;
        double m_cost;
};


class Graph {
    public:
        Graph(const std::vector<WordRepresentation>& dictionary,
              const std::map<int, TransitionRoot>& p_bigram,
              const std::vector<std::vector<double>>& pmfs);

        void run_search();

        list<int> optimal_path();
    private:
        std::vector<std::vector<double>> m_emission_costs;
        vector<vector<Node>> m_words;
        vector<InternalTransition> m_internal;
        vector<CrossTransition> m_crossing;
        vector<InitialTransition> m_initial;
        vector<double> m_word_probabilities;

        void precompute_costs(const std::vector<std::vector<double>>& pmfs);

        void init_models(const std::vector<WordRepresentation>& dictionary);

        void make_initial_transitions();

        void make_word_level_transitions(const std::vector<std::vector<double>>& pmfs);

        void make_cross_transitions(const std::map<int, TransitionRoot>& p_bigram);

        void iterate(int t);

        void commit();

        Node top_rated_node();
};


std::vector<std::vector<double>> debug_pmf();


list<int> token_passing(const std::vector<WordRepresentation>& dictionary,
                        const std::map<int, TransitionRoot>& transitions,
                        const std::vector<std::vector<double>>& pmfs);

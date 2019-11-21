#include <vector>
#include <map>
#include <string>

 
class WordRepresentation {
  public:
    WordRepresentation(std::vector<int> v, double p);
    std::vector<int> as_vector();
    double probability();
  private:
    std::vector<int> m_v;
    double m_p;
};




class TransitionRoot {
    public:
        TransitionRoot(int src, std::map<int, double> transitions);
        TransitionRoot(const TransitionRoot& root);
        double probability(int to);
        std::vector<std::pair<int, double>> children();
    private:
        int m_from;
        std::map<int, double> m_transitions;
};


std::vector<WordRepresentation> get_dictionary(std::string file_path);

std::map<int, TransitionRoot> get_transitions(std::string file_path);

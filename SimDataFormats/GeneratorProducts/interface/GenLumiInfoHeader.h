#ifndef SimDataFormats_GeneratorProducts_GenLumiInfoHeader_h
#define SimDataFormats_GeneratorProducts_GenLumiInfoHeader_h

#include <vector>
#include <utility>
#include <string>

/** \class GenLumiInfoHeader
 *
 */

class GenLumiInfoHeader {
  public:

    GenLumiInfoHeader() : randomConfigIndex_(-1) {};

    int randomConfigIndex() const { return randomConfigIndex_; }
    void setRandomConfigIndex(int idx) { randomConfigIndex_ = idx; }
    
    const std::string &configDescription() const { return configDescription_; }
    void setConfigDescription(const std::string &str) { configDescription_ = str; }
    
    const std::vector<std::pair<std::string, std::string> > &lheHeaders() const { return lheHeaders_; }
    std::vector<std::pair<std::string, std::string> > &lheHeaders() { return lheHeaders_; }
    
    const std::vector<std::string> &weightNames() const { return weightNames_; }
    std::vector<std::string> &weightNames() { return weightNames_; }

  private:
    int randomConfigIndex_;
    std::string configDescription_;
    std::vector<std::pair<std::string, std::string> > lheHeaders_; //header name, header content
    std::vector<std::string> weightNames_;

};

#endif // SimDataFormats_GeneratorProducts_GenLumiInfoHeader_h

#ifndef SimDataFormats_GeneratorProducts_GenLumiInfoHeader_h
#define SimDataFormats_GeneratorProducts_GenLumiInfoHeader_h

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

  private:
    int randomConfigIndex_;
    std::string configDescription_;

};

#endif // SimDataFormats_GeneratorProducts_GenLumiInfoHeader_h

#ifndef MultipleScatteringX0Data_H
#define MultipleScatteringX0Data_H

/**
 *
 */

#include "TH2F.h"
#include <string>


class SumX0AtEtaDataProvider{ 
public: virtual float sumX0atEta(float eta, float r) const = 0; 
}; 

class MultipleScatteringX0Data : public SumX0AtEtaDataProvider {

public:
  MultipleScatteringX0Data();
  virtual ~MultipleScatteringX0Data();
  int nBinsEta() const;
  float minEta() const;
  float maxEta() const;
  virtual float sumX0atEta(float eta, float r) const;

private:
  std::string fileName();

  //private:
  //  HistoType * theData;
   TH2F  * theData;
};

#endif

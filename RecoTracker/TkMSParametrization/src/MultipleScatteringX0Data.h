#ifndef MultipleScatteringX0Data_H
#define MultipleScatteringX0Data_H

/**
 *
 */

#include <string>
#include <memory>
class TH2F;

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class dso_hidden SumX0AtEtaDataProvider{ 
public: virtual float sumX0atEta(float eta, float r) const = 0; 
        virtual ~SumX0AtEtaDataProvider() {}
}; 

class dso_hidden MultipleScatteringX0Data : public SumX0AtEtaDataProvider {

public:
  MultipleScatteringX0Data();
  ~MultipleScatteringX0Data() override;
  int nBinsEta() const;
  float minEta() const;
  float maxEta() const;
  float sumX0atEta(float eta, float r) const override;

private:
  std::string fileName();

  std::unique_ptr<TH2F> theData;
};

#endif

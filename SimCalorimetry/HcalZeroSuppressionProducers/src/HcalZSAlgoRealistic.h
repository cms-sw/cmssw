#ifndef SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOREALISTIC_H
#define SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOREALISTIC_H 1

#include "HcalZeroSuppressionAlgo.h"

/** \class HcalZSAlgoRealistic
  *  
  * Simple amplitude-based zero suppression algorithm.  For each digi, add
  * up consecutive 2 samples in a slice of 10 time samples, beginning with (start) 
  * sample. If any of the sums are greater then the threshold, keep the event.
  *
  * \author S. Sengupta - Minnesota
  */
class HcalZSAlgoRealistic : public HcalZeroSuppressionAlgo {
public:
  HcalZSAlgoRealistic(bool markAndPass, std::pair<int,int> HBsearchTS, std::pair<int,int> HEsearchTS, std::pair<int,int> HOsearchTS, std::pair<int,int> HFsearchTS);
  HcalZSAlgoRealistic(bool markAndPass, int levelHB, int levelHE, int levelHO, int levelHF, std::pair<int,int> HBsearchTS, std::pair<int,int> HEsearchTS, std::pair<int,int> HOsearchTS, std::pair<int,int> HFsearchTS);
  virtual ~HcalZSAlgoRealistic() = default; 
protected:
  //these need to be overloads instead of templates to avoid linking issues when calling private member function templates
  virtual bool shouldKeep(const HBHEDataFrame& digi) const;
  virtual bool shouldKeep(const HODataFrame& digi) const;
  virtual bool shouldKeep(const HFDataFrame& digi) const;
  virtual bool shouldKeep(const QIE10DataFrame& digi) const;
  virtual bool shouldKeep(const QIE11DataFrame& digi) const;
private:
  bool usingDBvalues; 
  int thresholdHB_, thresholdHE_, thresholdHO_, thresholdHF_;
  std::pair<int,int> HBsearchTS_,  HEsearchTS_, HOsearchTS_, HFsearchTS_;
  template<class Digi> bool keepMe(const Digi& inp, int start, int finish, int threshold, uint32_t zsmask) const;
};

#endif

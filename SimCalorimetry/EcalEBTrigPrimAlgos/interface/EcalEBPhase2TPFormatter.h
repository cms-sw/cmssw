#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2TPFormatter_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2TPFormatter_h

#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveSample.h"

#include <vector>
#include <cstdint>

/* 
    \class EcalEBPhase2TPFormatter

*/

class EcalEBPhase2TPFormatter {
private:
  bool debug_;
  std::vector<int> inputAmp_;
  std::vector<int64_t> inputTime_;

public:
  EcalEBPhase2TPFormatter(bool debug);
  virtual ~EcalEBPhase2TPFormatter();
  virtual void process(std::vector<int>& ampl,
                       std::vector<int64_t>& time,
                       std::vector<int>& outampl,
                       std::vector<int64_t>& outtime);
};
#endif

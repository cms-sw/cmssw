#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXODDAMPLITUDEFILTER_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXODDAMPLITUDEFILTER_H

#include <cstdint>
#include <vector>
#include <string>

class EcalTPGOddWeightIdMap;
class EcalTPGOddWeightGroup;

/**
 \ class EcalFenixOddAmplitudeFilter
 *  The purpose of this class is to implement the second (odd) ECAL FENIX amplitude filter
 *  Derived from SimCalorimetry/EcalTrigPrimAlgos/src/EcalFenixAmplitudeFilter.cc, interface/EcalFenixAmplitudeFilter.h
 *  input: 18 bits
 *  output: 18 bits
 *
 */
class EcalFenixOddAmplitudeFilter {
private:
  int peakFlag_[5];
  int inputsAlreadyIn_;
  uint32_t stripid_;
  int buffer_[5];
  int weights_[5];
  int shift_;
  bool debug_;
  bool tpInfoPrintout_;
  int setInput(int input);
  void process();

  int processedOutput_;
  int processedFgvbOutput_;

public:
  EcalFenixOddAmplitudeFilter();
  EcalFenixOddAmplitudeFilter(bool TPinfoPrintout);
  virtual ~EcalFenixOddAmplitudeFilter();
  virtual void process(std::vector<int> &addout, std::vector<int> &output);
  void setParameters(uint32_t raw,
                     const EcalTPGOddWeightIdMap *ecaltpgOddWeightMap,
                     const EcalTPGOddWeightGroup *ecaltpgOddWeightGroup);
};

#endif

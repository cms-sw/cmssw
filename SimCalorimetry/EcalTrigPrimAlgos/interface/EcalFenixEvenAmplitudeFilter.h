#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXEVENAMPLITUDEFILTER_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXEVENAMPLITUDEFILTER_H

#include <cstdint>
#include <vector>

class EcalTPGWeightIdMap;
class EcalTPGWeightGroup;

/**
 \ class EcalFeniEvenxAmplitudeFilter
 \brief calculates .... for Fenix strip, barrel
 *  input: 18 bits
 *  output: 18 bits
 *  
 * Renamed to avoid clashes with the Phase2 replicated class. (D.Valsecchi 04/2021)
 */
class EcalFenixEvenAmplitudeFilter {
private:
  int peakFlag_[5];
  int inputsAlreadyIn_;
  uint32_t stripid_;
  int buffer_[5];
  int fgvbBuffer_[5];
  int weights_[5];
  int shift_;
  int setInput(int input, int fgvb);
  void process();
  bool tpInfoPrintout_;

  int processedOutput_;
  int processedFgvbOutput_;

public:
  EcalFenixEvenAmplitudeFilter();
  EcalFenixEvenAmplitudeFilter(bool TPinfoPrintout);
  virtual ~EcalFenixEvenAmplitudeFilter();
  virtual void process(std::vector<int> &addout,
                       std::vector<int> &output,
                       std::vector<int> &fgvbIn,
                       std::vector<int> &fgvbOut);
  void setParameters(uint32_t raw,
                     const EcalTPGWeightIdMap *ecaltpgWeightMap,
                     const EcalTPGWeightGroup *ecaltpgWeightGroup);
};

#endif

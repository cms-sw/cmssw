#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXAMPLITUDEFILTER_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXAMPLITUDEFILTER_H

#include <cstdint>
#include <vector>

class EcalTPGWeightIdMap;
class EcalTPGWeightGroup;

/**
 \ class EcalFenixAmplitudeFilter
 \brief calculates .... for Fenix strip, barrel
 *  input: 18 bits
 *  output: 18 bits
 *
 */
class EcalFenixAmplitudeFilter {
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
  EcalFenixAmplitudeFilter();
  EcalFenixAmplitudeFilter(bool TPinfoPrintout);
  virtual ~EcalFenixAmplitudeFilter();
  virtual void process(std::vector<int> &addout,
                       std::vector<int> &output,
                       std::vector<int> &fgvbIn,
                       std::vector<int> &fgvbOut);
  void setParameters(uint32_t raw,
                     const EcalTPGWeightIdMap *ecaltpgWeightMap,
                     const EcalTPGWeightGroup *ecaltpgWeightGroup);
};

#endif

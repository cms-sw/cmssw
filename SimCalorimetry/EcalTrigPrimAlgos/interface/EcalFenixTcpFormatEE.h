#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXTCPFORMATEE_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXTCPFORMATEE_H

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include <vector>

class EcalTPGLutGroup;
class EcalTPGLutIdMap;
class EcalTPGTowerStatus;
class EcalTPGSpike;
class EcalTPGTPMode;

/**
    \class EcalFenixStripFormatEE
    \brief Formatting for Fenix Tcp EE
    *  input 10 bits from Ettot
    *         1 bit from fgvb
    *         3 bits TriggerTowerFlag
    *  output: 16 bits
    *  simple formatting
    *
    */
class EcalFenixTcpFormatEE {
public:
  EcalFenixTcpFormatEE(bool tccFormat, bool debug, bool famos, int binOfMax);
  virtual ~EcalFenixTcpFormatEE();
  virtual std::vector<int> process(const std::vector<int> &, const std::vector<int> &) {
    std::vector<int> v;
    return v;
  }
  void process(std::vector<int> &Et_even_sum,
               std::vector<int> &Et_odd_sum,
               std::vector<int> &fgvb,
               std::vector<int> &sfgvb,
               int eTTotShift,
               std::vector<EcalTriggerPrimitiveSample> &out,
               std::vector<EcalTriggerPrimitiveSample> &outTcc,
               bool isInInnerRings);
  void setParameters(uint32_t towid,
                     const EcalTPGLutGroup *ecaltpgLutGroup,
                     const EcalTPGLutIdMap *ecaltpgLut,
                     const EcalTPGTowerStatus *ecaltpgbadTT,
                     const EcalTPGSpike *ecaltpgSpike,
                     const EcalTPGTPMode *ecaltpgTPMode);

private:
  const unsigned int *lut_;
  const uint16_t *badTTStatus_;
  uint16_t status_;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  unsigned int binOfMax_;
  uint16_t spikeZeroThresh_;
  const EcalTPGTPMode *ecaltpgTPMode_;
};

#endif

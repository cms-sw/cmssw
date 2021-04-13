#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXTCPFORMATEB_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXTCPFORMATEB_H

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include <vector>

class EcalTPGLutGroup;
class EcalTPGLutIdMap;
class EcalTPGTowerStatus;
class EcalTPGSpike;
class EcalTPGTPMode;

/**
    \class EcalFenixStripFormat
    \brief Formatting for Fenix Tcp EB
    *  input 10 bits from Ettot
    *         1 bit from fgvb  / ODD>even flag
    *         3 bits TriggerTowerFlag
    *  output: 16 bits
    *  simple formatting
    *
    *  Using even_sum and odd_sum as inputs. Deciding the option with TPmode options
     */
class EcalFenixTcpFormatEB {
public:
  EcalFenixTcpFormatEB(bool tccFormat, bool debug, bool famos, int binOfMax);
  virtual ~EcalFenixTcpFormatEB();
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
               std::vector<EcalTriggerPrimitiveSample> &outTcc);
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

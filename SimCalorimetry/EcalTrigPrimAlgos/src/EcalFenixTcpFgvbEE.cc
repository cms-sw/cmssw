#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFgvbEE.h>
#include <iostream>

//---------------------------------------------------------------
EcalFenixTcpFgvbEE::EcalFenixTcpFgvbEE(int maxNrSamples) {
  indexLut_.resize(maxNrSamples);
}  //---------------------------------------------------------------
EcalFenixTcpFgvbEE::~EcalFenixTcpFgvbEE() {}
//---------------------------------------------------------------
void EcalFenixTcpFgvbEE::process(std::vector<std::vector<int>> &bypasslin_out,
                                 int nStr,
                                 int bitMask,
                                 std::vector<int> &output) {
  //  std::vector<int> indexLut(output.size());

  for (unsigned int i = 0; i < output.size(); i++) {
    output[i] = 0;
    indexLut_[i] = 0;
  }

  for (unsigned int i = 0; i < output.size(); i++) {
    for (int istrip = 0; istrip < nStr; istrip++) {
      int res = (bypasslin_out[istrip])[i];
      res = (res >> bitMask) & 1;  // res is FGVB at this stage
      indexLut_[i] = indexLut_[i] | (res << istrip);
    }
    indexLut_[i] = indexLut_[i] | (nStr << 5);

    int mask = 1 << indexLut_[i];
    output[i] = fgee_lut_ & mask;
    if (output[i] > 0)
      output[i] = 1;
  }
  return;
}

//-------------------------------------------------------------------

void EcalFenixTcpFgvbEE::setParameters(uint32_t towid, const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE) {
  const EcalTPGFineGrainTowerEEMap &fgee_map = ecaltpgFineGrainTowerEE->getMap();

  EcalTPGFineGrainTowerEEMapIterator it = fgee_map.find(towid);
  if (it != fgee_map.end())
    fgee_lut_ = (*it).second;
  else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGFineGrainTowerEEMap for " << towid;
}

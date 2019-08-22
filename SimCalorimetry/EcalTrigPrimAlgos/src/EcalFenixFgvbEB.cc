#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvbEB.h>

EcalFenixFgvbEB::EcalFenixFgvbEB(int maxNrSamples) { add_out_8_.resize(maxNrSamples); }

EcalFenixFgvbEB::~EcalFenixFgvbEB() {}

void EcalFenixFgvbEB::process(std::vector<int> &add_out, std::vector<int> &maxof2_out, std::vector<int> &output) {
  int Elow, Ehigh, Tlow, Thigh, lut;
  int ERatLow, ERatHigh;
  //    std::vector<int> add_out_8(add_out.size());
  int COMP3, COMP2, COMP1, COMP0;

  //  Elow = (*params_)[1024];
  //     Ehigh = (*params_)[1025];
  //     Tlow = (*params_)[1026];
  //     Thigh = (*params_)[1027];
  //     lut = (*params_)[1028];

  Elow = ETlow_;
  Ehigh = EThigh_;
  Tlow = Ratlow_;
  Thigh = Rathigh_;
  lut = lut_;

  if (Tlow > 127)
    Tlow = Tlow - 128;
  if (Thigh > 127)
    Thigh = Thigh - 128;

  for (unsigned int i = 0; i < add_out.size(); i++) {
    ERatLow = add_out[i] * Tlow >> 7;
    if (ERatLow > 0xFFF)
      ERatLow = 0xFFF;
    ERatHigh = add_out[i] * Thigh >> 7;
    if (ERatHigh > 0xFFF)
      ERatHigh = 0xFFF;
    if (add_out[i] > 0XFF)
      add_out_8_[i] = 0xFF;
    else
      add_out_8_[i] = add_out[i];

    if (maxof2_out[i] >= ERatLow)
      COMP3 = 1;
    else
      COMP3 = 0;
    if (maxof2_out[i] >= ERatHigh)
      COMP2 = 1;
    else
      COMP2 = 0;
    if (add_out_8_[i] >= Elow)
      COMP1 = 1;
    else
      COMP1 = 0;
    if (add_out_8_[i] >= Ehigh)
      COMP0 = 1;
    else
      COMP0 = 0;

    int ilut = (COMP3 << 3) + (COMP2 << 2) + (COMP1 << 1) + COMP0;
    int mask = 1 << (ilut);
    output[i] = (lut) & (mask);
    if (output[i] > 0)
      output[i] = 1;
  }
  return;
}

void EcalFenixFgvbEB::setParameters(uint32_t towid,
                                    const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                                    const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB) {
  const EcalTPGGroups::EcalTPGGroupsMap &groupmap = ecaltpgFgEBGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr it = groupmap.find(towid);
  if (it != groupmap.end()) {
    //     uint32_t fgid =(*it).second;
    //     const EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMap fgmap =
    //     ecaltpgFineGrainEB -> getMap();
    //     EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMapItr itfg =
    //     fgmap.find(fgid);
    //     (*itfg).second.getValues( ETlow_,  EThigh_,  Ratlow_,  Rathigh_,
    //     lut_);
    EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMapItr itfg = (ecaltpgFineGrainEB->getMap()).find((*it).second);
    (*itfg).second.getValues(ETlow_, EThigh_, Ratlow_, Rathigh_, lut_);
  } else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGGroupsMap entry for " << towid;
}

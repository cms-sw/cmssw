#include "SimDataFormats/ValidationFormats/interface/PHGCalValidInfo.h"

///////////////////////////////////////////////////////////////////////////////
//// PHGCalValidInfo
///////////////////////////////////////////////////////////////////////////////

void PHGCalValidInfo::fillhgcHits(const std::vector<unsigned int>& hitdets,
                                  const std::vector<unsigned int>& hitindex,
                                  const std::vector<double>& hitvtxX,
                                  const std::vector<double>& hitvtxY,
                                  const std::vector<double>& hitvtxZ) {
  for (unsigned int i = 0; i < hitvtxX.size(); i++) {
    hgcHitVtxX.push_back((float)hitvtxX.at(i));
    hgcHitVtxY.push_back((float)hitvtxY.at(i));
    hgcHitVtxZ.push_back((float)hitvtxZ.at(i));
    hgcHitDets.push_back(hitdets.at(i));
    hgcHitIndex.push_back(hitindex.at(i));
  }
}

void PHGCalValidInfo::fillhgcLayers(const double edepEE,
                                    const double edepHEF,
                                    const double edepHEB,
                                    const std::vector<double>& eedep,
                                    const std::vector<double>& hefdep,
                                    const std::vector<double>& hebdep) {
  edepEETot = (float)edepEE;
  edepHEFTot = (float)edepHEF;
  edepHEBTot = (float)edepHEB;

  for (unsigned int i = 0; i < eedep.size(); i++) {
    double en = 0.001 * eedep[i];  //GeV
    hgcEEedep.push_back((float)en);
  }

  for (unsigned int i = 0; i < hefdep.size(); i++) {
    double en = 0.001 * hefdep[i];  //GeV
    hgcHEFedep.push_back((float)en);
  }

  for (unsigned int i = 0; i < hebdep.size(); i++) {
    double en = 0.001 * hebdep[i];  //GeV
    hgcHEBedep.push_back((float)en);
  }
}

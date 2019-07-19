#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"

//
HGCalSiNoiseMap::HGCalSiNoiseMap() : encpScale_(840.), encCommonNoiseSub_(sqrt(1.25)), qe2fc_(1.60217646E-4) {
  encsParam_[q80fC] = {636., 15.6, 0.0328};
  maxADCPerGain_[q80fC] = 80.;
  encsParam_[q160fC] = {1045., 8.74, 0.0685};
  maxADCPerGain_[q160fC] = 160.;
  encsParam_[q320fC] = {1915., 2.79, 0.0878};
  maxADCPerGain_[q320fC] = 320.;

  for (auto i : maxADCPerGain_)
    lsbPerGain_[i.first] = i.second / 1024.;

  mipEqfC_[HGCSiliconDetId::waferType::HGCalFine] = 120. * 67. * qe2fc_;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalFine] = 50;
  cellVolume_[HGCSiliconDetId::waferType::HGCalFine] = 0.52 * (120.e-4);

  mipEqfC_[HGCSiliconDetId::waferType::HGCalCoarseThin] = 200. * 70. * qe2fc_;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThin] = 65;
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThin] = 1.18 * (200.e-4);

  mipEqfC_[HGCSiliconDetId::waferType::HGCalCoarseThick] = 300. * 73. * qe2fc_;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThick] = 45;
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThick] = 1.18 * (300.e-4);
}

//
HGCalSiNoiseMap::SiCellOpCharacteristics HGCalSiNoiseMap::getSiCellOpCharacteristics(const HGCSiliconDetId &cellId,
                                                                                     GainRange_t gain,
                                                                                     bool ignoreFluence,
                                                                                     int aimMIPtoADC) {
  SiCellOpCharacteristics siop;

  //decode cell properties
  int layer(cellId.layer());
  HGCSiliconDetId::waferType cellThick(HGCSiliconDetId::waferType(cellId.type()));
  double cellCap(cellCapacitance_[cellThick]);
  double cellVol(cellVolume_[cellThick]);

  //get fluence
  if (getDoseMap().empty())
    return siop;

  //leakage current and CCE [muA]
  if (ignoreFluence) {
    siop.fluence = 0;
    siop.lnfluence = -1;
    siop.ileak = exp(ileakParam_[1]) * cellVol * 1e6;
    siop.cce = 1;
  } else {
    //compute the radius here
    auto xy(ddd()->locateCell(
        cellId.layer(), cellId.waferU(), cellId.waferV(), cellId.cellU(), cellId.cellV(), true, true));
    double radius2 = std::pow(xy.first, 2) + std::pow(xy.second, 2);  //in cm

    double radius = sqrt(radius2);
    double radius3 = radius * radius2;
    double radius4 = pow(radius2, 2);
    std::array<double, 8> radii{{radius, radius2, radius3, radius4, 0., 0., 0., 0.}};
    siop.fluence = getFluenceValue(cellId.subdet(), layer, radii);
    siop.lnfluence = log(siop.fluence);
    siop.ileak = exp(ileakParam_[0] * siop.lnfluence + ileakParam_[1]) * cellVol * 1e6;

    //lin+log parametrization
    siop.cce = siop.fluence <= cceParam_[cellThick][0] ? 1. + cceParam_[cellThick][1] * siop.fluence
                                                       : (1. - cceParam_[cellThick][2] * log(siop.fluence)) +
                                                             (cceParam_[cellThick][1] * cceParam_[cellThick][0] +
                                                              cceParam_[cellThick][2] * log(cceParam_[cellThick][0]));
  }

  //determine the gain to apply accounting for cce
  double S(siop.cce * mipEqfC_[cellThick]);
  if (gain == GainRange_t::AUTO) {
    double desiredLSB(S / aimMIPtoADC);
    std::vector<double> diffToPhysLSB = {fabs(desiredLSB - lsbPerGain_[GainRange_t::q80fC]),
                                         fabs(desiredLSB - lsbPerGain_[GainRange_t::q160fC]),
                                         fabs(desiredLSB - lsbPerGain_[GainRange_t::q320fC])};
    size_t gainIdx = std::min_element(diffToPhysLSB.begin(), diffToPhysLSB.end()) - diffToPhysLSB.begin();
    gain = HGCalSiNoiseMap::q80fC;
    if (gainIdx == 1)
      gain = HGCalSiNoiseMap::q160fC;
    if (gainIdx == 2)
      gain = HGCalSiNoiseMap::q320fC;
  }

  //fill in the parameters of the struct
  siop.gain = gain;
  siop.mipfC = S;
  siop.mipADC = std::floor(S / lsbPerGain_[gain]);
  siop.thrADC = std::floor(siop.mipADC / 2);

  //build noise estimate
  double enc_s(encsParam_[gain][0] + encsParam_[gain][1] * cellCap + encsParam_[gain][2] * pow(cellCap, 2));
  double enc_p(encpScale_ * sqrt(siop.ileak));
  siop.noise = hypot(enc_p, enc_s) * encCommonNoiseSub_ * qe2fc_;

  return siop;
}

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"

//
HGCalSiNoiseMap::HGCalSiNoiseMap()
    : encpScale_(840.),
      encCommonNoiseSub_(sqrt(1.25)),
      qe2fc_(1.60217646E-4),
      ignoreFluence_(false),
      ignoreCCE_(false),
      ignoreNoise_(false) {
  encsParam_.push_back({636., 15.6, 0.0328});   // q80fC: II order polynomial coefficients
  chargeAtFullScaleADCPerGain_.push_back(80.);  //        the num of fC (charge) which corresponds to the max ADC value
  encsParam_.push_back({1045., 8.74, 0.0685});  // q160fC
  chargeAtFullScaleADCPerGain_.push_back(160.);
  encsParam_.push_back({1915., 2.79, 0.0878});  // q320fC
  chargeAtFullScaleADCPerGain_.push_back(320.);

  // adc has 10 bits -> 1024 counts at max ( >0 baseline to be handled)
  for (auto i : chargeAtFullScaleADCPerGain_)
    lsbPerGain_.push_back(i / 1024.);

  //fine sensors: 120 mum -  67: MPV of charge[number of e-]/mum for a mip in silicon; srouce PDG
  const double mipEqfC_120 = 120. * 67. * qe2fc_;
  mipEqfC_[0] = mipEqfC_120;
  const double cellCapacitance_120 = 50;
  cellCapacitance_[0] = cellCapacitance_120;
  const double cellVolume_120 = 0.52 * (120.e-4);
  cellVolume_[0] = cellVolume_120;

  //thin sensors: 200 mum
  const double mipEqfC_200 = 200. * 70. * qe2fc_;
  mipEqfC_[1] = mipEqfC_200;
  const double cellCapacitance_200 = 65;
  cellCapacitance_[1] = cellCapacitance_200;
  const double cellVolume_200 = 1.18 * (200.e-4);
  cellVolume_[1] = cellVolume_200;

  //thick sensors: 300 mum
  const double mipEqfC_300 = 300. * 73. * qe2fc_;
  mipEqfC_[2] = mipEqfC_300;
  const double cellCapacitance_300 = 45;
  cellCapacitance_[2] = cellCapacitance_300;
  const double cellVolume_300 = 1.18 * (300.e-4);
  cellVolume_[2] = cellVolume_300;
}

//
void HGCalSiNoiseMap::setDoseMap(const std::string &fullpath, const unsigned int &algo) {
  //decode bits in the algo word
  ignoreFluence_ = ((algo >> FLUENCE) & 0x1);
  ignoreCCE_ = ((algo >> CCE) & 0x1);
  ignoreNoise_ = ((algo >> NOISE) & 0x1);

  //call base class method
  HGCalRadiationMap::setDoseMap(fullpath, algo);
}

//
HGCalSiNoiseMap::SiCellOpCharacteristics HGCalSiNoiseMap::getSiCellOpCharacteristics(const HGCSiliconDetId &cellId,
                                                                                     GainRange_t gain,
                                                                                     int aimMIPtoADC) {
  SiCellOpCharacteristics siop;

  //decode cell properties
  int layer(cellId.layer());
  unsigned int cellThick = cellId.type();
  double cellCap(cellCapacitance_[cellThick]);
  double cellVol(cellVolume_[cellThick]);

  //leakage current and CCE [muA]
  if (ignoreFluence_) {
    siop.fluence = 0;
    siop.lnfluence = -1;
    siop.ileak = exp(ileakParam_[1]) * cellVol * unitToMicro_;
    siop.cce = 1;
  } else {
    if (getDoseMap().empty()) {
      throw cms::Exception("BadConfiguration")
          << " Fluence is required but no DoseMap has been passed to HGCalSiNoiseMap";
      return siop;
    }

    //compute the radius here
    auto xy(ddd()->locateCell(
        cellId.layer(), cellId.waferU(), cellId.waferV(), cellId.cellU(), cellId.cellV(), true, true));
    double radius2 = std::pow(xy.first, 2) + std::pow(xy.second, 2);  //in cm

    double radius = sqrt(radius2);
    double radius3 = radius * radius2;
    double radius4 = pow(radius2, 2);
    radiiVec radii{{radius, radius2, radius3, radius4, 0., 0., 0., 0.}};
    siop.fluence = getFluenceValue(cellId.subdet(), layer, radii);
    siop.lnfluence = log(siop.fluence);
    siop.ileak = exp(ileakParam_[0] * siop.lnfluence + ileakParam_[1]) * cellVol * unitToMicro_;

    if (ignoreCCE_) {
      siop.cce = 1.0;
    } else {
      //lin+log parametrization
      //cceParam_ are parameters as defined in equation (2) of DN-19-045
      siop.cce = siop.fluence <= cceParam_[cellThick][0] ? 1. + cceParam_[cellThick][1] * siop.fluence
                                                         : (1. - cceParam_[cellThick][2] * siop.lnfluence) +
                                                               (cceParam_[cellThick][1] * cceParam_[cellThick][0] +
                                                                cceParam_[cellThick][2] * log(cceParam_[cellThick][0]));
      siop.cce = std::max(0., siop.cce);
    }
  }

  //determine the gain to apply accounting for cce
  //move computation to ROC level (one day)
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
  siop.thrADC = std::floor(S / 2. / lsbPerGain_[gain]);

  //build noise estimate
  if (ignoreNoise_) {
    siop.noise = 0.0;
  } else {
    double enc_s(encsParam_[gain][0] + encsParam_[gain][1] * cellCap + encsParam_[gain][2] * pow(cellCap, 2));
    double enc_p(encpScale_ * sqrt(siop.ileak));
    siop.noise = hypot(enc_p, enc_s * encCommonNoiseSub_) * qe2fc_;
  }

  return siop;
}

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"

//
HGCalSiNoiseMap::HGCalSiNoiseMap()
    : defaultGain_(GainRange_t::AUTO),
      defaultAimMIPtoADC_(10),
      encCommonNoiseSub_(sqrt(1.0)),
      qe2fc_(1.60217646E-4),
      ignoreFluence_(false),
      ignoreCCE_(false),
      ignoreNoise_(false),
      ignoreGainDependentPulse_(false),
      activateCachedOp_(false) {
  //q80fC
  encsParam_.push_back({636., 15.6, 0.0328});   //2nd order polynomial coefficients as function of capacitance
  chargeAtFullScaleADCPerGain_.push_back(80.);  //the num of fC (charge) which corresponds to the max ADC value
  adcPulses_.push_back({{0., 0., 1.0, 0.066 / 0.934, 0., 0.}});  //in-time bunch is the 3rd entry in the array
  //q160fC
  encsParam_.push_back({1045., 8.74, 0.0685});
  chargeAtFullScaleADCPerGain_.push_back(160.);
  adcPulses_.push_back({{0., 0., 1.0, 0.153 / 0.847, 0., 0.}});
  // q320fC
  encsParam_.push_back({1915., 2.79, 0.0878});
  chargeAtFullScaleADCPerGain_.push_back(320.);
  adcPulses_.push_back({{0., 0., 1.0, 0.0963 / 0.9037, 0., 0.}});

  //start with a default value
  defaultADCPulse_ = adcPulses_[(GainRange_t)q160fC];

  // adc has 10 bits -> 1024 counts at max ( >0 baseline to be handled)
  for (auto i : chargeAtFullScaleADCPerGain_)
    lsbPerGain_.push_back(i / 1024.);

  //fine sensors: 120 mum -  67: MPV of charge[number of e-]/mum for a mip in silicon; srouce PDG
  const double mipEqfC_120 = 120. * 67. * qe2fc_;
  mipEqfC_[0] = mipEqfC_120;
  const double cellCapacitance_120 = 50;
  cellCapacitance_[0] = cellCapacitance_120;
  const double cellVolume_120 = 0.56 * (120.e-4);
  cellVolume_[0] = cellVolume_120;

  //thin sensors: 200 mum
  const double mipEqfC_200 = 200. * 70. * qe2fc_;
  mipEqfC_[1] = mipEqfC_200;
  const double cellCapacitance_200 = 65;
  cellCapacitance_[1] = cellCapacitance_200;
  const double cellVolume_200 = 1.26 * (200.e-4);
  cellVolume_[1] = cellVolume_200;

  //thick sensors: 300 mum
  const double mipEqfC_300 = 300. * 73. * qe2fc_;
  mipEqfC_[2] = mipEqfC_300;
  const double cellCapacitance_300 = 45;
  cellCapacitance_[2] = cellCapacitance_300;
  const double cellVolume_300 = 1.26 * (300.e-4);
  cellVolume_[2] = cellVolume_300;
}

//
void HGCalSiNoiseMap::setDoseMap(const std::string &fullpath, const unsigned int &algo) {
  //decode bits in the algo word
  ignoreFluence_ = ((algo >> FLUENCE) & 0x1);
  ignoreCCE_ = ((algo >> CCE) & 0x1);
  ignoreNoise_ = ((algo >> NOISE) & 0x1);
  ignoreGainDependentPulse_ = ((algo >> PULSEPERGAIN) & 0x1);
  activateCachedOp_ = ((algo >> CACHEDOP) & 0x1);

  //call base class method
  HGCalRadiationMap::setDoseMap(fullpath, algo);
}

//
double HGCalSiNoiseMap::getENCpad(const double &ileak) {
  if (ileak > 45.40)
    return 23.30 * ileak + 1410.04;
  else if (ileak > 38.95)
    return 30.07 * ileak + 1156.76;
  else if (ileak > 32.50)
    return 38.58 * ileak + 897.94;
  else if (ileak > 26.01)
    return 193.67 * pow(ileak, 0.70) + 21.12;
  else if (ileak > 19.59)
    return 167.60 * pow(ileak, 0.77);
  else if (ileak > 13.06)
    return 162.35 * pow(ileak, 0.82);
  else if (ileak > 6.53)
    return 202.73 * pow(ileak, 0.81);
  else
    return 457.15 * pow(ileak, 0.57);
}

//
void HGCalSiNoiseMap::setGeometry(const CaloSubdetectorGeometry *hgcGeom, GainRange_t gain, int aimMIPtoADC) {
  //call base class method
  HGCalRadiationMap::setGeometry(hgcGeom);

  defaultGain_ = gain;
  defaultAimMIPtoADC_ = aimMIPtoADC;

  //exit if cache is to be ignored
  if (!activateCachedOp_)
    return;

  //fill cache if it's not filled
  if (!siopCache_.empty())
    return;

  const std::vector<DetId> &validDetIds = geom()->getValidDetIds();
  for (auto &did : validDetIds) {
    //use only positive side detIds
    unsigned int rawId(did.rawId());
    HGCSiliconDetId hgcDetId(rawId);
    if (hgcDetId.zside() != 1)
      continue;

    //compute and store in cache
    SiCellOpCharacteristicsCore siop = getSiCellOpCharacteristicsCore(hgcDetId);
    std::pair<uint32_t, SiCellOpCharacteristicsCore> toAdd(rawId, siop);
    siopCache_.insert(toAdd);
  }
}

//
HGCalSiNoiseMap::SiCellOpCharacteristicsCore HGCalSiNoiseMap::getSiCellOpCharacteristicsCore(
    const HGCSiliconDetId &cellId, GainRange_t gain, int aimMIPtoADC) {
  //re-compute
  if (!activateCachedOp_)
    return getSiCellOpCharacteristics(cellId, gain, aimMIPtoADC).core;

  //re-use from cache
  HGCSiliconDetId posCellId(cellId.subdet(),
                            1,
                            cellId.type(),
                            cellId.layer(),
                            cellId.waferU(),
                            cellId.waferV(),
                            cellId.cellU(),
                            cellId.cellV());
  uint32_t key(posCellId.rawId());
  return siopCache_[key];
}

//
HGCalSiNoiseMap::SiCellOpCharacteristics HGCalSiNoiseMap::getSiCellOpCharacteristics(const HGCSiliconDetId &cellId,
                                                                                     GainRange_t gain,
                                                                                     int aimMIPtoADC) {
  //decode cell properties
  int layer(cellId.layer());
  unsigned int cellThick = cellId.type();
  double cellCap(cellCapacitance_[cellThick]);
  double cellVol(cellVolume_[cellThick]);
  double mipEqfC(mipEqfC_[cellThick]);

  //location of the cell
  int subdet(cellId.subdet());
  std::vector<double> &cceParam = cceParam_[cellThick];
  auto xy(
      ddd()->locateCell(cellId.layer(), cellId.waferU(), cellId.waferV(), cellId.cellU(), cellId.cellV(), true, true));
  double radius = sqrt(std::pow(xy.first, 2) + std::pow(xy.second, 2));  //in cm

  //call baseline method and add to cache
  return getSiCellOpCharacteristics(cellCap, cellVol, mipEqfC, cceParam, subdet, layer, radius, gain, aimMIPtoADC);
}

//
HGCalSiNoiseMap::SiCellOpCharacteristics HGCalSiNoiseMap::getSiCellOpCharacteristics(double &cellCap,
                                                                                     double &cellVol,
                                                                                     double &mipEqfC,
                                                                                     std::vector<double> &cceParam,
                                                                                     int &subdet,
                                                                                     int &layer,
                                                                                     double &radius,
                                                                                     GainRange_t &gain,
                                                                                     int &aimMIPtoADC) {
  SiCellOpCharacteristics siop;

  //leakage current and CCE [muA]
  if (ignoreFluence_) {
    siop.fluence = 0;
    siop.lnfluence = -1;
    siop.ileak = exp(ileakParam_[1]) * cellVol * unitToMicro_;
    siop.core.cce = 1;
  } else {
    if (getDoseMap().empty()) {
      throw cms::Exception("BadConfiguration")
          << " Fluence is required but no DoseMap has been passed to HGCalSiNoiseMap";
      return siop;
    }

    siop.lnfluence = getFluenceValue(subdet, layer, radius, true);
    siop.fluence = exp(siop.lnfluence);

    double conv(log(cellVol) + unitToMicroLog_);
    siop.ileak = exp(ileakParam_[0] * siop.lnfluence + ileakParam_[1] + conv);

    //charge collection efficiency
    if (ignoreCCE_) {
      siop.core.cce = 1.0;
    } else {
      //lin+log parametrization
      //cceParam are parameters as defined in equation (2) of DN-19-045
      siop.core.cce = siop.fluence <= cceParam[0] ? 1. + cceParam[1] * siop.fluence
                                                  : (1. - cceParam[2] * siop.lnfluence) +
                                                        (cceParam[1] * cceParam[0] + cceParam[2] * log(cceParam[0]));
      siop.core.cce = std::max((float)0., siop.core.cce);
    }
  }

  //determine the gain to apply accounting for cce
  //algo:  start with the most favored = lowest gain possible (=highest range)
  //       test for the other gains in the preferred order
  //       the first to yield <=15 ADC counts is taken
  //       this relies on the fact that these gains shift the mip peak by factors of 2
  //       in the presence of more gains 15 should be updated accordingly
  //note:  move computation to higher granularity level (ROC, trigger tower, once decided)
  double S(siop.core.cce * mipEqfC);
  if (gain == GainRange_t::AUTO) {
    gain = GainRange_t::q320fC;

    //@franzoni: i think the order needs to be this one (i.e. take the first according to preference) - tbc
    std::vector<GainRange_t> orderedGainChoice = {GainRange_t::q160fC, GainRange_t::q80fC};
    for (const auto &igain : orderedGainChoice) {
      double mipPeakADC(S / lsbPerGain_[igain]);
      if (mipPeakADC > 16)
        break;
      gain = igain;
    }

    //previous algo (kept commented for the moment)
    //    double S(siop.core.cce * mipEqfC);
    //    if (gain == GainRange_t::AUTO) {
    //      double desiredLSB(S / aimMIPtoADC);
    //      std::vector<double> diffToPhysLSB = {fabs(desiredLSB - lsbPerGain_[GainRange_t::q80fC]),
    //                                           fabs(desiredLSB - lsbPerGain_[GainRange_t::q160fC]),
    //                                           fabs(desiredLSB - lsbPerGain_[GainRange_t::q320fC])};
    //      size_t gainIdx = std::min_element(diffToPhysLSB.begin(), diffToPhysLSB.end()) - diffToPhysLSB.begin();
    //      gain = HGCalSiNoiseMap::q80fC;
    //      if (gainIdx == 1)
    //        gain = HGCalSiNoiseMap::q160fC;
    //      if (gainIdx == 2)
    //        gain = HGCalSiNoiseMap::q320fC;
    //    }
  }

  //fill in the parameters of the struct
  siop.core.gain = gain;
  siop.mipfC = S;
  siop.mipADC = std::floor(S / lsbPerGain_[gain]);
  siop.core.thrADC = std::floor(S / 2. / lsbPerGain_[gain]);

  //build noise estimate
  if (ignoreNoise_) {
    siop.core.noise = 0.0;
    siop.enc_s = 0.0;
    siop.enc_p = 0.0;
  } else {
    siop.enc_s = encsParam_[gain][0] + encsParam_[gain][1] * cellCap + encsParam_[gain][2] * pow(cellCap, 2);
    siop.enc_p = getENCpad(siop.ileak);
    siop.core.noise = hypot(siop.enc_p * encCommonNoiseSub_, siop.enc_s) * qe2fc_;
  }

  return siop;
}

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/Random.h"
#include <fstream>

//
HGCalSciNoiseMap::HGCalSciNoiseMap()
    : refEdge_(3.),
      ignoreSiPMarea_(false),
      ignoreTileArea_(false),
      ignoreDoseScale_(false),
      ignoreFluenceScale_(false),
      ignoreNoise_(false),
      refDarkCurrent_(0.5),
      aimMIPtoADC_(15),
      maxSiPMPE_(8888) {
  //number of photo electrons per MIP per scintillator type (irradiated, based on testbeam results)
  //reference is a 30*30 mm^2 tile and 2 mm^2 SiPM (with 15um pixels), at the 2 V over-voltage
  //based on https://indico.cern.ch/event/927798/contributions/3900921/attachments/2054679/3444966/2020Jun10_sn_scenes.pdf
  nPEperMIP_[CAST] = 80.31 / 2;     //cast
  nPEperMIP_[MOULDED] = 57.35 / 2;  //moulded

  //full scale charge per gain in nPE
  //this is chosen for now such that the ref. MIP peak is at N ADC counts
  //to be changed once the specs are fully defined such that the algorithm chooses the best gain
  //for the MIP peak to be close to N ADC counts (similar to Si) or applying other specific criteria
  fscADCPerGain_[GAIN_2] = nPEperMIP_[CAST] * 1024. / aimMIPtoADC_;      //2 mm^2  SiPM
  fscADCPerGain_[GAIN_4] = 2 * nPEperMIP_[CAST] * 1024. / aimMIPtoADC_;  //4mm^2   SiPM

  //lsb: adc has 10 bits -> 1024 counts at max
  for (size_t i = 0; i < GAINRANGE_N; i++)
    lsbPerGain_[i] = fscADCPerGain_[i] / 1024.f;
}

//
void HGCalSciNoiseMap::setDoseMap(const std::string& fullpath, const unsigned int algo) {
  //decode bits of the algo word
  ignoreSiPMarea_ = ((algo >> IGNORE_SIPMAREA) & 0x1);
  ignoreTileArea_ = ((algo >> IGNORE_TILEAREA) & 0x1);
  ignoreDoseScale_ = ((algo >> IGNORE_DOSESCALE) & 0x1);
  ignoreFluenceScale_ = ((algo >> IGNORE_FLUENCESCALE) & 0x1);
  ignoreNoise_ = ((algo >> IGNORE_NOISE) & 0x1);
  ignoreTileType_ = ((algo >> IGNORE_TILETYPE) & 0x1);

  //call base class method
  HGCalRadiationMap::setDoseMap(fullpath, algo);
}

//
void HGCalSciNoiseMap::setNpePerMIP(float npePerMIP) {
  nPEperMIP_[CAST] = npePerMIP;
  nPEperMIP_[MOULDED] = npePerMIP;
}

//
void HGCalSciNoiseMap::setReferenceDarkCurrent(double idark) { refDarkCurrent_ = idark; }

//
HGCalSciNoiseMap::SiPMonTileCharacteristics HGCalSciNoiseMap::scaleByDose(const HGCScintillatorDetId& cellId,
                                                                          int aimMIPtoADC,
                                                                          GainRange_t gainPreChoice) {
  //position
  GlobalPoint global = geom()->getPosition(cellId);
  double radius = std::sqrt(std::pow(global.x(), 2) + std::pow(global.y(), 2));

  int layer = cellId.layer();
  bool hasDoseMap(!(getDoseMap().empty()));
  double cellDose(0.), cellFluence(0.);

  //LIGHT YIELD
  double lyScaleFactor(1.f);
  //formula is: A = A0 * exp( -D^0.65 / 199.6)
  //where A0 is the response of the undamaged detector, D is the dose
  if (!ignoreDoseScale_ && hasDoseMap) {
    cellDose = getDoseValue(DetId::HGCalHSc, layer, radius);  //in kRad
    constexpr double expofactor = 1. / 199.6;
    const double dosespower = 0.65;
    lyScaleFactor = std::exp(-std::pow(cellDose, dosespower) * expofactor);
  }

  //NOISE
  double noise(0.f);
  if (!ignoreNoise_) {
    cellFluence = getFluenceValue(DetId::HGCalHSc, layer, radius);  //in 1-Mev-equivalent neutrons per cm2

    //MODEL 1 : formula is N = 2.18 * sqrt(F * A / 2e13)
    //where F is the fluence and A is the SiPM area (scaling with the latter is done below)
    if (refDarkCurrent_ < 0) {
      noise = 2.18;
      if (!ignoreFluenceScale_ && hasDoseMap) {
        constexpr double fluencefactor = 2. / (2 * 1e13);  //reference SiPM area = 2mm^2
        noise *= sqrt(cellFluence * fluencefactor);
      }
    }

    //MODEL 2 : formula is  3.16 *  sqrt( (Idark * 1e-12) / (qe * gain) * (F / F0) )
    //where F is the fluence (neq/cm2), gain is the SiPM gain, qe is the electron charge (C), Idark is dark current (mA)
    else {
      constexpr double refFluence(2.0E+13);
      constexpr double refGain(235000.);
      double Rdark = (refDarkCurrent_ * 1E-12) / (CLHEP::e_SI * refGain);
      if (!ignoreFluenceScale_ && hasDoseMap)
        Rdark *= (cellFluence / refFluence);
      noise = 3.16 * sqrt(Rdark);
    }
  }

  //ADDITIONAL SCALING FACTORS
  double tileAreaSF = scaleByTileArea(cellId, radius);
  std::pair<double, HGCalSciNoiseMap::GainRange_t> sipm = scaleBySipmArea(cellId, gainPreChoice);
  double sipmAreaSF = sipm.first;
  HGCalSciNoiseMap::GainRange_t gain = sipm.second;

  lyScaleFactor *= tileAreaSF * sipmAreaSF;
  noise *= sqrt(sipmAreaSF);

  //final signal depending on scintillator type
  double S(nPEperMIP_[CAST]);
  if (!ignoreTileType_ && cellId.type() == 2)
    S = nPEperMIP_[MOULDED];
  S *= lyScaleFactor;

  HGCalSciNoiseMap::SiPMonTileCharacteristics sipmChar;
  sipmChar.s = S;
  sipmChar.lySF = lyScaleFactor;
  sipmChar.n = noise;
  sipmChar.gain = gain;
  sipmChar.thrADC = std::floor(0.5 * S / lsbPerGain_[gain]);
  sipmChar.ntotalPE = maxSiPMPE_ * sipmAreaSF;
  sipmChar.xtalk = refXtalk_;
  sipmChar.lnfluence = cellFluence > 0 ? vdt::fast_log(cellFluence) : -1;
  sipmChar.lndose = cellDose > 0 ? vdt::fast_log(cellDose) : -1;
  return sipmChar;
}

//
double HGCalSciNoiseMap::scaleByTileArea(const HGCScintillatorDetId& cellId, const double radius) {
  double scaleFactor(1.f);

  if (ignoreTileArea_)
    return scaleFactor;

  [[clang::suppress]]
  double edge(refEdge_);  //start with reference 3cm of edge
  if (cellId.type() == 0) {
    constexpr double factor = 2 * M_PI * 1. / 360.;
    edge = radius * factor;  //1 degree
  } else {
    constexpr double factor = 2 * M_PI * 1. / 288.;
    edge = radius * factor;  //1.25 degrees
  }
  scaleFactor = refEdge_ / edge;
  return scaleFactor;
}

//
HGCalSciNoiseMap::SiPMonTileCharacteristics HGCalSciNoiseMap::scaleByDoseLy(const HGCScintillatorDetId& cellId,
                                                                            const double radius,
                                                                            const double edge,
                                                                            const double rawID,
                                                                            const double LYsigma,
                                                                            GainRange_t gainPreChoice) {
  int layer = cellId.layer();
  bool hasDoseMap(!(getDoseMap().empty()));

  // copied over, but not used and not updated
  //**

  //LIGHT YIELD
  double lyScaleFactor(1.f);
  //formula is: A = A0 * exp( -D^0.65 / 199.6)
  //where A0 is the response of the undamaged detector, D is the dose
  if (!ignoreDoseScale_ && hasDoseMap) {
    double cellDose = getDoseValue(DetId::HGCalHSc, layer, radius);  //in kRad
    constexpr double expofactor = 1. / 199.6;
    const double dosespower = 0.65;
    lyScaleFactor = std::exp(-std::pow(cellDose, dosespower) * expofactor);
  }

  //NOISE
  double noise(0.f);
  if (!ignoreNoise_) {
    double cellFluence = getFluenceValue(DetId::HGCalHSc, layer, radius);  //in 1-Mev-equivalent neutrons per cm2

    //MODEL 1 : formula is N = 2.18 * sqrt(F * A / 2e13)
    //where F is the fluence and A is the SiPM area (scaling with the latter is done below)
    if (refDarkCurrent_ < 0) {
      noise = 2.18;
      if (!ignoreFluenceScale_ && hasDoseMap) {
        constexpr double fluencefactor = 2. / (2 * 1e13);  //reference SiPM area = 2mm^2
        noise *= sqrt(cellFluence * fluencefactor);
      }
    }

    //MODEL 2 : formula is  3.16 *  sqrt( (Idark * 1e-12) / (qe * gain) * (F / F0) )
    //where F is the fluence (neq/cm2), gain is the SiPM gain, qe is the electron charge (C), Idark is dark current (mA)
    else {
      constexpr double refFluence(2.0E+13);
      constexpr double refGain(235000.);
      double Rdark = (refDarkCurrent_ * 1E-12) / (CLHEP::e_SI * refGain);
      if (!ignoreFluenceScale_ && hasDoseMap)
        Rdark *= (cellFluence / refFluence);
      noise = 3.16 * sqrt(Rdark);
    }
  }

  //ADDITIONAL SCALING FACTORS
  std::pair<double, HGCalSciNoiseMap::GainRange_t> sipm = scaleBySipmArea(cellId, gainPreChoice);
  double sipmAreaSF = sipm.first;
  HGCalSciNoiseMap::GainRange_t gain = sipm.second;

  // LY scale factor
  CLHEP::HepRandom::setTheSeed(
      rawID);  // seeded RNG with tile id as the seed, to be used when a mean value for LY for a given tile size is provided;
  // this allows to generate a unique LY for all tiles based on measured mean LY and standard deviation, constant for all events
  double var = CLHEP::RandGauss::shoot(1, LYsigma);
  // using the calculated tile edge to calculate the mean value for the light yield based on parametrisation for various materials
  // Formulas for 9 mm^2 SiPMs, and SiPM overvoltage of 2V
  // lyVsize(tile type (1==cast, 2 == molded), "phase" = 10 or 0 (pre-series or production), tile edge, sipm area (2 or 4 or 9 [mm^2]), overvoltage (e.g. 4 [V]))
  double ly(lyVsize(cellId.type(), 0, edge, 9, 2));
  // returning the mean value for a given tile size multiplied by the random modifier dependant on the given tile id = LY of that specific tile
  lyScaleFactor *= ly * var;
  noise *= sqrt(sipmAreaSF);

  //final signal depending on scintillator type
  double S(nPEperMIP_[CAST]);
  if (!ignoreTileType_ && cellId.type() == 2)
    S = nPEperMIP_[MOULDED];
  S *= lyScaleFactor;
  //**

  HGCalSciNoiseMap::SiPMonTileCharacteristics sipmChar;
  sipmChar.s = S;  //unused in the digitiser
  sipmChar.lySF = lyScaleFactor;
  sipmChar.n = noise;                                         //unused in the digitiser
  sipmChar.gain = gain;                                       //unused in the digitiser
  sipmChar.thrADC = std::floor(0.5 * S / lsbPerGain_[gain]);  //unused in the digitiser, set elsewhere
  sipmChar.ntotalPE = maxSiPMPE_ * sipmAreaSF;                //unused in the digitiser
  sipmChar.xtalk = refXtalk_;                                 //unused in the digitiser
  return sipmChar;
}

// SiPM Photo Detection Efficiency (PDE) function. The LY is dependant on overvoltage through PDE, hense needed for calculation if a different OV is used (HGCROC settings)

float pde(float ov) {
  float pdei;
  pdei = (-0.72 * ov * ov) + (10.0 * ov) + 4.41;
  return pdei;
}

// getting tile light yield based on its edge length, LY v edge as formulas, parametrisation from production measurements in DESY's LY test stand

float HGCalSciNoiseMap::lyVsize(int type, int phase, float edge, float sipm = 4.0, float ov = 4.0) {
  //type = Cast or Molded; Molded = 2, Cast = 1 (??) // Should be like that in CMSSW
  //phase = 10 = pre-series, 0 = production // Daria's addition to create a single map for the function parameters

  // Values A and C parameters for LY function for production tiles if 2 and 1, 12 and 11 if pre-series (by adding 10):
  const std::map<int, std::vector<float>>& LY_params{
      {2, {2066.53, -7.6}}, {1, {1378.53, 32.34}}, {12, {1854.5, -6.52}}, {11, {1110.5, 54.8}}};

  float LYi, sipmfactor, ovfactor, A, C;
  int key;
  sipmfactor = sipm / 4;        // ratio between SiPM areas in mm^2, nominal is 4 mm^2
  ovfactor = pde(ov) / pde(4);  // ratio of PDE at a given overvoltage and nominal (4V)

  key = type + phase;
  const std::vector<float>& Z = LY_params.at(key);
  A = Z[0];
  C = Z[1];
  LYi = ((A / edge) + C) * sipmfactor * ovfactor;
  //std::cout << "A: " << A << " C: " << C << std::endl;

  return LYi;
}

std::pair<double, HGCalSciNoiseMap::GainRange_t> HGCalSciNoiseMap::scaleBySipmArea(
    const HGCScintillatorDetId& cellId, const HGCalSciNoiseMap::GainRange_t& gainPreChoice) {
  //start with the prechosen gain
  //if auto then override it according to the SiPM area
  HGCalSciNoiseMap::GainRange_t gain(gainPreChoice);
  if (gainPreChoice == HGCalSciNoiseMap::GainRange_t::AUTO)
    gain = GainRange_t::GAIN_2;

  double scaleFactor(1.f);

  if (!ignoreSiPMarea_) {
    int sipm = cellId.sipm();
    if (sipm == 0) {
      scaleFactor = 2.12f;
      if (gainPreChoice == HGCalSciNoiseMap::GainRange_t::AUTO)
        gain = GainRange_t::GAIN_4;
    }
  }

  return std::pair<double, HGCalSciNoiseMap::GainRange_t>(scaleFactor, gain);
}

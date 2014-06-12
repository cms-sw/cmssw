#ifndef GEMDigitizer_GEMSimpleModel_h
#define GEMDigitizer_GEMSimpleModel_h

/**
* \class GEMSimpleModel
*
* Class for the GEM strip response simulation based on a very simple model
*
* \author Sven Dildick
* \modified by Roumyana Hadjiiska
*/

#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

class GEMGeometry;

namespace CLHEP
{
  class HepRandomEngine;
  class RandFlat;
  class RandPoissonQ;
  class RandGaussQ;
  class RandGamma;
  class RandBinomial;
}

class GEMSimpleModel: public GEMDigiModel
{
public:

  GEMSimpleModel(const edm::ParameterSet&);

  ~GEMSimpleModel();

  void setRandomEngine(CLHEP::HepRandomEngine&);

  void setup();

  void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&);

  int getSimHitBx(const PSimHit*);

  void simulateNoise(const GEMEtaPartition*);

  std::vector<std::pair<int,int> >
    simulateClustering(const GEMEtaPartition*, const PSimHit*, const int);

private:

  double averageEfficiency_;
  double averageShapingTime_;
  double timeResolution_;
  double timeJitter_;
  double averageNoiseRate_;
// double averageClusterSize_;
  std::vector<double> clsParametrization_;
  double signalPropagationSpeed_;
  bool cosmics_;
  int bxwidth_;
  int minBunch_;
  int maxBunch_;
  bool digitizeOnlyMuons_;
  bool doBkgNoise_;
  bool doNoiseCLS_;
  bool fixedRollRadius_;
  double minPabsNoiseCLS_;
  bool simulateIntrinsicNoise_;
  bool simulateElectronBkg_;
  bool simulateLowNeutralRate_;

  CLHEP::RandFlat* flat1_;
  CLHEP::RandFlat* flat2_;
  CLHEP::RandFlat* flat3_;
  CLHEP::RandFlat* flat4_;
  CLHEP::RandPoissonQ* poisson_;
  CLHEP::RandGaussQ* gauss1_;
  CLHEP::RandGaussQ* gauss2_;
  CLHEP::RandGamma* gamma1_;

//parameters from the fit:
//params for pol3 model of electron bkg for GE1/1:
  double GE11ElecBkgParam0;
  double GE11ElecBkgParam1;
  double GE11ElecBkgParam2;
  double GE11ElecBkgParam3;
//params for expo of electron bkg for GE2/1:
  double constElecGE21;
  double slopeElecGE21;

//Neutral Bkg
//Low Rate model L=10^{34}cm^{-2}s^{-1}
//const and slope for expo model of neutral bkg for GE1/1:
  double constNeuGE11;
  double slopeNeuGE11;
//params for pol5 model of neutral bkg for GE2/1:
  double GE21NeuBkgParam0;
  double GE21NeuBkgParam1;
  double GE21NeuBkgParam2;
  double GE21NeuBkgParam3;
  double GE21NeuBkgParam4;
  double GE21NeuBkgParam5;

//High Rate model L=5x10^{34}cm^{-2}s^{-1}
//params for expo model of neutral bkg for GE1/1:
  double constNeuGE11_highRate;
  double slopeNeuGE11_highRate;
//params for pol5 model of neutral bkg for GE2/1:
  double GE21ModNeuBkgParam0;
  double GE21ModNeuBkgParam1;
  double GE21ModNeuBkgParam2;
  double GE21ModNeuBkgParam3;
  double GE21ModNeuBkgParam4;
  double GE21ModNeuBkgParam5;

};
#endif



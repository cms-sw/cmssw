#ifndef SimMuon_GEMDigitizer_GEMSimpleModel_h
#define SimMuon_GEMDigitizer_GEMSimpleModel_h

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
}

class GEMSimpleModel: public GEMDigiModel
{
public:

  GEMSimpleModel(const edm::ParameterSet&);

  ~GEMSimpleModel();

  void setup();

  void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*) override;

  int getSimHitBx(const PSimHit*, CLHEP::HepRandomEngine*);

  void simulateNoise(const GEMEtaPartition*, CLHEP::HepRandomEngine*) override;

  std::vector<std::pair<int,int> > 
    simulateClustering(const GEMEtaPartition*, const PSimHit*, const int, CLHEP::HepRandomEngine*) override;

private:

  double averageEfficiency_;
  double averageShapingTime_;
  double timeResolution_;
  double timeJitter_;
  double averageNoiseRate_;
  double signalPropagationSpeed_;
  bool cosmics_;
  int bxwidth_;
  int minBunch_;
  int maxBunch_;
  bool digitizeOnlyMuons_;
  bool doBkgNoise_;
  bool doNoiseCLS_;
  bool fixedRollRadius_;
  bool simulateIntrinsicNoise_;
  bool simulateElectronBkg_;
  bool simulateLowNeutralRate_;
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



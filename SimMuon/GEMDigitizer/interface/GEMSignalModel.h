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

  ~GEMSimpleModel() override;

  void setup() override;

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
  int bxwidth_;
  int minBunch_;
  int maxBunch_;
  bool digitizeOnlyMuons_;
  bool doBkgNoise_;
  bool doNoiseCLS_;
  bool fixedRollRadius_;
  bool simulateIntrinsicNoise_;
  bool simulateElectronBkg_;
  double instLumi_;
  double rateFact_;
  const double referenceInstLumi_; 
  double resolutionX_; 

  //params for pol3 model of electron bkg for GE1/1 and GE2/1:
  double GE11ElecBkgParam0_;
  double GE11ElecBkgParam1_;
  double GE11ElecBkgParam2_;
  double GE21ElecBkgParam0_;
  double GE21ElecBkgParam1_;
  double GE21ElecBkgParam2_;
  //params for pol3 model of neutral bkg for GE1/1 and GE2/1:
  double GE11ModNeuBkgParam0_;
  double GE11ModNeuBkgParam1_;
  double GE11ModNeuBkgParam2_;
  double GE21ModNeuBkgParam0_;
  double GE21ModNeuBkgParam1_;
  double GE21ModNeuBkgParam2_;
    
};
#endif



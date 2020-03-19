#ifndef SimMuon_GEMDigitizer_ME0SimpleModel_h
#define SimMuon_GEMDigitizer_ME0SimpleModel_h

/** 
 * \class ME0SimpleModel
 *
 * Class for the ME0 strip response simulation based on a very simple model
 *
 * \author Sven Dildick
 * \modified by Roumyana Hadjiiska
 */

#include <vector>
#include "SimMuon/GEMDigitizer/interface/ME0DigiModel.h"

class ME0Geometry;

namespace CLHEP {
  class HepRandomEngine;
}

class ME0SimpleModel : public ME0DigiModel {
public:
  ME0SimpleModel(const edm::ParameterSet&);

  ~ME0SimpleModel() override;

  void setup() override;

  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*) override;

  int getSimHitBx(const PSimHit*, CLHEP::HepRandomEngine*);

  void simulateNoise(const ME0EtaPartition*, CLHEP::HepRandomEngine*) override;

  std::vector<std::pair<int, int> > simulateClustering(const ME0EtaPartition*,
                                                       const PSimHit*,
                                                       const int,
                                                       CLHEP::HepRandomEngine*) override;

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
  double referenceInstLumi_;
  //params for charged background model for ME0
  double ME0ElecBkgParam0_;
  double ME0ElecBkgParam1_;
  double ME0ElecBkgParam2_;
  double ME0ElecBkgParam3_;
  //params for neutral background model for ME0
  double ME0NeuBkgParam0_;
  double ME0NeuBkgParam1_;
  double ME0NeuBkgParam2_;
  double ME0NeuBkgParam3_;
};
#endif

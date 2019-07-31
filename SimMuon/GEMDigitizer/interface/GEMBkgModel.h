#ifndef SimMuon_GEMDigitizer_GEMBkgModel_h
#define SimMuon_GEMDigitizer_GEMBkgModel_h

/** 
 * \class GEMBkgModel
 *
 * Class for the GEM strip response to background simulation based on a very simple model
 * Originally comes from GEMSimpleModel
 *
 * \author Sven Dildick
 * \modified by Roumyana Hadjiiska
 * \splitted by Yechan Kang
 */

#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

class GEMGeometry;

namespace CLHEP {
  class HepRandomEngine;
}

class GEMBkgModel : public GEMDigiModel {
public:
  GEMBkgModel(const edm::ParameterSet&);

  ~GEMBkgModel() override;

  void simulate(
      const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*, Strips&, DetectorHitMap&) override;

private:
  const double clusterSizeCut;
  double averageEfficiency_;
  int minBunch_;
  int maxBunch_;
  bool digitizeOnlyMuons_;
  bool simulateNoiseCLS_;
  bool fixedRollRadius_;
  bool simulateElectronBkg_;
  double instLumi_;
  double rateFact_;
  double bxWidth_;
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

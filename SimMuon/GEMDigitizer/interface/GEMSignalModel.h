#ifndef SimMuon_GEMDigitizer_GEMSignalModel_h
#define SimMuon_GEMDigitizer_GEMSignalModel_h

/** 
 * \class GEMSignalModel
 *
 * Class for the GEM strip response simulation based on a very simple model
 * Originally comes from GEMSimpleModel
 *
 * \author Sven Dildick
 * \modified by Roumyana Hadjiiska
 * \splitted by Yechan Kang
 */

#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

class GEMGeometry;
class TrapezoidalStripTopology;

namespace CLHEP {
  class HepRandomEngine;
}

class GEMSignalModel : public GEMDigiModel {
public:
  GEMSignalModel(const edm::ParameterSet&);

  ~GEMSignalModel() override;

  void simulate(
      const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*, Strips&, DetectorHitMap&) override;

  int getSimHitBx(const PSimHit*, CLHEP::HepRandomEngine*);

  std::vector<std::pair<int, int> > simulateClustering(const TrapezoidalStripTopology*,
                                                       const PSimHit*,
                                                       const int,
                                                       CLHEP::HepRandomEngine*);

private:
  double averageEfficiency_;
  double averageShapingTime_;
  double timeResolution_;
  double timeJitter_;
  double signalPropagationSpeed_;
  bool digitizeOnlyMuons_;
  double resolutionX_;

  const int muonPdgId = 13;
  const double momConvFact = 1000.;
  const double elecMomCut1 = 1.96e-03;
  const double elecMomCut2 = 10.e-03;
  const double elecEffLowCoeff = 1.7e-05;
  const double elecEffLowParam0 = 2.1;
  const double elecEffMidCoeff = 1.34;
  const double elecEffMidParam0 = -5.75e-01;
  const double elecEffMidParam1 = 7.96e-01;
};
#endif

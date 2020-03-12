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

  const int muonPdgId;
  const double cspeed;
  const double momConvFact;
  const double elecMomCut1;
  const double elecMomCut2;
  const double elecEffLowCoeff;
  const double elecEffLowParam0;
  const double elecEffMidCoeff;
  const double elecEffMidParam0;
  const double elecEffMidParam1;
};
#endif

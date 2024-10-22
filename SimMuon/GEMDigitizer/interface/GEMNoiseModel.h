#ifndef SimMuon_GEMDigitizer_GEMNoiseModel_h
#define SimMuon_GEMDigitizer_GEMNoiseModel_h

/** 
 * \class GEMSignalModel
 *
 * Class for the GEM strip intrinsic noise simulation based on a very simple model
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

class GEMNoiseModel : public GEMDigiModel {
public:
  GEMNoiseModel(const edm::ParameterSet&);

  ~GEMNoiseModel() override;

  void simulate(
      const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*, Strips&, DetectorHitMap&) override;

private:
  double averageNoiseRate_;
  double bxWidth_;
  int minBunch_;
  int maxBunch_;
};
#endif

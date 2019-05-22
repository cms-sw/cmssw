#ifndef SimMuon_GEMDigitizer_GEMTrivialModel_h
#define SimMuon_GEMDigitizer_GEMTrivialModel_h

/**
 * \class GEMTrivialModel
 *
 * Class for the GEM strip response simulation based on a trivial model
 *
 * \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

class GEMGeometry;

class GEMTrivialModel : public GEMDigiModel {
public:
  GEMTrivialModel(const edm::ParameterSet&);

  ~GEMTrivialModel() override {}

  void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine* engine) override;

  void simulateNoise(const GEMEtaPartition*, CLHEP::HepRandomEngine* engine) override {}

  std::vector<std::pair<int, int> > simulateClustering(const GEMEtaPartition*,
                                                       const PSimHit*,
                                                       const int,
                                                       CLHEP::HepRandomEngine* engine) override;

  void setup() override {}

private:
};
#endif

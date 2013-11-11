#ifndef GEMDigitizer_GEMTrivialModel_h
#define GEMDigitizer_GEMTrivialModel_h

/**
 * \class GEMTrivialModel
 *
 * Class for the GEM strip response simulation based on a trivial model
 *
 * \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

class GEMGeometry;

class GEMTrivialModel: public GEMDigiModel
{
public:

  GEMTrivialModel(const edm::ParameterSet&);

  ~GEMTrivialModel() {}

  void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&);

  void simulateNoise(const GEMEtaPartition*) {}

  std::vector<std::pair<int,int> > 
    simulateClustering(const GEMEtaPartition*, const PSimHit*, const int);

  void setRandomEngine(CLHEP::HepRandomEngine&) {}

  void setup() {}

private:
};
#endif

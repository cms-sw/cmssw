#ifndef ME0Digitizer_ME0TrivialModel_h
#define ME0Digitizer_ME0TrivialModel_h

/**
 * \class ME0TrivialModel
 *
 * Class for the ME0 strip response simulation based on a trivial model
 *
 * \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/interface/ME0DigiModel.h"

class ME0Geometry;

class ME0TrivialModel: public ME0DigiModel
{
public:

  ME0TrivialModel(const edm::ParameterSet&);

  ~ME0TrivialModel() {}

  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&);

  void simulateNoise(const ME0EtaPartition*) {}

  std::vector<std::pair<int,int> > 
    simulateClustering(const ME0EtaPartition*, const PSimHit*, const int);

  void setRandomEngine(CLHEP::HepRandomEngine&) {}

  void setup() {}

private:
};
#endif

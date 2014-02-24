#ifndef GEMDigitizer_ME0PreRecoNoSmearModel_h
#define GEMDigitizer_ME0PreRecoNoSmearModel_h

/**
 * \class ME0PreRecoNoSmearModel
 *
 * Class for the ME0 NoSmear response simulation as pre-reco step 
 */

#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModel.h"

class ME0Geometry;

class ME0PreRecoNoSmearModel: public ME0DigiPreRecoModel
{
public:

  ME0PreRecoNoSmearModel(const edm::ParameterSet&);

  ~ME0PreRecoNoSmearModel() {}

  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&);

  void setRandomEngine(CLHEP::HepRandomEngine&) {}

  void setup() {}

private:
};
#endif

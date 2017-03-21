#ifndef SimMuon_GEMDigitizer_ME0PreRecoNoSmearModel_h
#define SimMuon_GEMDigitizer_ME0PreRecoNoSmearModel_h

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

  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*) override;

  void simulateNoise(const ME0EtaPartition*, CLHEP::HepRandomEngine*) override;

  void setup() override {}

private:
};
#endif

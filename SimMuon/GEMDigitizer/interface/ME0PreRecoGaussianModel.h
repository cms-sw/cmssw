#ifndef GEMDigitizer_ME0PreRecoGaussianModel_h
#define GEMDigitizer_ME0PreRecoGaussianModel_h

/**
 * \class ME0PreRecoGaussianModel
 *
 * Class for the ME0 Gaussian response simulation as pre-reco step 
 */

#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModel.h"

class ME0Geometry;
namespace CLHEP
{
  class HepRandomEngine;
  class RandGaussQ;
}

class ME0PreRecoGaussianModel: public ME0DigiPreRecoModel
{
public:

  ME0PreRecoGaussianModel(const edm::ParameterSet&);

  ~ME0PreRecoGaussianModel();

  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&);

  void setRandomEngine(CLHEP::HepRandomEngine&);

  void setup() {}

private:
  double sigma_t;
  double sigma_u;
  double sigma_v;
  bool corr;
  bool etaproj;
  CLHEP::RandGaussQ* gauss_;
};
#endif

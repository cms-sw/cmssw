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
  class RandFlat;
  class RandGaussQ;
  class RandPoissonQ;
}
class ME0PreRecoGaussianModel: public ME0DigiPreRecoModel
{
public:
  ME0PreRecoGaussianModel(const edm::ParameterSet&);
  ~ME0PreRecoGaussianModel();
  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&);
  void setRandomEngine(CLHEP::HepRandomEngine&);
  void simulateNoise(const ME0EtaPartition*);
  void setup()
  {
  }
private:
  double sigma_t;
  double sigma_u;
  double sigma_v;
  bool corr;
  bool etaproj;
  bool digitizeOnlyMuons_;
  double gaussianSmearing_;
  double averageEfficiency_;
  // bool simulateIntrinsicNoise_; // not implemented
  // double averageNoiseRate_;     // not implemented
  bool simulateElectronBkg_;
  bool simulateNeutralBkg_;
  int minBunch_;
  int maxBunch_;
  CLHEP::RandGaussQ* gauss_;
  CLHEP::RandFlat* flat1_;
  CLHEP::RandFlat* flat2_;
  CLHEP::RandFlat* poisson_;
  // params for the simple pol6 model of neutral bkg for ME0:
  std::vector<double> neuBkg, eleBkg;
};
#endif


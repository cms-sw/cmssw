#ifndef SimMuon_GEMDigitizer_ME0PreRecoGaussianModel_h
#define SimMuon_GEMDigitizer_ME0PreRecoGaussianModel_h

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
}

class ME0PreRecoGaussianModel: public ME0DigiPreRecoModel
{
public:

  ME0PreRecoGaussianModel(const edm::ParameterSet&);

  ~ME0PreRecoGaussianModel();

  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*) override;

  void simulateNoise(const ME0EtaPartition*, CLHEP::HepRandomEngine*) override;

  void setup() {}

private:
  double sigma_t;
  double sigma_u;
  double sigma_v;
  bool corr;
  bool etaproj;
  bool digitizeOnlyMuons_;
  double averageEfficiency_;
  bool doBkgNoise_;
  bool simulateIntrinsicNoise_;
  bool simulateElectronBkg_;

  double averageNoiseRate_;
  int bxwidth_;
  int minBunch_;
  int maxBunch_;

  //params for the simple pol6 model of neutral bkg for ME0:
  double ME0ModNeuBkgParam0;
  double ME0ModNeuBkgParam1;
  double ME0ModNeuBkgParam2;
  double ME0ModNeuBkgParam3;
  double ME0ModNeuBkgParam4;
  double ME0ModNeuBkgParam5;
  double ME0ModNeuBkgParam6;

  double ME0ModElecBkgParam0;
  double ME0ModElecBkgParam1;
  double ME0ModElecBkgParam2;
  double ME0ModElecBkgParam3;
  double ME0ModElecBkgParam4;
  double ME0ModElecBkgParam5;
  double ME0ModElecBkgParam6;
  double ME0ModElecBkgParam7;

};
#endif

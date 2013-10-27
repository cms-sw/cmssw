#ifndef GEMDigitizer_GEMSimpleAnalyzeInFlightModel_h
#define GEMDigitizer_GEMSimpleAnalyzeInFlightModel_h

/** 
 * \class GEMSimpleAnalyzeInFlightModel
 *
 * Class for the GEM strip response simulation based on a very simple model
 *
 * \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"


class GEMGeometry;

namespace CLHEP
{
  class HepRandomEngine;
  class RandFlat;
  class RandPoissonQ;
  class RandGaussQ;
  class RandGamma;
  class RandLandau;
}

class GEMSimpleAnalyzeInFlightModel: public GEMDigiModel
{
public:

  GEMSimpleAnalyzeInFlightModel(const edm::ParameterSet&);

  ~GEMSimpleAnalyzeInFlightModel();

  void setRandomEngine(CLHEP::HepRandomEngine&);

  void setup();

  void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&);

  int getSimHitBx(const PSimHit*);

  void simulateNoise(const GEMEtaPartition*);

  std::vector<std::pair<int,int> > 
    simulateClustering(const GEMEtaPartition*, const PSimHit*, const int);

private:

  double averageEfficiency_;
  double averageShapingTime_;
  double timeResolution_;
  double timeJitter_;
  double timeCalibrationOffset_;
  double averageNoiseRate_;
  double averageClusterSize_;
  double signalPropagationSpeed_;
  bool cosmics_;
  int bxwidth_;
  int minBunch_;
  int maxBunch_;
  bool digitizeOnlyMuons_;
  double cutElecMomentum_;
  int cutForCls_;

  CLHEP::RandFlat* flat1_;
  CLHEP::RandFlat* flat2_;
  CLHEP::RandPoissonQ* poisson_;
  CLHEP::RandGaussQ* gauss1_;
  CLHEP::RandGaussQ* gauss2_;
  CLHEP::RandGamma* gamma1_;
  CLHEP::RandLandau* landau1_;

  TH1F *particleId_h;
  TH1F *energyLoss_el;
  TH1F *energyLoss_mu;
  TH1F *tof_el;
  TH1F *tof_mu;
  TH1F *pabs_el;
  TH1F *pabs_mu;
  TH1F *process_el;
  TH1F *process_mu;
  TH1F *cls_el;
  TH1F *cls_mu;
  TH1F *cls_all;
  TH1F *res_mu;
  TH1F *res_el;
  TH1F *res_all;
  TH1F *bx_h;
  TH1F *numbDigis;
  TH1F *poisHisto;
  TH1F *noisyBX;
  TH1F *bx_final;
  TH1F *stripProfile;

  std::vector<PSimHit> *selPsimHits;

  double  neutronGammaRoll1_;
  double  neutronGammaRoll2_;
  double  neutronGammaRoll3_;
  double  neutronGammaRoll4_;
  double  neutronGammaRoll5_;
  double  neutronGammaRoll6_;
  double  neutronGammaRoll7_;
  double  neutronGammaRoll8_;

  TH1F *res_mu1;
  TH1F *res_mu8;


};
#endif

#ifndef ME0Digitizer_ME0SimpleModel_h
#define ME0Digitizer_ME0SimpleModel_h

/** 
 * \class ME0SimpleModel
 *
 * Class for the ME0 strip response simulation based on a very simple model
 *
 * \author Sven Dildick
 * \modified by Roumyana Hadjiiska
 */

#include "SimMuon/GEMDigitizer/interface/ME0DigiModel.h"

class ME0Geometry;
namespace CLHEP
{
  class HepRandomEngine;
  class RandFlat;
  class RandPoissonQ;
  class RandGaussQ;
  class RandGamma;
}

class ME0SimpleModel: public ME0DigiModel
{
public:

  ME0SimpleModel(const edm::ParameterSet&);

  ~ME0SimpleModel();

  void setRandomEngine(CLHEP::HepRandomEngine&);

  void setup();

  void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&);

  int getSimHitBx(const PSimHit*);

  void simulateNoise(const ME0EtaPartition*);

  std::vector<std::pair<int,int> > 
    simulateClustering(const ME0EtaPartition*, const PSimHit*, const int);

private:

  double averageEfficiency_;
  double averageShapingTime_;
  double timeResolution_;
  double timeJitter_;
  double timeCalibrationOffset1_;
  double timeCalibrationOffset23_;
  double averageNoiseRate_;
  double averageClusterSize_;
  double signalPropagationSpeed_;
  bool cosmics_;
  int bxwidth_;
  int minBunch_;
  int maxBunch_;
  bool digitizeOnlyMuons_;
  std::vector<double> neutronGammaRoll1_;
  std::vector<double> neutronGammaRoll2_;
  std::vector<double> neutronGammaRoll3_;
  bool doNoiseCLS_;
  double minPabsNoiseCLS_;
  bool simulateIntrinsicNoise_;
  double scaleLumi_;

  CLHEP::RandFlat* flat1_;
  CLHEP::RandFlat* flat2_;
  CLHEP::RandPoissonQ* poisson_;
  CLHEP::RandGaussQ* gauss1_;
  CLHEP::RandGaussQ* gauss2_;
  CLHEP::RandGamma* gamma1_;

};
#endif



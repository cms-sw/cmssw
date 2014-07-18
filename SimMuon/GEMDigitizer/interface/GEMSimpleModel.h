#ifndef GEMDigitizer_GEMSimpleModel_h
#define GEMDigitizer_GEMSimpleModel_h

/** 
 * \class GEMSimpleModel
 *
 * Class for the GEM strip response simulation based on a very simple model
 *
 * \author Sven Dildick
 * \modified by Roumyana Hadjiiska
 */

#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

class GEMGeometry;

namespace CLHEP
{
  class HepRandomEngine;
}

class GEMSimpleModel: public GEMDigiModel
{
public:

  GEMSimpleModel(const edm::ParameterSet&);

  ~GEMSimpleModel();

  void setup();

  void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine* engine) override;

  int getSimHitBx(const PSimHit*, CLHEP::HepRandomEngine* engine);

  void simulateNoise(const GEMEtaPartition*, CLHEP::HepRandomEngine* engine) override;

  std::vector<std::pair<int,int> > 
    simulateClustering(const GEMEtaPartition*, const PSimHit*, const int, CLHEP::HepRandomEngine* engine) override;

private:

  double averageEfficiency_;
  double averageShapingTime_;
  double timeResolution_;
  double timeJitter_;
  double averageNoiseRate_;
//  double averageClusterSize_;
  std::vector<double> clsParametrization_;
  double signalPropagationSpeed_;
  bool cosmics_;
  int bxwidth_;
  int minBunch_;
  int maxBunch_;
  bool digitizeOnlyMuons_;
  bool doBkgNoise_;
  bool doNoiseCLS_;
  bool fixedRollRadius_;
  double minPabsNoiseCLS_;
  bool simulateIntrinsicNoise_;
  double scaleLumi_;
  bool simulateElectronBkg_;
  double constNeuGE11_;
  double slopeNeuGE11_;
  std::vector<double> GE21NeuBkgParams_;
  std::vector<double> GE11ElecBkgParams_;
  std::vector<double> GE21ElecBkgParams_;
};
#endif





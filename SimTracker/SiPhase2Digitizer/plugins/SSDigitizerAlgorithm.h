#ifndef _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

class SSDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm {
public:
  SSDigitizerAlgorithm(const edm::ParameterSet& conf);
  ~SSDigitizerAlgorithm() override;

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es) override;

  // run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const size_t inputBeginGlobalIndex,
                         const uint32_t tofBin,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield) override;
  bool select_hit(const PSimHit& hit, double tCorr, double& sigScale) override;
  bool isAboveThreshold(const DigitizerUtility::SimHitInfo* hitInfo, float charge, float thr) override;

private:
  enum { SquareWindow, SampledMode, LatchedMode, SampledOrLachedMode, HIPFindingMode };
  double nFactorial(int n);
  double aScalingConstant(int N, int i);
  double cbc3PulsePolarExpansion(double x);
  double signalShape(double x);
  double getSignalScale(double xval);
  void storeSignalShape();
  bool select_hit_sampledMode(const PSimHit& hit, double tCorr, double& sigScale);
  bool select_hit_latchedMode(const PSimHit& hit, double tCorr, double& sigScale);

  int hitDetectionMode_;
  std::vector<double> pulseShapeVec_;
  std::vector<double> pulseShapeParameters_;
  float deadTime_;
  static constexpr float bx_time{25};
  static constexpr size_t interpolationPoints{1000};
  static constexpr int interpolationStep{10};
};
#endif

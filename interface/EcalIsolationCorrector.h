//
// to use this code outside of CMSSW
// set this definition
//

//#define STANDALONE_ECALCORR
#ifndef STANDALONE_ECALCORR
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#endif

#include <string>

class EcalIsolationCorrector {
 public:

  enum RunRange {RunAB, RunC, RunD};

  EcalIsolationCorrector() {};
  ~EcalIsolationCorrector() {};

#ifndef STANDALONE_ECALCORR
  // Global correction for ABCD together
  float correctForNoise(reco::GsfElectron e, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForNoise(reco::GsfElectron e, int runNumber, bool isData=false);

  // Global correction for ABCD together
  float correctForHLTDefinition(reco::GsfElectron e, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForHLTDefinition(reco::GsfElectron e, int runNumber, bool isData=false);
#else
  // Global correction for ABCD together
  float correctForNoise(float unCorrIso, bool isBarrel, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForNoise(float unCorrIso, bool isBarrel, int runNumber, bool isData=false);

  // Global correction for ABCD together
  float correctForHLTDefinition(float unCorrIso, bool isBarrrel, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForHLTDefinition(float unCorrIso, bool isBarrel, int runNumber, bool isData=false);
#endif

 protected:
  RunRange checkRunRange(int runNumber);
  float correctForNoise(float iso, bool isBarrel, RunRange runRange, bool isData=false);
  float correctForHLTDefinition(float iso, bool isBarrel, RunRange runRange);
  
};

#ifndef SimMuon_RPCDigitizer_IRPCSimModelTiming_h
#define SimMuon_RPCDigitizer_IRPCSimModelTiming_h

#include "DataFormats/RPCDigi/interface/IRPCDigiCollection.h"
#include "SimMuon/RPCDigitizer/src/RPCSimModelTimingBase.h"

class IRPCSimModelTiming : public RPCSimModelTimingBase {
public:
  explicit IRPCSimModelTiming(const edm::ParameterSet& config);

  ~IRPCSimModelTiming() override;

protected:
  //---------------------------------------------------------
  // Create normal RPC digis from a simulated hit cluster
  //---------------------------------------------------------
  void digitizeCluster(const RPCRoll* roll,
                       const PSimHit& hit,
                       const std::vector<int>& cls,
                       float striplength,
                       CLHEP::HepRandomEngine* engine) override;

  //---------------------------------------------------------
  // Create noise digis
  //---------------------------------------------------------

  void createNoiseDigi(int strip, int hits, CLHEP::HepRandomEngine* engine) override;
};

#endif

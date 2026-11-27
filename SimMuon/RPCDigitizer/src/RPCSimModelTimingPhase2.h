#ifndef SimMuon_RPCDigitizer_RPCSimModelTimingPhase2_h
#define SimMuon_RPCDigitizer_RPCSimModelTimingPhase2_h

#include "DataFormats/RPCDigi/interface/RPCDigiPhase2Collection.h"
#include "SimMuon/RPCDigitizer/src/RPCSimModelTimingBase.h"

class RPCSimModelTimingPhase2 : public RPCSimModelTimingBase {
public:
  explicit RPCSimModelTimingPhase2(const edm::ParameterSet& config);

  ~RPCSimModelTimingPhase2() override;

protected:
  //---------------------------------------------------------
  // Create Phase-2 digis from simulated hits
  //---------------------------------------------------------

  void digitizeCluster(const RPCRoll* roll,
                       const PSimHit& hit,
                       const std::vector<int>& cls,
                       float striplength,
                       CLHEP::HepRandomEngine* engine) override;

  //---------------------------------------------------------
  // Create Phase-2 noise digis
  //---------------------------------------------------------
  void createNoiseDigi(int strip, int hits, CLHEP::HepRandomEngine* engine) override;
};

#endif

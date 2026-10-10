#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimMuon/RPCDigitizer/src/RPCSimModelTimingPhase2.h"

#include "CLHEP/Random/RandFlat.h"

//-------------------------------------------------------------
// Constructor
//-------------------------------------------------------------
RPCSimModelTimingPhase2::RPCSimModelTimingPhase2(const edm::ParameterSet& config) : RPCSimModelTimingBase(config) {}

//-------------------------------------------------------------
// Destructor
//-------------------------------------------------------------
RPCSimModelTimingPhase2::~RPCSimModelTimingPhase2() = default;

//-------------------------------------------------------------
// Digitization of signal hits
//-------------------------------------------------------------
void RPCSimModelTimingPhase2::digitizeCluster(const RPCRoll* roll,
                                              const PSimHit& hit,
                                              const std::vector<int>& cls,
                                              float striplength,
                                              CLHEP::HepRandomEngine* engine) {
  //---------------------------------------------------------
  // Calculate timing
  //---------------------------------------------------------

  double precise_time = rpcSync_->getTiming(&hit, engine, striplength);
  std::pair<int, int> tdc = rpcSync_->getBX_SBX(precise_time);

  //---------------------------------------------------------
  // Create digis for all strips in the cluster
  //---------------------------------------------------------
  for (const auto& strip : cls) {
    std::pair<int, int> digiKey(strip, tdc.first);
    RPCDigiPhase2 digi(strip, tdc.first, tdc.second);
    rpc_digis_phase2.emplace(digi);
    theDetectorHitMap.insert(DetectorHitMap::value_type(digiKey, &hit));
  }
}

//-------------------------------------------------------------
// Creation of Phase-2 noise digis
//-------------------------------------------------------------
void RPCSimModelTimingPhase2::createNoiseDigi(int strip, int hits, CLHEP::HepRandomEngine* engine) {
  for (int i = 0; i < hits; ++i) {
    //-------------------------------------------------                                                                                         // Timing information                                                                                                                       //-------------------------------------------------
    int bx = static_cast<int>(CLHEP::RandFlat::shoot(engine, nBXing_)) - nBXing_ / 2;
    int sbx = CLHEP::RandFlat::shootInt(long(0), long(10));

    RPCDigiPhase2 digi(strip, bx, sbx);
    rpc_digis_phase2.emplace(digi);
  }
}

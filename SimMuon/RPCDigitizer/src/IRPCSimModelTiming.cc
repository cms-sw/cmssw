#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimMuon/RPCDigitizer/src/IRPCSimModelTiming.h"

#include "CLHEP/Random/RandFlat.h"

//-------------------------------------------------------------
// Constructor
//-------------------------------------------------------------
IRPCSimModelTiming::IRPCSimModelTiming(const edm::ParameterSet& config) : RPCSimModelTimingBase(config) {}

//-------------------------------------------------------------
// Destructor
//-------------------------------------------------------------
IRPCSimModelTiming::~IRPCSimModelTiming() = default;

//-------------------------------------------------------------
// Digitization of signal hits
//-------------------------------------------------------------
void IRPCSimModelTiming::digitizeCluster(const RPCRoll* roll,
                                         const PSimHit& hit,
                                         const std::vector<int>& cls,
                                         float striplength,
                                         CLHEP::HepRandomEngine* engine) {
  //---------------------------------------------------------
  // Calculate timing for the two TDCs
  //---------------------------------------------------------
  std::pair<float, float> TDCs = rpcSync_->getDoubleTiming(&hit, engine, striplength);
  std::tuple<int, int, int> tdc1 = rpcSync_->getBX_SBX_fine_time(TDCs.first);
  std::tuple<int, int, int> tdc2 = rpcSync_->getBX_SBX_fine_time(TDCs.second);

  //---------------------------------------------------------
  // Create digis for all strips in the cluster
  //---------------------------------------------------------
  for (const auto& strip : cls) {
    std::pair<int, int> digiKey(strip, std::get<0>(tdc1));
    IRPCDigi digi(strip,
                  std::get<0>(tdc1),
                  std::get<0>(tdc2),
                  std::get<1>(tdc1),
                  std::get<1>(tdc2),
                  std::get<2>(tdc1),
                  std::get<2>(tdc2));

    irpc_digis.emplace(digi);
    theDetectorHitMap.insert(DetectorHitMap::value_type(digiKey, &hit));
  }
}

//-------------------------------------------------------------
// Creation of noise digis
//-------------------------------------------------------------
void IRPCSimModelTiming::createNoiseDigi(int strip, int hits, CLHEP::HepRandomEngine* engine) {
  for (int i = 0; i < hits; ++i) {
    //-------------------------------------------------
    // Timing information
    //-------------------------------------------------
    int bx1 = static_cast<int>(CLHEP::RandFlat::shoot(engine, nBXing_)) - nBXing_ / 2;
    int sbx1 = CLHEP::RandFlat::shootInt(long(0), long(10));
    int bx2 = static_cast<int>(CLHEP::RandFlat::shoot(engine, nBXing_)) - nBXing_ / 2;
    int sbx2 = CLHEP::RandFlat::shootInt(long(0), long(10));
    int fine1 = CLHEP::RandFlat::shootInt(long(0), long(12));
    int fine2 = CLHEP::RandFlat::shootInt(long(0), long(12));

    IRPCDigi digi(strip, bx1, bx2, sbx1, sbx2, fine1, fine2);
    irpc_digis.emplace(digi);
  }
}

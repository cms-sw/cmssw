#ifndef SimMuon_RPCDigitizerPhase2_h
#define SimMuon_RPCDigitizerPhase2_h
//

/** \class RPCDigitizerPhase2
 *  Digitizer class for RPC Phase2 upgrade
 *
 *  \author Borislav Pavlov -- Sofia University
 *
 */
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/RPCDigi/interface/RPCDigiPhase2Collection.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "CLHEP/Random/RandomEngine.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class RPCRoll;
class RPCSim;
class RPCSimSetUp;

namespace CLHEP {
  class HepRandomEngine;
}

class RPCDigitizerPhase2 {
public:
  typedef edm::DetSetVector<RPCDigiSimLink> RPCDigiSimLinks;
  RPCDigitizerPhase2(const edm::ParameterSet& config);
  ~RPCDigitizerPhase2();

  // *** digitize ***
  void doAction(MixCollection<PSimHit>& simHits,
                RPCDigiPhase2Collection& rpcDigis,
                RPCDigiSimLinks& rpcDigiSimLink,
                CLHEP::HepRandomEngine*);

  /// sets geometry
  void setGeometry(const RPCGeometry* geom) { theGeometry_ = geom; }

  void setRPCSimSetUp(RPCSimSetUp* simsetup) { theSimSetUp_ = simsetup; }

  RPCSimSetUp* theSimSetUp() { return theSimSetUp_; }

  /// finds the rpc det unit in the geometry associated with this det ID
  const RPCRoll* findDet(int detId) const;

private:
  const RPCGeometry* theGeometry_;
  std::unique_ptr<RPCSim> theRPCSim_;
  RPCSimSetUp* theSimSetUp_;
  bool theNoise_;
};

#endif

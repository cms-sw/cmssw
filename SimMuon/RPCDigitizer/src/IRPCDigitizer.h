#ifndef SimMuon_IRPCDigitizer_h
#define SimMuon_IRPCDigitizer_h
//

/** \class IRPCDigitizer
 *  Digitizer class for RPC
 *
 *  \author Borislav Pavlov -- University of Sofia
 *
 */
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/RPCDigi/interface/IRPCDigiCollection.h"
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

class IRPCDigitizer {
public:
  typedef edm::DetSetVector<RPCDigiSimLink> RPCDigiSimLinks;
  IRPCDigitizer(const edm::ParameterSet& config);
  ~IRPCDigitizer();

  // *** digitize ***
  void doAction(MixCollection<PSimHit>& simHits,
                IRPCDigiCollection& rpcDigis,
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
  std::string theName_;
  bool theNoise_;
};

#endif

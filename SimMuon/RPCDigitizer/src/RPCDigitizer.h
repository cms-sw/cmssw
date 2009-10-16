#ifndef SimMuon_RPCDigitizer_h
#define SimMuon_RPCDigitizer_h
// 

/** \class RPCDigitizer
 *  Digitizer class for RPC
 *
 *  \author Marcello Maggi -- INFN Bari
 *
 */
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include <string>
#include "CLHEP/Random/RandomEngine.h"

namespace edm{
  class ParameterSet;
}

class RPCRoll;
class RPCSim;
class RPCSimSetUp;
class RPCDigitizer
{
public:
  typedef edm::DetSetVector<RPCDigiSimLink> RPCDigiSimLinks;
  RPCDigitizer(const edm::ParameterSet& config, CLHEP::HepRandomEngine&);

  ~RPCDigitizer();

  /**  digitize
   */
  void doAction(MixCollection<PSimHit> & simHits,
                RPCDigiCollection & rpcDigis,
		RPCDigiSimLinks & rpcDigiSimLink);


  /// sets geometry
  void setGeometry(const RPCGeometry * geom) {theGeometry = geom;}

  void setRPCSimSetUp(RPCSimSetUp *simsetup){theSimSetUp = simsetup;}

  RPCSimSetUp* getRPCSimSetUp(){ return theSimSetUp; }
  
  /// finds the rpc det unit in the geometry associated with this det ID
  const RPCRoll * findDet(int detId) const;

private:
  const RPCGeometry * theGeometry;
  RPCSim* theRPCSim;
  RPCSimSetUp * theSimSetUp;
  std::string theName;

};

#endif


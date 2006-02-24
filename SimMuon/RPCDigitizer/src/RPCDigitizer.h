#ifndef SimMuon_RPCDigitizer_h
#define SimMuon_RPCDigitizer_h
// 

/** \class RPCDigitizer
 *  Digitizer class for RPC
 *
 *  \author Marcello Maggi -- INFN Bari
 *
 */


#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

class RPCRoll;
class RPCSim;
class RPCDigitizer 
{
public:
  RPCDigitizer();
  ~RPCDigitizer();

  /**  digitize
   */
  void doAction(MixCollection<PSimHit> & simHits,
                RPCDigiCollection & rpcDigis);


  /// sets geometry
  void setGeometry(const TrackingGeometry * geom) {theTrackingGeometry = geom;}

  /// sets the magnetic field
  void setMagneticField(const MagneticField * field);

  /// finds the rpc det unit in the geometry associated with this det ID
  const RPCRoll * findDet(int detId) const;

private:
  const TrackingGeometry * theTrackingGeometry;
  RPCSim* theRPCSim;
};

#endif


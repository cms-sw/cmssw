#ifndef RPCDigitizer_RPCSim_h
#define RPCDigitizer_RPCSim_h

/** \class RPCSim
 *   Base Class for the RPC strip response simulation
 *  
 *  \author Marcello Maggi -- INFN Bari
 */
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <set>

class RPCRoll;
class RPCDigiCollection;
class RPCSim
{
 public:
  virtual ~RPCSim(){};
  virtual void simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits )=0;
  virtual void fillDigis(int rollDetId, RPCDigiCollection& digis)=0;
 protected:
  RPCSim(){};
  virtual void init()=0;
 protected:
  std::set<int> strips;

};
#endif

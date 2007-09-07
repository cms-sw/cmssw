#ifndef RPCDigitizer_RPCSim_h
#define RPCDigitizer_RPCSim_h

/** \class RPCSim
 *   Base Class for the RPC strip response simulation
 *  
 *  \author Marcello Maggi -- INFN Bari
 */
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <set>

class RPCRoll;
class RPCGeometry;

class RPCSim
{
 public:
  virtual ~RPCSim(){};

  virtual void simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits, 
			const RPCGeometry*)=0;

  virtual void fillDigis(int rollDetId, RPCDigiCollection& digis);
 protected:
  RPCSim(const edm::ParameterSet& config);
  virtual void init()=0;
 protected:
  std::set< std::pair<int,int> > strips;
  //  int _bx;

};
#endif

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
//#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include <set>

class RPCRoll;
class RPCGeometry;
class RPCSimSetUp;

class RPCSim
{
 public:
  virtual ~RPCSim(){};

  virtual void simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits)=0;

  virtual void simulateNoise(const RPCRoll* roll)=0;

  virtual void fillDigis(int rollDetId, RPCDigiCollection& digis);

  void setRPCSimSetUp(RPCSimSetUp* setup){theSimSetUp = setup;}

  RPCSimSetUp* getRPCSimSetUp(){ return theSimSetUp; }

 protected:
  RPCSim(const edm::ParameterSet& config);
  virtual void init()=0;

 protected:
  std::set< std::pair<int,int> > strips;

 protected:
  RPCSimSetUp* theSimSetUp;
};
#endif

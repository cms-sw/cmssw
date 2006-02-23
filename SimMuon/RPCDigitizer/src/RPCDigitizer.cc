#include "Utilities/Timing/interface/TimingReport.h" 
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
//#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCHitSim.h"
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCSimAlgo/interface/RPCRoll.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>


// default constructor allocates default wire and strip digitizers
RPCDigitizer::RPCDigitizer() {
  RPCSimFactory f;
  theRPCSim = f.rpcSim();
}

RPCDigitizer::~RPCDigitizer() {
  if( theRPCSim )
    delete theRPCSim;
  theRPCSim = 0;
}



void RPCDigitizer::doAction(MixCollection<PSimHit> & simHits, 
                            RPCDigiCollection & rpcDigis)
{
  // arrange the hits by roll
  std::map<int, edm::PSimHitContainer> hitMap;
  for(MixCollection<PSimHit>::MixItr hitItr = simHits.begin();
      hitItr != simHits.end(); ++hitItr) 
  {
    hitMap[hitItr->detUnitId()].push_back(*hitItr);
  }

  // now loop over rolls and run the simulation for each one
  for(std::map<int, edm::PSimHitContainer>::const_iterator hitMapItr = hitMap.begin();
      hitMapItr != hitMap.end(); ++hitMapItr)
  {
    int rollDetId = hitMapItr->first;
    const RPCRoll* roll = this->findDet(rollDetId);
    const edm::PSimHitContainer & rollSimHits = hitMapItr->second;

    LogDebug("RPCDigitizer") << "RPCDigitizer: found " << rollSimHits.size() <<" hit(s) in the rpc roll";
    Timet2("RPCSim");

    theRPCSim->simulate(roll, rollSimHits);
    theRPCSim->fillDigis(rollDetId,rpcDigis);
    
  }
}


const RPCRoll * RPCDigitizer::findDet(int detId) const {
  assert(theTrackingGeometry != 0);
  const GeomDetUnit* detUnit = theTrackingGeometry->idToDet(RPCDetId(detId));
  return dynamic_cast<const RPCRoll *>(detUnit);
}


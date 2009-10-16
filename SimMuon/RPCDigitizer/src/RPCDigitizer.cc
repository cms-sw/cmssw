#include "Utilities/Timing/interface/TimingReport.h" 
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

// default constructor allocates default wire and strip digitizers

RPCDigitizer::RPCDigitizer(const edm::ParameterSet& config, CLHEP::HepRandomEngine& eng) {
  theName = config.getParameter<std::string>("digiModel");
  theRPCSim = RPCSimFactory::get()->create(theName,config.getParameter<edm::ParameterSet>("digiModelConfig"));
  theRPCSim->setRandomEngine(eng);
}

RPCDigitizer::~RPCDigitizer() {
  if( theRPCSim )
    delete theRPCSim;
  theRPCSim = 0;
}

void RPCDigitizer::doAction(MixCollection<PSimHit> & simHits, 
                            RPCDigiCollection & rpcDigis,
			    RPCDigiSimLinks & rpcDigiSimLink)
{

  theRPCSim->setRPCSimSetUp(theSimSetUp);

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
    TimeMe t2("RPCSim");

    theRPCSim->simulate(roll,rollSimHits);
    theRPCSim->simulateNoise(roll);
    theRPCSim->fillDigis(rollDetId,rpcDigis);
    rpcDigiSimLink.insert(theRPCSim->rpcDigiSimLinks());

  }
}

const RPCRoll * RPCDigitizer::findDet(int detId) const {
  assert(theGeometry != 0);
  const GeomDetUnit* detUnit = theGeometry->idToDetUnit(RPCDetId(detId));
  return dynamic_cast<const RPCRoll *>(detUnit);
}


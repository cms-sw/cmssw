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

   if ( ! theGeometry) {
   throw cms::Exception("Configuration")
     << "RPCDigitizer requires the RPCGeometry \n which is not present in the configuration file.  You must add the service\n in the configuration file or remove the modules that require it.";
  }


  std::vector<RPCRoll*>  rpcRolls = theGeometry->rolls() ;
  for(std::vector<RPCRoll*>::iterator r = rpcRolls.begin();
      r != rpcRolls.end(); r++){

    const edm::PSimHitContainer & rollSimHits = hitMap[(*r)->id()];
    
//    LogDebug("RPCDigitizer") << "RPCDigitizer: found " << rollSimHits.size() 
//			     <<" hit(s) in the rpc roll";  
    
    theRPCSim->simulate(*r,rollSimHits);
    theRPCSim->simulateNoise(*r);
    theRPCSim->fillDigis((*r)->id(),rpcDigis);
    rpcDigiSimLink.insert(theRPCSim->rpcDigiSimLinks());
  }
}

const RPCRoll * RPCDigitizer::findDet(int detId) const {
  assert(theGeometry != 0);
  const GeomDetUnit* detUnit = theGeometry->idToDetUnit(RPCDetId(detId));
  return dynamic_cast<const RPCRoll *>(detUnit);
}


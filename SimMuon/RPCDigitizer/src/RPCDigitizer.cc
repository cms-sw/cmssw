#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

// default constructor allocates default wire and strip digitizers

RPCDigitizer::RPCDigitizer(const edm::ParameterSet& config) {
  theName = config.getParameter<std::string>("digiModel");
  theRPCSim = RPCSimFactory::get()->create(theName,config.getParameter<edm::ParameterSet>("digiModelConfig"));
  theNoise=config.getParameter<bool>("doBkgNoise");
}

RPCDigitizer::~RPCDigitizer() {
  if( theRPCSim )
    delete theRPCSim;
  theRPCSim = 0;
}

void RPCDigitizer::doAction(MixCollection<PSimHit> & simHits, 
                            RPCDigiCollection & rpcDigis,
			    RPCDigiSimLinks & rpcDigiSimLink,
                            CLHEP::HepRandomEngine* engine)
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
  
  
  const std::vector<const RPCRoll*>&  rpcRolls = theGeometry->rolls() ;
  for(auto r = rpcRolls.begin();
      r != rpcRolls.end(); r++){
    
    RPCDetId id = (*r)->id();
    //    const edm::PSimHitContainer & rollSimHits = hitMap[(*r)->id()];
    const edm::PSimHitContainer & rollSimHits = hitMap[id];
    
    
    //    LogDebug("RPCDigitizer") << "RPCDigitizer: found " << rollSimHits.size() 
    //			     <<" hit(s) in the rpc roll";  
    
    //if( (id.region()!=0) && ((id.station()==3)||(id.station()==4))&&(id.ring()==1))
    //{
    //std::cout<<"YESID\t"<<id<<'\t'<<(*r)->nstrips()<<std::endl;
    //} else
    //{
    //std::cout<<"NOID\t"<<id<<'\t'<<(*r)->nstrips()<<std::endl;
    //}
    
    if(!((id.region()!=0) && ((id.station()==3)||(id.station()==4))&&(id.ring()==1))) // true if not IRPC (RE3/1 or RE4/1)
      {
	theRPCSim->simulate(*r, rollSimHits, engine); //"standard" RPC
      } else {
      theRPCSim->simulateIRPC(*r, rollSimHits, engine); // IRPC
    }
    
    if(theNoise){
      if(!((id.region()!=0) && ((id.station()==3)||(id.station()==4))&&(id.ring()==1))) // true if not IRPC (RE3/1 or RE4/1)
	{
          theRPCSim->simulateNoise(*r, engine); //"standard" RPC
	} else {
	theRPCSim->simulateIRPCNoise(*r, engine); // IRPC
      }
    }
    theRPCSim->fillDigis((*r)->id(),rpcDigis);
    rpcDigiSimLink.insert(theRPCSim->rpcDigiSimLinks());
  }
}

const RPCRoll * RPCDigitizer::findDet(int detId) const {
  assert(theGeometry != 0);
  const GeomDetUnit* detUnit = theGeometry->idToDetUnit(RPCDetId(detId));
  return dynamic_cast<const RPCRoll *>(detUnit);
}


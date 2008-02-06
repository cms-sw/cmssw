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

RPCDigitizer::RPCDigitizer(const edm::ParameterSet& config) {
  theName = config.getParameter<std::string>("digiModel");
  theRPCSim = RPCSimFactory::get()->create(theName,config.getParameter<edm::ParameterSet>("digiModelConfig"));
}

RPCDigitizer::~RPCDigitizer() {
  if( theRPCSim )
    delete theRPCSim;
  theRPCSim = 0;
}

void RPCDigitizer::doAction(MixCollection<PSimHit> & simHits, 
                            RPCDigiCollection & rpcDigis,
			    DigiSimLinks & RPCDigiSimLinks, 
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
  std::cout<<"------------RPC DIGITIZER : NUM SIMHIT TOTALI PER EVENTO: "<<hitMap.size()<<std::endl;

  // now loop over rolls and run the simulation for each one
  for(std::map<int, edm::PSimHitContainer>::const_iterator hitMapItr = hitMap.begin();
      hitMapItr != hitMap.end(); ++hitMapItr)
  {
    int rollDetId = hitMapItr->first;
    const RPCRoll* roll = this->findDet(rollDetId);
    const edm::PSimHitContainer & rollSimHits = hitMapItr->second;

    LogDebug("RPCDigitizer") << "RPCDigitizer: found " << rollSimHits.size() <<" hit(s) in the rpc roll";
    TimeMe t2("RPCSim");
    std::cout<<"------------RPC DIGITIZER : SIMULATE"<<std::endl;
    theRPCSim->simulate(roll,rollSimHits);
    //theRPCSim->fillDigis(rollDetId,rpcDigis);

    theRPCSim->simulateNoise(roll);

    std::cout<<"------------RPC DIGITIZER : FILL DIGIS"<<std::endl;
    theRPCSim->fillDigis(rollDetId,rpcDigis);

    std::cout<<"------------RPC DIGITIZER : INSERT DETSETVECTOR COMP."<<std::endl;
    rpcDigiSimLink.insert(theRPCSim->rpcDigiSimLinks());

    std::cout<<"------------------------------ BEGIN LOOP DV ------------------------------------------"<<std::endl;
    for (edm::DetSetVector<RPCDigiSimLink>::const_iterator itlink = rpcDigiSimLink.begin(); itlink != rpcDigiSimLink.end(); itlink++)
      {
	std::cout<<"------------------------------DETSET BEGIN  ------------------------------------------"<<std::endl;
	for(edm::DetSet<RPCDigiSimLink>::const_iterator digi_iter=itlink->data.begin();digi_iter != itlink->data.end();++digi_iter){
	  const PSimHit* hit = digi_iter->getSimHit();
	  float xpos = hit->localPosition().x();
	  int strip = digi_iter->getStrip();
	  int bx = digi_iter->getBx();
	  
	  std::cout<<"DetUnit: "<<hit->detUnitId()<<"  "<<"Event ID: "<<hit->eventId().event()<<"  "<<"Pos X: "<<xpos<<"  "<<"Strip: "<<strip<<"  "<<"Bx: "<<bx<<std::endl;
	}
	std::cout<<"------------------------------DETSET END  ------------------------------------------"<<std::endl;
      }
    std::cout<<"------------------------------ END LOOP DV ------------------------------------------"<<std::endl;

  }
}

const RPCRoll * RPCDigitizer::findDet(int detId) const {
  assert(theGeometry != 0);
  const GeomDetUnit* detUnit = theGeometry->idToDetUnit(RPCDetId(detId));
  return dynamic_cast<const RPCRoll *>(detUnit);
}


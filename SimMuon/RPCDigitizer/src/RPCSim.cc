#include "RPCSim.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

RPCSim::RPCSim(const edm::ParameterSet& config)
{
}

void
RPCSim::fillDigis(int rollDetId, RPCDigiCollection& digis)
{
  //  theRpcDigiSimLinks.clear();
  std::vector<std::pair<int,int> > vdigi;
  vdigi.clear();

  for (std::set< std::pair<int,int> >::iterator i=strips.begin();
       i!=strips.end(); i++){
    if(i->second != -999){
      RPCDigi rpcDigi(i->first,i->second);
      //NCA
      digis.insertDigi(RPCDetId(rollDetId),rpcDigi);
      this->addLinks(i->first,i->second);
    }
  }
  strips.clear();
}

void RPCSim::addLinks(unsigned int strip, int bx) {

  std::pair<unsigned int, int > digi(strip, bx);
  std::pair<DetectorHitMap::iterator, DetectorHitMap::iterator> channelHitItr 
     = theDetectorHitMap.equal_range(digi);

  for( DetectorHitMap::iterator hitItr = channelHitItr.first; 
                                hitItr != channelHitItr.second; ++hitItr){
    const PSimHit * hit = (hitItr->second);

    if(hit != 0) {
      theRpcDigiSimLinks.push_back( RPCDigiSimLink(digi, hit->entryPoint(),hit->momentumAtEntry(),
						   hit->timeOfFlight(),hit->energyLoss(),hit->particleType(), 
						   hit->detUnitId(), hit->trackId(), hit->eventId(), hit->processType() ) );

    }
  }
}





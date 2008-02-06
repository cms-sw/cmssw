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
  for (std::set< std::pair<int,int> >::iterator i=strips.begin();
       i!=strips.end(); i++){
    if(i->second != -999){
      RPCDigi rpcDigi(i->first,i->second);
      //NCA
      digis.insertDigi(RPCDetId(rollDetId),rpcDigi);
      this->addLinks(i->first,i->second);
    }
  }

  std::cout<<"DIMENSIONE VECTOR: "<<theRpcDigiSimLinks.size()<<std::endl;
  for(edm::DetSet<RPCDigiSimLink>::const_iterator digi_iter=theRpcDigiSimLinks.data.begin();digi_iter != theRpcDigiSimLinks.data.end();++digi_iter){
    const PSimHit* hit = digi_iter->getSimHit();
    float xpos = hit->localPosition().x();
    int strip = digi_iter->getStrip();
    int bx = digi_iter->getBx();
    std::cout<<"DetUnit: "<<hit->detUnitId()<<"  "<<"Pos X: "<<hit->entryPoint()<<"  "<<"Strip: "<<strip<<"  "<<"Bx: "<<bx<<std::endl;
  }

  strips.clear();
}

void RPCSim::addLinks(unsigned int strip, int bx) {

  std::cout<<"----------- Add link --------------"<<std::endl;

  std::pair<unsigned int, int > digi(strip, bx);
  //  DetectorHitMap::iterator itp = theDetectorHitMap.find(digi);
  //  if(itp != theDetectorHitMap.end() ) std::cout<<"MAPPA"<<std::endl;

  std::pair<DetectorHitMap::iterator, DetectorHitMap::iterator> channelHitItr 
     = theDetectorHitMap.equal_range(digi);

  // find the fraction contribution for each SimTrack
  std::map<int,float> simTrackChargeMap;
  std::map<int, EncodedEventId> eventIdMap;
  float totalCharge = 0;
  for( DetectorHitMap::iterator hitItr = channelHitItr.first; 
                                hitItr != channelHitItr.second; ++hitItr){
    const PSimHit * hit = hitItr->second;
    // might be zero for unit tests and such
    if(hit != 0) {
      int simTrackId = (hitItr->second)->trackId();

      //     std::cout<<"SIMTRAKID: "<<simTrackId<< std::endl;

      float charge = ((hitItr->second)->particleType()/fabs((hitItr->second)->particleType()));
      std::map<int,float>::iterator chargeItr = simTrackChargeMap.find(simTrackId);
      if( chargeItr == simTrackChargeMap.end() ) {
        simTrackChargeMap[simTrackId] = charge;
        eventIdMap[simTrackId] = hit->eventId();
      } else {
        chargeItr->second += charge;
      }
      totalCharge += charge;

      //------------- MIO ------------------

      theRpcDigiSimLinks.push_back( RPCDigiSimLink(digi, hit) );
      RPCDigiSimLink* prova = new RPCDigiSimLink(digi, hit);
      std::cout<<"Strip: "<<prova->getStrip()<<"  "<<"Bx: "<<prova->getBx()<<"  "<<"SimHit: "<<hit->detUnitId()<<"  "<<"Event ID: "<<hit->eventId().event()<<"  "<<"Position: "<<(prova->getSimHit())->localPosition().x()<<std::endl;
      std::cout<<"-----------------------------------------"<<std::endl; 
      //____________________________________


    }
  }

  for(std::map<int,float>::iterator chargeItr = simTrackChargeMap.begin(); 
                          chargeItr != simTrackChargeMap.end(); ++chargeItr) {
    int simTrackId = chargeItr->first;
    theDigiSimLinks.push_back( StripDigiSimLink(strip, simTrackId,  
                                  eventIdMap[simTrackId], chargeItr->second/totalCharge ) );
    
  }

}





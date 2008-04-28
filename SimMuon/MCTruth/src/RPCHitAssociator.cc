#include "SimMuon/MCTruth/interface/RPCHitAssociator.h"

using namespace std;

typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;

// Constructor
RPCHitAssociator::RPCHitAssociator(const edm::Event& e, const edm::EventSetup& eventSetup, const edm::ParameterSet& conf) {

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByLabel("mix", "MuonRPCHits", cf);

  std::auto_ptr<MixCollection<PSimHit> > 
    hits( new MixCollection<PSimHit>(cf.product()) );
  MixCollection<PSimHit> & simHits = *hits;

  for(MixCollection<PSimHit>::MixItr hitItr = simHits.begin();
      hitItr != simHits.end(); ++hitItr) 
  {
    _SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
  }

  edm::Handle< edm::DetSetVector<RPCDigiSimLink> > thelinkDigis;
  e.getByLabel("muonRPCDigis","RPCDigiSimLink", thelinkDigis);
  _thelinkDigis = thelinkDigis;
}
// end of constructor

std::vector<SimHitIdpr> RPCHitAssociator::associateRecHit(const TrackingRecHit & hit) {
 
  std::vector<SimHitIdpr> matched;
  matched.clear();

  const TrackingRecHit * hitp = &hit;
  const RPCRecHit * rpcrechit = dynamic_cast<const RPCRecHit *>(hitp);

  RPCDetId rpcDetId = rpcrechit->rpcId();
  int fstrip = rpcrechit->firstClusterStrip();
  int cls = rpcrechit->clusterSize();
  int bx = rpcrechit->BunchX();

  for(int i = fstrip; i < fstrip+cls; ++i) {
    std::set<RPCDigiSimLink> links = findRPCDigiSimLink(rpcDetId.rawId(),i,bx);
    for(std::set<RPCDigiSimLink>::iterator itlink = links.begin(); itlink != links.end(); ++itlink) {
      SimHitIdpr currentId(itlink->getTrackId(),itlink->getEventId());
      if(find(matched.begin(),matched.end(),currentId ) == matched.end())
        matched.push_back(currentId);
    }
  }

  return  matched;
}
  
std::set<RPCDigiSimLink>  RPCHitAssociator::findRPCDigiSimLink(uint32_t rpcDetId, int strip, int bx){

  std::set<RPCDigiSimLink> links;

  for (edm::DetSetVector<RPCDigiSimLink>::const_iterator itlink = _thelinkDigis->begin(); itlink != _thelinkDigis->end(); itlink++){
    for(edm::DetSet<RPCDigiSimLink>::const_iterator digi_iter=itlink->data.begin();digi_iter != itlink->data.end();++digi_iter){

      uint32_t detid = digi_iter->getDetUnitId();
      int str = digi_iter->getStrip();
      int bunchx = digi_iter->getBx();

      if(detid == rpcDetId && str == strip && bunchx == bx){
        links.insert(*digi_iter);
      }

    }
  }

  return links;
}



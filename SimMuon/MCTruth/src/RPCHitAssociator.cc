#include "SimMuon/MCTruth/interface/RPCHitAssociator.h"

using namespace std;



// Constructor
RPCHitAssociator::RPCHitAssociator( const edm::ParameterSet& conf, 
				    edm::ConsumesCollector && iC):
  RPCdigisimlinkTag(conf.getParameter<edm::InputTag>("RPCdigisimlinkTag")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  RPCsimhitsTag(conf.getParameter<edm::InputTag>("RPCsimhitsTag")),
  RPCsimhitsXFTag(conf.getParameter<edm::InputTag>("RPCsimhitsXFTag"))
{
  if (crossingframe){
    RPCsimhitsXFToken_=iC.consumes<CrossingFrame<PSimHit> >(RPCsimhitsXFTag);
  } else if (!RPCsimhitsTag.label().empty()) {
    RPCsimhitsToken_=iC.consumes<edm::PSimHitContainer>(RPCsimhitsTag);
  }

  RPCdigisimlinkToken_=iC.consumes< edm::DetSetVector<RPCDigiSimLink> >(RPCdigisimlinkTag); 
}

RPCHitAssociator::RPCHitAssociator(const edm::Event& e, const edm::EventSetup& eventSetup, const edm::ParameterSet& conf ):
  RPCdigisimlinkTag(conf.getParameter<edm::InputTag>("RPCdigisimlinkTag")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  RPCsimhitsTag(conf.getParameter<edm::InputTag>("RPCsimhitsTag")),
  RPCsimhitsXFTag(conf.getParameter<edm::InputTag>("RPCsimhitsXFTag"))
{
  initEvent(e,eventSetup);
}


void RPCHitAssociator::initEvent(const edm::Event& e, const edm::EventSetup& eventSetup)


{
  if (crossingframe) {
    
    edm::Handle<CrossingFrame<PSimHit> > cf;
    LogTrace("RPCHitAssociator") <<"getting CrossingFrame<PSimHit> collection - "<<RPCsimhitsXFTag;
    e.getByLabel(RPCsimhitsXFTag, cf);
    
    std::auto_ptr<MixCollection<PSimHit> > 
      RPCsimhits( new MixCollection<PSimHit>(cf.product()) );
    LogTrace("RPCHitAssociator") <<"... size = "<<RPCsimhits->size();

    //   MixCollection<PSimHit> & simHits = *hits;
    
    for(MixCollection<PSimHit>::MixItr hitItr = RPCsimhits->begin();
	hitItr != RPCsimhits->end(); ++hitItr) 
      {
	_SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
      }
    
  } else if (!RPCsimhitsTag.label().empty()) {
    edm::Handle<edm::PSimHitContainer> RPCsimhits;
    LogTrace("RPCHitAssociator") <<"getting PSimHit collection - "<<RPCsimhitsTag;
    e.getByLabel(RPCsimhitsTag, RPCsimhits);    
    LogTrace("RPCHitAssociator") <<"... size = "<<RPCsimhits->size();
    
    // arrange the hits by detUnit
    for(edm::PSimHitContainer::const_iterator hitItr = RPCsimhits->begin();
	hitItr != RPCsimhits->end(); ++hitItr)
      {
	_SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
      }
  }

  edm::Handle< edm::DetSetVector<RPCDigiSimLink> > thelinkDigis;
  LogTrace("RPCHitAssociator") <<"getting RPCDigiSimLink collection - "<<RPCdigisimlinkTag;
  e.getByLabel(RPCdigisimlinkTag, thelinkDigis);
  _thelinkDigis = thelinkDigis;
}
// end of constructor

std::vector<RPCHitAssociator::SimHitIdpr> RPCHitAssociator::associateRecHit(const TrackingRecHit & hit) const {
  
  std::vector<SimHitIdpr> matched;

  const TrackingRecHit * hitp = &hit;
  const RPCRecHit * rpcrechit = dynamic_cast<const RPCRecHit *>(hitp);

  if (rpcrechit) {
    
    RPCDetId rpcDetId = rpcrechit->rpcId();
    int fstrip = rpcrechit->firstClusterStrip();
    int cls = rpcrechit->clusterSize();
    int bx = rpcrechit->BunchX();
    
    for(int i = fstrip; i < fstrip+cls; ++i) {
      std::set<RPCDigiSimLink> links = findRPCDigiSimLink(rpcDetId.rawId(),i,bx);
      
      if (links.empty()) edm::LogWarning("RPCHitAssociator")
	<<"*** WARNING in RPCHitAssociator::associateRecHit, RPCRecHit "<<*rpcrechit<<", strip "<<i<<" has no associated RPCDigiSimLink !"<<endl;
      
      for(std::set<RPCDigiSimLink>::iterator itlink = links.begin(); itlink != links.end(); ++itlink) {
	SimHitIdpr currentId(itlink->getTrackId(),itlink->getEventId());
	if(find(matched.begin(),matched.end(),currentId ) == matched.end())
	  matched.push_back(currentId);
      }
    }
    
  } else edm::LogWarning("RPCHitAssociator")<<"*** WARNING in RPCHitAssociator::associateRecHit, null dynamic_cast !";
  
  return  matched;
}
  
std::set<RPCDigiSimLink>  RPCHitAssociator::findRPCDigiSimLink(uint32_t rpcDetId, int strip, int bx) const {

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



#include "SimMuon/MCTruth/interface/GEMHitAssociator.h"

using namespace std;

// Constructor
GEMHitAssociator::GEMHitAssociator( const edm::ParameterSet& conf,
				    edm::ConsumesCollector && iC):
  GEMdigisimlinkTag(conf.getParameter<edm::InputTag>("GEMdigisimlinkTag")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  useGEMs_(conf.getParameter<bool>("useGEMs")),
  GEMsimhitsTag(conf.getParameter<edm::InputTag>("GEMsimhitsTag")),
  GEMsimhitsXFTag(conf.getParameter<edm::InputTag>("GEMsimhitsXFTag"))
{
  if (crossingframe){
    GEMsimhitsXFToken_=iC.consumes<CrossingFrame<PSimHit> >(GEMsimhitsXFTag);
  } else if (!GEMsimhitsTag.label().empty()) {
    GEMsimhitsToken_=iC.consumes<edm::PSimHitContainer>(GEMsimhitsTag);
  }

  GEMdigisimlinkToken_=iC.consumes< edm::DetSetVector<StripDigiSimLink> >(GEMdigisimlinkTag);
}

GEMHitAssociator::GEMHitAssociator(const edm::Event& e, const edm::EventSetup& eventSetup, const edm::ParameterSet& conf ):
  GEMdigisimlinkTag(conf.getParameter<edm::InputTag>("GEMdigisimlinkTag")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  useGEMs_(conf.getParameter<bool>("useGEMs")),
  GEMsimhitsTag(conf.getParameter<edm::InputTag>("GEMsimhitsTag")),
  GEMsimhitsXFTag(conf.getParameter<edm::InputTag>("GEMsimhitsXFTag"))
{
  initEvent(e,eventSetup);
}

void GEMHitAssociator::initEvent(const edm::Event& e, const edm::EventSetup& eventSetup)
{

  if(useGEMs_){

	  if (crossingframe) {
	    
	    edm::Handle<CrossingFrame<PSimHit> > cf;
	    LogTrace("GEMHitAssociator") <<"getting CrossingFrame<PSimHit> collection - "<<GEMsimhitsXFTag;
	    e.getByLabel(GEMsimhitsXFTag, cf);
	    
	    std::unique_ptr<MixCollection<PSimHit> > 
	      GEMsimhits( new MixCollection<PSimHit>(cf.product()) );
	    LogTrace("GEMHitAssociator") <<"... size = "<<GEMsimhits->size();

	    //   MixCollection<PSimHit> & simHits = *hits;
	    
	    for(MixCollection<PSimHit>::MixItr hitItr = GEMsimhits->begin();
		hitItr != GEMsimhits->end(); ++hitItr) 
	      {
		_SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
	      }
	    
	  } else if (!GEMsimhitsTag.label().empty()) {
	    edm::Handle<edm::PSimHitContainer> GEMsimhits;
	    LogTrace("GEMHitAssociator") <<"getting PSimHit collection - "<<GEMsimhitsTag;
	    e.getByLabel(GEMsimhitsTag, GEMsimhits);    
	    LogTrace("GEMHitAssociator") <<"... size = "<<GEMsimhits->size();
	    
	    // arrange the hits by detUnit
	    for(edm::PSimHitContainer::const_iterator hitItr = GEMsimhits->begin();
		hitItr != GEMsimhits->end(); ++hitItr)
	      {
		_SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
	      }
	  }

	  edm::Handle<DigiSimLinks> digiSimLinks;
	  LogTrace("GEMHitAssociator") <<"getting GEM Strip DigiSimLink collection - "<<GEMdigisimlinkTag;
	  e.getByLabel(GEMdigisimlinkTag, digiSimLinks);
	  theDigiSimLinks = digiSimLinks.product();

  }

}
// end of constructor

std::vector<GEMHitAssociator::SimHitIdpr> GEMHitAssociator::associateRecHit(const GEMRecHit * gemrechit) const {
  
  std::vector<SimHitIdpr> matched;

  if(useGEMs_){
    if (gemrechit) {
	    
	    GEMDetId gemDetId = gemrechit->gemId();
	    int fstrip = gemrechit->firstClusterStrip();
	    int cls = gemrechit->clusterSize();
	    //int bx = gemrechit->BunchX();

	    DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(gemDetId);

	    if (layerLinks != theDigiSimLinks->end()) {
	    
		for(int i = fstrip; i < (fstrip+cls); ++i) {
			      
			for(LayerLinks::const_iterator itlink = layerLinks->begin(); itlink != layerLinks->end(); ++itlink) {

		  		int ch = static_cast<int>(itlink->channel());
				if(ch != i) continue;

				SimHitIdpr currentId(itlink->SimTrackId(), itlink->eventId());
				if(find(matched.begin(),matched.end(),currentId ) == matched.end())
					matched.push_back(currentId);

			}

		}

	    }else edm::LogWarning("GEMHitAssociator")
	      <<"*** WARNING in GEMHitAssociator: GEM layer "<<gemDetId<<" has no DigiSimLinks !"<<std::endl;
	    
	  } else edm::LogWarning("GEMHitAssociator")<<"*** WARNING in GEMHitAssociator::associateRecHit, null dynamic_cast !";

  }
  
  return  matched;

}


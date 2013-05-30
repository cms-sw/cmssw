#ifndef HitPixelLayersTrackSelection_h
#define HitPixelLayersTrackSelection_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

/**
 Selector to select only tracking particles that leave hits in three pixel layers
 Additional selection done on pt, rapidity, impact parameter, min hits, pdg id, etc.
 
 Inspired by CommonTools.RecoAlgos.TrackingParticleSelector.h 
**/

class HitPixelLayersTPSelector
{
  
 public:
  // input collection type
  typedef TrackingParticleCollection collection;
  
  
  // output collection type
  typedef std::vector<const TrackingParticle*> container;
  
  // iterator over result collection type.
  typedef container::const_iterator const_iterator;
  
  // constructor from parameter set configurability
  HitPixelLayersTPSelector(const edm::ParameterSet & iConfig) : 
    tripletSeedOnly_(iConfig.getParameter<bool>("tripletSeedOnly")),
    ptMin_(iConfig.getParameter<double>("ptMin")), 
    minRapidity_(iConfig.getParameter<double>("minRapidity")), 
    maxRapidity_(iConfig.getParameter<double>("maxRapidity")),
    tip_(iConfig.getParameter<double>("tip")), 
    lip_(iConfig.getParameter<double>("lip")), 
    minHit_(iConfig.getParameter<int>("minHit")), 
    signalOnly_(iConfig.getParameter<bool>("signalOnly")), 
    chargedOnly_(iConfig.getParameter<bool>("chargedOnly")), 
    primaryOnly_(iConfig.getParameter<bool>("primaryOnly")),
    tpStatusBased_(iConfig.getParameter<bool>("tpStatusBased")),
    pdgId_(iConfig.getParameter< std::vector<int> >("pdgId"))				   
      {};
    
    // select object from a collection and
    // possibly event content
    void select( const edm::Handle<collection> & TPCH, const edm::Event & iEvent, const edm::EventSetup & iSetup)
    {
      selected_.clear();
      //Retrieve tracker topology from geometry
      edm::ESHandle<TrackerTopology> tTopoHand;
      iSetup.get<IdealGeometryRecord>().get(tTopoHand);
      const TrackerTopology *tTopo=tTopoHand.product();
      

      const collection & tpc = *(TPCH.product());
      
      for (TrackingParticleCollection::size_type i=0; i<tpc.size(); i++)
	{
	  TrackingParticleRef tpr(TPCH, i);
	  
	  // quickly reject if it is from pile-up
	  if (signalOnly_ && !(tpr->eventId().bunchCrossing()==0 && tpr->eventId().event()==0) ) continue;
	  if (chargedOnly_ && tpr->charge()==0) continue; //select only if charge!=0
	  if (tpStatusBased_ && primaryOnly_ && tpr->status()!=1 ) continue; // TP status based sel primary
	  if ((!tpStatusBased_) && primaryOnly_ && tpr->parentVertex()->nSourceTracks()!=0 ) continue; // vertex based sel for primary

	  // loop over specified PID values 
	  bool testId = false;
	  unsigned int idSize = pdgId_.size();
	  if (idSize==0) testId = true;
	  else for (unsigned int it=0;it!=idSize;++it){
	    if (tpr->pdgId()==pdgId_[it]) testId = true;
	  }
	  
	  // selection criteria
	  if ( tpr->matchedHit() >= minHit_ &&   
	       sqrt(tpr->momentum().perp2()) >= ptMin_ && 
	       tpr->momentum().eta() >= minRapidity_ && tpr->momentum().eta() <= maxRapidity_ && 
	       sqrt(tpr->vertex().perp2()) <= tip_ &&
	       fabs(tpr->vertex().z()) <= lip_ &&
	       testId)
	    {
	      if (tripletSeedOnly_ && !goodHitPattern(pixelHitPattern(tpr,tTopo)) ) continue; //findable triplet seed
	      const TrackingParticle * trap = &(tpc[i]);
	      selected_.push_back(trap);
	    } 
	  
	}
    }
    
    // return pixel layer hit pattern
    std::vector<bool> pixelHitPattern( const TrackingParticleRef& simTrack, const TrackerTopology *tTopo )
      {
	std::vector<bool> hitpattern(5,false); // PXB 0,1,2  PXF 0,1
	
	for(std::vector<PSimHit>::const_iterator simHit = simTrack->pSimHit_begin();simHit!= simTrack->pSimHit_end();simHit++){
	  
	  DetId id = DetId(simHit->detUnitId());
	  uint32_t detid = id.det();
	  uint32_t subdet = id.subdetId();
	  
	  if (detid == DetId::Tracker) {
	    if (subdet == PixelSubdetector::PixelBarrel) 
	      hitpattern[tTopo->pxbLayer(id)-1]=true;
	    else if (subdet == PixelSubdetector::PixelEndcap) 
	      hitpattern[tTopo->pxfDisk(id)+2]=true;
	  }
	  
	}// end simhit loop
	
	return hitpattern;
      }
    
    // test whether hit pattern would give a pixel triplet seed
    bool goodHitPattern( const std::vector<bool> hitpattern )
    {
      if( (hitpattern[0] && hitpattern[1] && hitpattern[2]) ||
	  (hitpattern[0] && hitpattern[1] && hitpattern[3]) ||
	  (hitpattern[0] && hitpattern[3] && hitpattern[4]) ) 
	return true;
      else 
	return false;
    }
    
    // iterators over selected objects: collection begin
    const_iterator begin() const
    {
      return selected_.begin();
    }
    
    // iterators over selected objects: collection end
    const_iterator end() const
    {
      return selected_.end();
    }
    
    // true if no object has been selected
    size_t size() const
    {
      return selected_.size();
    }
    
    //private:
    
    container selected_;
    bool tripletSeedOnly_;
    double ptMin_;
    double minRapidity_;
    double maxRapidity_;
    double tip_;
    double lip_;
    int    minHit_;
    bool signalOnly_;
    bool chargedOnly_;
    bool primaryOnly_;
    bool tpStatusBased_;
    std::vector<int> pdgId_;
    
    
};


#endif

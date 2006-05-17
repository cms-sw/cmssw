// File: TrackerHitAssociator.cc

#include <memory>
#include <string>
#include <vector>

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
//--- for Geometry:
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

using namespace std;
using namespace edm;

namespace cms{
  
  std::vector<PSimHit> TrackerHitAssociator::associateHit(const TrackingRecHit & thit) 
  {
    
    //vector with the matched SimHit
    std::vector<PSimHit> result; 
    
    //get the Detector type of the rechit
    DetId detid=  thit.geographicalId();
    uint32_t detID = detid.rawId();
    //cout << "Associator ---> get Detid " << detID << endl;
    //check we are in the strip tracker
    if(detid.subdetId() == StripSubdetector::TIB ||
       detid.subdetId() == StripSubdetector::TOB || 
       detid.subdetId() == StripSubdetector::TID ||
       detid.subdetId() == StripSubdetector::TEC) 
      {
	//check if it is a simple SiStripRecHit2DLocalPos
	if(const SiStripRecHit2DLocalPos * rechit = 
	   dynamic_cast<const SiStripRecHit2DLocalPos *>(&thit))
	  {	  
	    simtrackid = associateSimpleRecHit(rechit);
	  }
	//check if it is a matched SiStripRecHit2DMatchedLocalpos
	if(const SiStripRecHit2DMatchedLocalPos * rechit = 
	   dynamic_cast<const SiStripRecHit2DMatchedLocalPos *>(&thit))
	  {	  
	    simtrackid = associateMatchedRecHit(rechit);
	  }
      }
    //check we are in the pixel tracker
    if( detid.subdetId() == PixelSubdetector::PixelBarrel || 
	detid.subdetId() == PixelSubdetector::PixelEndcap) 
      {
	if(const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>(&thit))
	  {	  
	    simtrackid = associatePixelRecHit(rechit);
	  }
      }
    
    
    //now get the SimHit from the trackid
    vector<PSimHit> simHit; 
    std::map<unsigned int, std::vector<PSimHit> >::const_iterator it = SimHitMap.find(detID);
    simHit.clear();
    if (it!= SimHitMap.end()){
      simHit = it->second;
      vector<PSimHit>::const_iterator simHitIter = simHit.begin();
      vector<PSimHit>::const_iterator simHitIterEnd = simHit.end();
      for (;simHitIter != simHitIterEnd; ++simHitIter) {
	const PSimHit ihit = *simHitIter;
	unsigned int simHitid = ihit.trackId();
	for(size_t i=0; i<simtrackid.size();i++){
	  //cout << " Associator -->  check sihit id's = " << simHitid << endl;
	  if(simHitid == simtrackid[i] && simtrackid[i]!= 65535){ //exclude the geant particles. they all have the same id
	    // cout << "Associator ---> ID" << ihit.trackId() << " Simhit x= " << ihit.localPosition().x() 
	    //	   << " y= " <<  ihit.localPosition().y() << " z= " <<  ihit.localPosition().x() << endl;	    
	    result.push_back(ihit);
	  }
	}
      }
    }
    return result;  
  }
  
 std::vector<unsigned int>  TrackerHitAssociator::associateSimpleRecHit(const SiStripRecHit2DLocalPos * simplerechit)
  {
    DetId detid=  simplerechit->geographicalId();
    uint32_t detID = detid.rawId();
    
    edm::DetSetVector<StripDigiSimLink>::const_iterator isearch = stripdigisimlink->find(detID); 
    if(isearch != stripdigisimlink->end()) {  //if it is not empty
      //link_detset is a structure, link_detset.data is a std::vector<StripDigiSimLink>
      edm::DetSet<StripDigiSimLink> link_detset = (*stripdigisimlink)[detID];
      //cout << "Associator ---> get digilink! in Detid n = " << link_detset.data.size() << endl;

      const std::vector<const SiStripCluster*> clust=simplerechit->cluster();
      //cout << "Associator ---> get cluster info " << endl;
      for(vector<const SiStripCluster*>::const_iterator ic = clust.begin(); ic!=clust.end(); ic++) {
	unsigned int clusiz = (*ic)->amplitudes().size();
	unsigned int first  = (*ic)->firstStrip();     
	unsigned int last   = first + clusiz;
	// cout << "Associator ---> clus size = " << clusiz << " first = " << first << " last = " << last << endl;
	for(edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin(); linkiter != link_detset.data.end(); linkiter++){
	  StripDigiSimLink link = *linkiter;
	  if( link.channel() >= first  && link.channel() < last ){
	    simtrackid.push_back(link.SimTrackId());
	    cout << "Associator --> digi list first= " << first << " last = " << last << endl;
	    cout << "Associator link--> channel= " << link.channel() << "  trackid = " << link.SimTrackId() << endl;
	  }
	}
      }
    }

    return simtrackid;
  }

  
 std::vector<unsigned int>  TrackerHitAssociator::associateMatchedRecHit(const SiStripRecHit2DMatchedLocalPos * matchedrechit)
  {
    //to be written
    return simtrackid;
  }

 std::vector<unsigned int>  TrackerHitAssociator::associatePixelRecHit(const SiPixelRecHit * pixelrechit)
  {
    //to be written
    return simtrackid;
  }
  
  //constructor
  TrackerHitAssociator::TrackerHitAssociator(const edm::Event& e)  : myEvent_(e)  {
    //  using namespace edm;
    
    //get stuff from the event
    //changed to detsetvector
    //edm::Handle< edm::DetSetVector<StripDigiSimLink> >  stripdigisimlink;
    //    e.getByLabel("stripdigi", "stripdigi", stripdigisimlink);
    e.getByLabel("stripdigi", stripdigisimlink);
    //cout << "Associator : get digilink from the event" << endl;
    
    theStripHits.clear();
    edm::Handle<edm::PSimHitContainer> TIBHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TIBHitsHighTof;
    edm::Handle<edm::PSimHitContainer> TIDHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TIDHitsHighTof;
    edm::Handle<edm::PSimHitContainer> TOBHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TOBHitsHighTof;
    edm::Handle<edm::PSimHitContainer> TECHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TECHitsHighTof;
    
    e.getByLabel("SimG4Object","TrackerHitsTIBLowTof", TIBHitsLowTof);
    e.getByLabel("SimG4Object","TrackerHitsTIBHighTof", TIBHitsHighTof);
    e.getByLabel("SimG4Object","TrackerHitsTIDLowTof", TIDHitsLowTof);
    e.getByLabel("SimG4Object","TrackerHitsTIDHighTof", TIDHitsHighTof);
    e.getByLabel("SimG4Object","TrackerHitsTOBLowTof", TOBHitsLowTof);
    e.getByLabel("SimG4Object","TrackerHitsTOBHighTof", TOBHitsHighTof);
    e.getByLabel("SimG4Object","TrackerHitsTECLowTof", TECHitsLowTof);
    e.getByLabel("SimG4Object","TrackerHitsTECHighTof", TECHitsHighTof);
    
    theStripHits.insert(theStripHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
    theStripHits.insert(theStripHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
    theStripHits.insert(theStripHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
    theStripHits.insert(theStripHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());
    
    SimHitMap.clear();
    for (std::vector<PSimHit>::iterator isim = theStripHits.begin();
	 isim != theStripHits.end(); ++isim){
      SimHitMap[(*isim).detUnitId()].push_back((*isim));
    }

    thePixelHits.clear();
    
    edm::Handle<edm::PSimHitContainer> PixelBarrelHitsLowTof;
    edm::Handle<edm::PSimHitContainer> PixelBarrelHitsHighTof;
    edm::Handle<edm::PSimHitContainer> PixelEndcapHitsLowTof;
    edm::Handle<edm::PSimHitContainer> PixelEndcapHitsHighTof;
    
    e.getByLabel("SimG4Object","TrackerHitsPixelBarrelLowTof", PixelBarrelHitsLowTof);
    e.getByLabel("SimG4Object","TrackerHitsPixelBarrelHighTof", PixelBarrelHitsHighTof);
    e.getByLabel("SimG4Object","TrackerHitsPixelEndcapLowTof", PixelEndcapHitsLowTof);
    e.getByLabel("SimG4Object","TrackerHitsPixelEndcapHighTof", PixelEndcapHitsHighTof);
    
    thePixelHits.insert(thePixelHits.end(), PixelBarrelHitsLowTof->begin(), PixelBarrelHitsLowTof->end()); 
    thePixelHits.insert(thePixelHits.end(), PixelBarrelHitsHighTof->begin(), PixelBarrelHitsHighTof->end());
    thePixelHits.insert(thePixelHits.end(), PixelEndcapHitsLowTof->begin(), PixelEndcapHitsLowTof->end()); 
    thePixelHits.insert(thePixelHits.end(), PixelEndcapHitsHighTof->begin(), PixelEndcapHitsHighTof->end());
    
  }

}

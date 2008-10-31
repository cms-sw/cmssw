//
// Package:         TrackingTools/RoadSearchHitAccess/test
// Class:           RoadSearchHitDumper.cc
// 
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/07 22:07:33 $
// $Revision: 1.3 $
//

#include <sstream>

#include "TrackingTools/RoadSearchHitAccess/test/RoadSearchHitDumper.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/RingRecord/interface/RingRecord.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

RoadSearchHitDumper::RoadSearchHitDumper(const edm::ParameterSet& conf) {

  // retrieve InputTags for rechits
  matchedStripRecHitsInputTag_ = conf.getParameter<edm::InputTag>("matchedStripRecHits");
  rphiStripRecHitsInputTag_    = conf.getParameter<edm::InputTag>("rphiStripRecHits");
  stereoStripRecHitsInputTag_  = conf.getParameter<edm::InputTag>("stereoStripRecHits");
  pixelRecHitsInputTag_        = conf.getParameter<edm::InputTag>("pixelRecHits");

  ringsLabel_                  = conf.getParameter<std::string>("RingsLabel");

}

RoadSearchHitDumper::~RoadSearchHitDumper(){
}

void RoadSearchHitDumper::analyze(const edm::Event& e, const edm::EventSetup& es){

    
  // get Inputs
  edm::Handle<SiStripRecHit2DCollection> rphiRecHitsHandle;
  e.getByLabel(rphiStripRecHitsInputTag_ ,rphiRecHitsHandle);
  const SiStripRecHit2DCollection* rphiRecHits = rphiRecHitsHandle.product();
  edm::Handle<SiStripRecHit2DCollection> stereoRecHitsHandle;
  e.getByLabel(stereoStripRecHitsInputTag_ ,stereoRecHitsHandle);
  const SiStripRecHit2DCollection* stereoRecHits = stereoRecHitsHandle.product();
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedRecHitsHandle;
  e.getByLabel(matchedStripRecHitsInputTag_ ,matchedRecHitsHandle);
  const SiStripMatchedRecHit2DCollection* matchedRecHits = matchedRecHitsHandle.product();
 
    
  // special treatment for getting pixel collection
  // if collection exists in file, use collection from file
  // if collection does not exist in file, create empty collection
  const SiPixelRecHitCollection *pixelRecHits = 0;
  try {
    edm::Handle<SiPixelRecHitCollection> pixelRecHitsHandle;
    e.getByLabel(pixelRecHitsInputTag_, pixelRecHitsHandle);
    pixelRecHits = pixelRecHitsHandle.product();
  }
  catch (edm::Exception const& x) {
    if ( x.categoryCode() == edm::errors::ProductNotFound ) {
      if ( x.history().size() == 1 ) {
	static const SiPixelRecHitCollection s_empty;
	pixelRecHits = &s_empty;
	edm::LogWarning("RoadSearch") << "Collection SiPixelRecHitCollection with InputTag " << pixelRecHitsInputTag_ << " cannot be found, using empty collection of same type. The RoadSearch algorithm is also fully functional without Pixel RecHits.";
      }
    }
  }

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  const TrackerGeometry *tracker = trackerHandle.product();

  // get rings
  edm::ESHandle<Rings> ringsHandle;
  es.get<RingRecord>().get(ringsLabel_, ringsHandle);
  const Rings *rings = ringsHandle.product();

  std::ostringstream output;

  output<< "Format - hit_number rawid ringindex global_r global_phi global_x global_y global_z " << std::endl;

  output<< "rphi collection size " << rphiRecHits->size() << std::endl;
  output<< "stereo collection size " << stereoRecHits->size() << std::endl;
  output<< "matched collection size " << matchedRecHits->size() << std::endl;
  output<< "pixel collection size " << pixelRecHits->size() << std::endl;

  unsigned int nrphi=0;
  for ( SiStripRecHit2DCollection::DataContainer::const_iterator recHit = rphiRecHits->data().begin() , recHitEnd = rphiRecHits->data().end();
	recHit != recHitEnd; 
	++recHit ) {
    DetId id = recHit->geographicalId();
    GlobalPoint outer = tracker->idToDet(id)->surface().toGlobal(recHit->localPosition());
    const Ring *ring = rings->getRing(id);
    ++nrphi; 
    output<< "rphi hit " << nrphi << " " << id.rawId() << " " << ring->getindex() << " "
	  << outer.perp() << " " << outer.phi() 
	  << " " << outer.x() << " " << outer.y() << " " << outer.z() << std::endl;
  }//end of loop over rphi hits

  unsigned int nstereo=0;
  for ( SiStripRecHit2DCollection::DataContainer::const_iterator recHit = stereoRecHits->data().begin() , recHitEnd = stereoRecHits->data().end();
	recHit != recHitEnd; 
	++recHit ) {
    DetId id = recHit->geographicalId();
    GlobalPoint outer = tracker->idToDet(id)->surface().toGlobal(recHit->localPosition());
    DetId rphiDetId(id.rawId()+1);
    const Ring *ring = rings->getRing(rphiDetId);
    ++nstereo; 
    output<< "stereo hit " << nstereo << " " << id.rawId() << " " << ring->getindex() << " "
	  << outer.perp() << " " << outer.phi() 
	  << " " << outer.x() << " " << outer.y() << " " << outer.z() << std::endl;
  }//end of loop over stereo hits

  unsigned int nmatched=0;
  for ( SiStripMatchedRecHit2DCollection::DataContainer::const_iterator recHit = matchedRecHits->data().begin() , recHitEnd = matchedRecHits->data().end();
	recHit != recHitEnd; 
	++recHit ) {
    DetId id = recHit->geographicalId();
    GlobalPoint outer = tracker->idToDet(id)->surface().toGlobal(recHit->localPosition());
    DetId rphiDetId(id.rawId()+2);
    const Ring *ring = rings->getRing(rphiDetId);
    ++nmatched; 
    output<< "matched hit " << nmatched << " " << id.rawId() << " " << ring->getindex() << " "
	  << outer.perp() << " " << outer.phi() 
	  << " " << outer.x() << " " << outer.y() << " " << outer.z() << std::endl;
  }//end of loop over matched hits
  
  unsigned int npixel=0;
  for ( SiPixelRecHitCollection::DataContainer::const_iterator recHit = pixelRecHits->data().begin() , recHitEnd = pixelRecHits->data().end();
	recHit != recHitEnd; 
	++recHit ) {
    DetId id = recHit->geographicalId();
    GlobalPoint outer = tracker->idToDet(id)->surface().toGlobal(recHit->localPosition());
    const Ring *ring = rings->getRing(id);
    ++npixel; 
    output<< "pixel hit " << npixel << " " << id.rawId() << " " << ring->getindex() << " "
	  << outer.perp() << " " << outer.phi() 
	  << " " << outer.x() << " " << outer.y() << " " << outer.z() << std::endl;
  }//end of loop over pixel hits

  edm::LogInfo("RoadSearchHitDumper") << output.str();

}

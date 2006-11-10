//
// Package:         RecoTracker/RoadSearchHelixMaker
// Class:           RoadSearchHelixMakerAlgorithm
// 
// Description:     
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/04/10 22:06:41 $
// $Revision: 1.6 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMakerAlgorithm.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/LocalPoint.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxFittedHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxTrackCandidatesToTracks.hh"

//using namespace std;

RoadSearchHelixMakerAlgorithm::RoadSearchHelixMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchHelixMakerAlgorithm::~RoadSearchHelixMakerAlgorithm() {
}

void RoadSearchHelixMakerAlgorithm::run(const TrackCandidateCollection* input,
					const edm::EventSetup& es,
					reco::TrackCollection &output)
{

  edm::LogInfo("RoadSearch") << "Input of " << input->size() << " track candidates"; 

  //
  //  no track candidates - nothing to try fitting
  //
  if ( input->empty() ){
    edm::LogInfo("RoadSearch") << "Created " << output.size() << " tracks.";
    return;  
  }

  //
  //  got > 0 track candidate - try fitting
  //

  // get roads
  //  edm::ESHandle<Roads> roads;
  //  es.get<TrackerDigiGeometryRecord>().get(roads);

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);


  //   bool useKF = conf_.getParameter<bool>("UseKF");
  unsigned int trackcandidate_ctr=0;
  // loop over clouds
  for ( TrackCandidateCollection::const_iterator trackcandidate = input->begin(); trackcandidate != input->end(); ++trackcandidate ) {
    TrackCandidate currentCandidate = *trackcandidate;
    ++trackcandidate_ctr;
    unsigned int trackcandidate_number_rechits = 0;
    TrackCandidate::range recHitRange = currentCandidate.recHits();
    for ( TrackCandidate::const_iterator recHit = recHitRange.first; recHit != recHitRange.second; ++recHit ) {
      ++trackcandidate_number_rechits;
    }

    LogDebug("RoadSearch") << "Track candidate number, number of TrackingRecHits = " << trackcandidate_ctr << " " 
			   << trackcandidate_number_rechits; 

    //
    // helix fitting here
    //
    edm::LogInfo("RoadSearch") << "Beware - Use Simple Helix Fitted Tracks only for Debugging Purposes!!" ;

    std::vector<DcxHit*> listohits; 
//    edm::LogInfo("RoadSearch") << "listohits.size() " << listohits.size() ;

    for ( TrackCandidate::const_iterator recHit = recHitRange.first; recHit != recHitRange.second; ++recHit ) {
      const TrackingRecHit* temp_hit = &(*recHit);
      GlobalPoint hit_global_pos = tracker->idToDetUnit(temp_hit->geographicalId())->surface().toGlobal(temp_hit->localPosition());
      //       double rhit=sqrt(hit_global_pos.x()*hit_global_pos.x()+hit_global_pos.y()*hit_global_pos.y());
      DetId idi = temp_hit->geographicalId();
      if (isBarrelSensor(idi)){
	const RectangularStripTopology *topi = 
	  dynamic_cast<const RectangularStripTopology*>(&(tracker->idToDetUnit(idi)->topology()));
	double iLength = topi->stripLength();
	LocalPoint temp_lpos = temp_hit->localPosition();
	LocalPoint temp_lpos_f(temp_lpos.x(),temp_lpos.y()+iLength/2.0,temp_lpos.z());
	LocalPoint temp_lpos_b(temp_lpos.x(),temp_lpos.y()-iLength/2.0,temp_lpos.z());
	GlobalPoint temp_gpos_f = tracker->idToDetUnit(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_f);
	GlobalPoint temp_gpos_b = tracker->idToDetUnit(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_b);
	GlobalVector fir_uvec((temp_gpos_f.x()-temp_gpos_b.x())/iLength,
			      (temp_gpos_f.y()-temp_gpos_b.y())/iLength,(temp_gpos_f.z()-temp_gpos_b.z())/iLength);
//	LogInfo("RoadSearch") << "hit global position  = " << hit_global_pos.x() << " " << hit_global_pos.y() << " " << hit_global_pos.z()
//			      << " " << fir_uvec.x() << " " << fir_uvec.y() << " " << fir_uvec.z() ;
	DcxHit* try_me = new DcxHit(hit_global_pos.x(), hit_global_pos.y(), hit_global_pos.z(), 
				    fir_uvec.x(), fir_uvec.y(), fir_uvec.z());
	listohits.push_back(try_me);
      }else{
	const TrapezoidalStripTopology *topi = 
	  dynamic_cast<const TrapezoidalStripTopology*>(&(tracker->idToDetUnit(idi)->topology()));
	double iLength = topi->stripLength();
	LocalPoint temp_lpos = temp_hit->localPosition();
	LocalPoint temp_lpos_f(temp_lpos.x(),temp_lpos.y()+iLength/2.0,temp_lpos.z());
	LocalPoint temp_lpos_b(temp_lpos.x(),temp_lpos.y()-iLength/2.0,temp_lpos.z());
	GlobalPoint temp_gpos_f = tracker->idToDetUnit(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_f);
	GlobalPoint temp_gpos_b = tracker->idToDetUnit(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_b);
	GlobalVector fir_uvec((temp_gpos_f.x()-temp_gpos_b.x())/iLength,
			      (temp_gpos_f.y()-temp_gpos_b.y())/iLength,(temp_gpos_f.z()-temp_gpos_b.z())/iLength);
//	LogInfo("RoadSearch") << "hit global position  = " << hit_global_pos.x() << " " << hit_global_pos.y() << " " << hit_global_pos.z()
//			      << " " << fir_uvec.x() << " " << fir_uvec.y() << " " << fir_uvec.z() ;
//        DcxHit* try_me = new DcxHit(hit_global_pos.x(), hit_global_pos.y(), hit_global_pos.z(), 
//                                    fir_uvec.x(), fir_uvec.y(), fir_uvec.z());
//        listohits.push_back(try_me);
      }//make DcxHit from Barrel or Endcap sensor
    }
//    edm::LogInfo("RoadSearch") << "finished DcxHit making; listohits.size() " << listohits.size() ;
    DcxTrackCandidatesToTracks make_tracks(listohits,output);
//      listohits.~vector<DcxHit*>();// leak or crash - right now we're leaking
  }//iterate over all track candidates

  edm::LogInfo("RoadSearch") << "Created " << output.size() << " tracks.";

}

bool RoadSearchHelixMakerAlgorithm::isBarrelSensor(DetId id) {

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    return true;
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    return true;
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
    return true;
  } else {
    return false;
  }

}

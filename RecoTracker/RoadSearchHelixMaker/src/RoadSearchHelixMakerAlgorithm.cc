//
// Package:         RecoTracker/RoadSearchHelixMaker
// Class:           RoadSearchHelixMakerAlgorithm
// 
// Description:     
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: gpetrucc $
// $Date: 2010/12/14 15:50:20 $
// $Revision: 1.11 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMakerAlgorithm.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxFittedHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxTrackCandidatesToTracks.hh"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

RoadSearchHelixMakerAlgorithm::RoadSearchHelixMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchHelixMakerAlgorithm::~RoadSearchHelixMakerAlgorithm() {
}

void RoadSearchHelixMakerAlgorithm::run(const TrackCandidateCollection* input,
					const edm::EventSetup& es,
					reco::TrackCollection &output)
{

  edm::LogInfo("RoadSearch") << "Input of " << input->size() << " track candidate(s)."; 

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

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  // magnetic field
  edm::ESHandle<MagneticField> fieldHandle;
  es.get<IdealMagneticFieldRecord>().get(fieldHandle);
  const MagneticField *field = fieldHandle.product();

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

    for ( TrackCandidate::const_iterator recHit = recHitRange.first; recHit != recHitRange.second; ++recHit ) {
      DetId recHitId = recHit->geographicalId();
      const GeomDet *recHitGeomDet = tracker->idToDet(recHitId);
      GlobalPoint hit_global_pos = recHitGeomDet->surface().toGlobal(recHit->localPosition());
      // only for TIB and TOB (for now ... )
      if ( (unsigned int)recHitId.subdetId() == StripSubdetector::TIB ||
	   (unsigned int)recHitId.subdetId() == StripSubdetector::TOB ) {
	const GeomDetUnit *recHitGeomDetUnit;
	// take rphi sensor in case of matched rechit to determine topology
	const GluedGeomDet *recHitGluedGeomDet = dynamic_cast<const GluedGeomDet*>(recHitGeomDet);
	if ( recHitGluedGeomDet != 0 ) {
	  recHitGeomDetUnit = recHitGluedGeomDet->monoDet();
	} else {
	  recHitGeomDetUnit = tracker->idToDetUnit(recHitId);
	}
	const StripTopology *recHitTopology = dynamic_cast<const StripTopology*>(&(recHitGeomDetUnit->topology()));
	double iLength = recHitTopology->stripLength();
	LocalPoint temp_lpos = recHit->localPosition();
	LocalPoint temp_lpos_f(temp_lpos.x(),temp_lpos.y()+iLength/2.0,temp_lpos.z());
	LocalPoint temp_lpos_b(temp_lpos.x(),temp_lpos.y()-iLength/2.0,temp_lpos.z());
	GlobalPoint temp_gpos_f = recHitGeomDet->surface().toGlobal(temp_lpos_f);
	GlobalPoint temp_gpos_b = recHitGeomDet->surface().toGlobal(temp_lpos_b);
	GlobalVector fir_uvec((temp_gpos_f.x()-temp_gpos_b.x())/iLength,
			      (temp_gpos_f.y()-temp_gpos_b.y())/iLength,(temp_gpos_f.z()-temp_gpos_b.z())/iLength);
	DcxHit* try_me = new DcxHit(hit_global_pos.x(), hit_global_pos.y(), hit_global_pos.z(), 
				    fir_uvec.x(), fir_uvec.y(), fir_uvec.z());
	listohits.push_back(try_me);
      }
    }
    DcxTrackCandidatesToTracks make_tracks(listohits,output, field);
  }//iterate over all track candidates

  edm::LogInfo("RoadSearch") << "Created " << output.size() << " tracks.";

}

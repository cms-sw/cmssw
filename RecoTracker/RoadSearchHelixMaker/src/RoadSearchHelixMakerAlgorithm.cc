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
// $Date: 2006/02/13 19:32:28 $
// $Revision: 1.4 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMakerAlgorithm.h"

#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseSiStripRecHit2DLocalPos.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/LocalPoint.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxFittedHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxCloudsToTracks.hh"

using namespace std;

RoadSearchHelixMakerAlgorithm::RoadSearchHelixMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchHelixMakerAlgorithm::~RoadSearchHelixMakerAlgorithm() {
}

void RoadSearchHelixMakerAlgorithm::run(const RoadSearchCloudCollection* input,
			      const edm::EventSetup& es,
			      reco::TrackCollection &output)
{

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    std::cout << "[RoadSearchHelixMakerAlgorithm] has " << input->size() << " clean clouds" << std::endl; 
  }

//
//  no clean clouds - nothing to try fitting
//
  if ( input->empty() ){
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      std::cout << "[RoadSearchHelixMakerAlgorithm] make " << output.size() << " tracks." << std::endl;
    }
    return;  
  }

//
//  got > 0 clean cloud - try fitting
//

  // get roads
//  edm::ESHandle<Roads> roads;
//  es.get<TrackerDigiGeometryRecord>().get(roads);

  // get tracker geometry
  edm::ESHandle<TrackingGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);


  bool useKF = conf_.getParameter<bool>("UseKF");
  int clean_cloud_ctr=0;
  // loop over clouds
  for ( RoadSearchCloudCollection::const_iterator clean_cloud = input->begin(); clean_cloud != input->end(); ++clean_cloud) {
    clean_cloud_ctr++;
    std::cout << "[RoadSearchHelixMakerAlgorithm] cloud number, size = " << clean_cloud_ctr << " " 
              << clean_cloud->size() << std::endl; 

    if (useKF){
//
//  standard KF code here
//
      std::cout << "KF loop empty, for now" << std::endl;

    }else{
//
// helix fitting here
//
      std::cout << "beware - are helix fitting tracks" << std::endl;

      std::vector<const SiStripRecHit2DLocalPos*> clean_hits = clean_cloud->detHits();
      std::vector<DcxHit*> listohits; 
//      cout << "listohits.size() " << listohits.size() << endl;
      for (unsigned int i=0; i<clean_hits.size(); ++i) {
        SiStripRecHit2DLocalPos* temp_hit = const_cast<SiStripRecHit2DLocalPos*>(clean_hits[i]);
        GlobalPoint hit_global_pos = tracker->idToDet(temp_hit->geographicalId())->surface().toGlobal(temp_hit->localPosition());
        double rhit=sqrt(hit_global_pos.x()*hit_global_pos.x()+hit_global_pos.y()*hit_global_pos.y());
	DetId idi = temp_hit->geographicalId();
        if (isBarrelSensor(idi)){
          const RectangularStripTopology *topi = 
                                       dynamic_cast<const RectangularStripTopology*>(&(tracker->idToDet(idi)->topology()));
	  double iLength = topi->stripLength();
          LocalPoint temp_lpos = temp_hit->localPosition();
          LocalPoint temp_lpos_f(temp_lpos.x(),temp_lpos.y()+iLength/2.0,temp_lpos.z());
          LocalPoint temp_lpos_b(temp_lpos.x(),temp_lpos.y()-iLength/2.0,temp_lpos.z());
          GlobalPoint temp_gpos_f = tracker->idToDet(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_f);
          GlobalPoint temp_gpos_b = tracker->idToDet(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_b);
          GlobalVector fir_uvec((temp_gpos_f.x()-temp_gpos_b.x())/iLength,
                                (temp_gpos_f.y()-temp_gpos_b.y())/iLength,(temp_gpos_f.z()-temp_gpos_b.z())/iLength);
//          std::cout << "hit global position  = " << hit_global_pos.x() << " " << hit_global_pos.y() << " " << hit_global_pos.z()
//                    << " " << fir_uvec.x() << " " << fir_uvec.y() << " " << fir_uvec.z() << std::endl;
          DcxHit* try_me = new DcxHit(hit_global_pos.x(), hit_global_pos.y(), hit_global_pos.z(), 
                                    fir_uvec.x(), fir_uvec.y(), fir_uvec.z());
          listohits.push_back(try_me);
        }else{
          const TrapezoidalStripTopology *topi = 
                                       dynamic_cast<const TrapezoidalStripTopology*>(&(tracker->idToDet(idi)->topology()));
	  double iLength = topi->stripLength();
          LocalPoint temp_lpos = temp_hit->localPosition();
          LocalPoint temp_lpos_f(temp_lpos.x(),temp_lpos.y()+iLength/2.0,temp_lpos.z());
          LocalPoint temp_lpos_b(temp_lpos.x(),temp_lpos.y()-iLength/2.0,temp_lpos.z());
          GlobalPoint temp_gpos_f = tracker->idToDet(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_f);
          GlobalPoint temp_gpos_b = tracker->idToDet(temp_hit->geographicalId())->surface().toGlobal(temp_lpos_b);
          GlobalVector fir_uvec((temp_gpos_f.x()-temp_gpos_b.x())/iLength,
                                (temp_gpos_f.y()-temp_gpos_b.y())/iLength,(temp_gpos_f.z()-temp_gpos_b.z())/iLength);
          std::cout << "hit global position  = " << hit_global_pos.x() << " " << hit_global_pos.y() << " " << hit_global_pos.z()
                    << " " << fir_uvec.x() << " " << fir_uvec.y() << " " << fir_uvec.z() << std::endl;
//        DcxHit* try_me = new DcxHit(hit_global_pos.x(), hit_global_pos.y(), hit_global_pos.z(), 
//                                    fir_uvec.x(), fir_uvec.y(), fir_uvec.z());
//        listohits.push_back(try_me);
        }//make DcxHit from Barrel or Endcap sensor
      }
//      cout << "finished DcxHit making; listohits.size() " << listohits.size() << endl;
      DcxCloudsToTracks make_tracks(listohits);
//      listohits.~vector<DcxHit*>();// leak or crash - right now we're leaking
    }//end fit the cloud (useKF or not)
  }//iterate over all clean clouds

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    std::cout << "[RoadSearchHelixMakerAlgorithm] made " << output.size() << " tracks." << std::endl;
  }

};

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


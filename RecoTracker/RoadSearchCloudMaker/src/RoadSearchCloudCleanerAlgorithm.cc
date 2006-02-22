//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudCleanerAlgorithm
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

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudCleanerAlgorithm.h"

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

using namespace std;

RoadSearchCloudCleanerAlgorithm::RoadSearchCloudCleanerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchCloudCleanerAlgorithm::~RoadSearchCloudCleanerAlgorithm() {
}

void RoadSearchCloudCleanerAlgorithm::run(const RoadSearchCloudCollection* input,
                              const TrackingSeedCollection* seeds,
			      const edm::EventSetup& es,
			      RoadSearchCloudCollection &output)
{

//
//  right now cloud cleaning solely consist of merging clouds based on the number
//  of shared hits (getting rid of obvious duplicates) - don't need roads and
//  geometry for this - eventually this stage will become a sub-process (probably
//  early on) of cloud cleaning - but for now that's all, folks
//

  // get roads
//  edm::ESHandle<Roads> roads;
//  es.get<TrackerDigiGeometryRecord>().get(roads);

  // get tracker geometry
//  edm::ESHandle<TrackingGeometry> tracker;
//  es.get<TrackerDigiGeometryRecord>().get(tracker);

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    std::cout << "[RoadSearchCloudCleanerAlgorithm] has " << input->size() << " raw clouds" << std::endl; 
  }

//
//  no raw clouds - nothing to try merging
//
  if ( input->empty() ){
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      std::cout << "[RoadSearchCloudCleanerAlgorithm] found " << output.size() << " clouds." << std::endl;
    }
    return;  
  }
//
//  1 raw cloud - nothing to try merging, but one cloud to duplicate
//
  if ( 1==input->size() ){
    RoadSearchCloud lone_cloud;
    for ( RoadSearchCloudCollection::const_iterator raw_cloud = input->begin(); raw_cloud != input->end(); ++raw_cloud) {
      std::vector<const SiStripRecHit2DLocalPos*> raw_hits = raw_cloud->detHits();
      for (unsigned int i=0; i<raw_hits.size(); ++i) {
        SiStripRecHit2DLocalPos* temp_hit = const_cast<SiStripRecHit2DLocalPos*>(raw_hits[i]);
        lone_cloud.addHit(temp_hit);
      }
    }
    output.push_back(lone_cloud);
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      std::cout << "[RoadSearchCloudCleanerAlgorithm] found " << output.size() << " clouds." << std::endl;
    }
    return;
  }  
//
//  got > 1 raw cloud - something to try merging
//
  double mergingFraction = conf_.getParameter<double>("MergingFraction");
  unsigned int maxRecHitsInCloud = conf_.getParameter<int>("MaxRecHitsInCloud");
  std::vector<bool> dont_merge;
  std::vector<bool> already_gone;
  for (unsigned int i=0; i<input->size(); ++i) {
//    if (1==i){
      already_gone.push_back(false); 
//    }else{ 
//      already_gone.push_back(true); 
//    }
    dont_merge.push_back(false);
  }
  int raw_cloud_ctr=0;
  // loop over clouds
  for ( RoadSearchCloudCollection::const_iterator raw_cloud = input->begin(); raw_cloud != input->end(); ++raw_cloud) {
    raw_cloud_ctr++;
    if (already_gone[raw_cloud_ctr-1])continue;
    RoadSearchCloud lone_cloud;
    std::vector<const SiStripRecHit2DLocalPos*> raw_hits = raw_cloud->detHits();
    for (unsigned int i=0; i<raw_hits.size(); ++i) {
      SiStripRecHit2DLocalPos* temp_hit = const_cast<SiStripRecHit2DLocalPos*>(raw_hits[i]);
      lone_cloud.addHit(temp_hit);
    }

//    std::cout << " got to RoadSearchCloudCleanerAlgorithm::run in raw_cloud loop; cloud size = "  
//              << raw_cloud->size() << std::endl; 

    int second_cloud_ctr=0;
    for ( RoadSearchCloudCollection::const_iterator second_cloud = input->begin(); second_cloud != input->end(); ++second_cloud) {
      second_cloud_ctr++;
      if ( already_gone[second_cloud_ctr-1] || dont_merge[raw_cloud_ctr-1] )continue;
      if (second_cloud_ctr > raw_cloud_ctr){
//        std::cout << " raw_cloud #, size, second_cloud #, size = " << raw_cloud_ctr << " " 
//                  << raw_cloud->size() << " " << second_cloud_ctr << " " << second_cloud->size() << std::endl;

        std::vector<const SiStripRecHit2DLocalPos*> lone_cloud_hits = lone_cloud.detHits();
        std::vector<const SiStripRecHit2DLocalPos*> second_cloud_hits = second_cloud->detHits();
        std::vector<const SiStripRecHit2DLocalPos*> unsharedhits;
        for (unsigned int j=0; j<second_cloud_hits.size(); ++j) {
          bool is_shared=false;
          for (unsigned int i=0; i<lone_cloud_hits.size(); ++i) {
            if (lone_cloud_hits[i]==second_cloud_hits[j]){is_shared=true; break;}
          }
          if (!is_shared)unsharedhits.push_back(second_cloud_hits[j]);
        }

        float f_lone_shared=float(second_cloud->size()-unsharedhits.size())/float(lone_cloud.size());
        float f_second_shared=float(second_cloud->size()-unsharedhits.size())/float(second_cloud->size());
//        if ((f_lone_shared != 1.0)||(f_second_shared != 1.0)){std::cout << "lookey here";}
//        std::cout << " f_lone_shared, f_second_shared = " << f_lone_shared << " " << f_second_shared << std::endl;
        if ( ( (f_lone_shared > mergingFraction)||(f_second_shared > mergingFraction) ) 
                       && (lone_cloud.size()+unsharedhits.size() <= maxRecHitsInCloud) ){

//
//  got a cloud to merge
//
          for (unsigned int k=0; k<unsharedhits.size(); ++k) {
            SiStripRecHit2DLocalPos* temp_hit = const_cast<SiStripRecHit2DLocalPos*>(unsharedhits[k]);
            lone_cloud.addHit(temp_hit);
          }
          already_gone[second_cloud_ctr-1]=true;
//          std::cout << "merge em " << unsharedhits.size() << " " << raw_cloud->size()+unsharedhits.size() <<std::endl;
        }//end got a cloud to merge
      }//second cloud acceptable for inspection
    }//interate over all second clouds
    output.push_back(lone_cloud);
  }//iterate over all raw clouds

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    std::cout << "[RoadSearchCloudCleanerAlgorithm] found " << output.size() << " clouds." << std::endl;
  }

};



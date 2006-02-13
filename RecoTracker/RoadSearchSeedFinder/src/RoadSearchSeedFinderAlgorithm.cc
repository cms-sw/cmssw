//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchSeedFinderAlgorithm
// 
// Description:     Loops over Roads, checks for every
//                  RoadSeed if hits are in the inner and
//                  outer SeedRing, applies cuts for all 
//                  combinations of inner and outer SeedHits,
//                  stores valid combination in TrackingSeed
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/02/08 02:06:48 $
// $Revision: 1.3 $
//

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinderAlgorithm.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

RoadSearchSeedFinderAlgorithm::RoadSearchSeedFinderAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchSeedFinderAlgorithm::~RoadSearchSeedFinderAlgorithm() {
}


void RoadSearchSeedFinderAlgorithm::run(const edm::Handle<SiStripRecHit2DMatchedLocalPosCollection> &handle,
                                        const edm::Handle<SiStripRecHit2DLocalPosCollection> &handle2,
			      const edm::EventSetup& es,
			      TrackingSeedCollection &output)
{

  const SiStripRecHit2DMatchedLocalPosCollection* input = handle.product();
  const SiStripRecHit2DLocalPosCollection* input2 = handle2.product();

  const std::vector<unsigned int> availableIDs = input->detIDs();
  const std::vector<unsigned int> availableIDs2 = input2->detIDs();

  // get roads
  edm::ESHandle<Roads> roads;
  es.get<TrackerDigiGeometryRecord>().get(roads);

  // get tracker geometry
  edm::ESHandle<TrackingGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  // calculate maximal possible delta phi for given delta r and parameter pTmin
  double ptmin = conf_.getParameter<double>("MinimalReconstructedTransverseMomentum");

  // correction for B given in T, delta r given in cm, ptmin given in GeV/c
  double speedOfLight = 2.99792458e8;
  double unitCorrection = speedOfLight * 1e-2 * 1e-9;

  // B in T, right now hardcoded, has to come from magnetic field service
  double B = 4.0;

  // loop over seed Ring pairs
  for ( Roads::const_iterator road = roads->begin(); road != roads->end(); ++road ) {

    Roads::RoadSeed seed = (*road).first;

    // loop over detid's in seed rings
    for ( Ring::const_iterator innerRingDetId = seed.first.begin(); innerRingDetId != seed.first.end(); ++innerRingDetId ) {

      if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),innerRingDetId->second.rawId()) ) {

	for ( Ring::const_iterator outerRingDetId = seed.second.begin(); outerRingDetId != seed.second.end(); ++outerRingDetId ) {

	  if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),outerRingDetId->second.rawId()) ) {

	    SiStripRecHit2DMatchedLocalPosCollection::Range innerSeedDetHits = input->get(innerRingDetId->second.rawId());
      
	    // loop over inner dethits
	    for ( SiStripRecHit2DMatchedLocalPosCollection::ContainerConstIterator innerSeedDetHit = innerSeedDetHits.first;
		  innerSeedDetHit != innerSeedDetHits.second; ++innerSeedDetHit ) {

	      SiStripRecHit2DLocalPosCollection::Range outerSeedDetHits = input2->get(outerRingDetId->second.rawId());

	      for ( SiStripRecHit2DLocalPosCollection::ContainerConstIterator outerSeedDetHit = outerSeedDetHits.first;
		    outerSeedDetHit != outerSeedDetHits.second; ++outerSeedDetHit ) {

		GlobalPoint inner = tracker->idToDet(innerSeedDetHit->geographicalId())->surface().toGlobal(innerSeedDetHit->localPosition());
		GlobalPoint outer = tracker->idToDet(outerSeedDetHit->geographicalId())->surface().toGlobal(outerSeedDetHit->localPosition());

		// calculate deltaPhi in [0,2pi]
		double deltaPhi = std::abs(inner.phi() - outer.phi());
		double pi = 3.14159265358979312;
		if ( deltaPhi < 0 ) deltaPhi = 2*pi - deltaPhi;
	    
		double innerr = std::sqrt(inner.x()*inner.x()+inner.y()*inner.y());
		double outerr = std::sqrt(outer.x()*outer.x()+outer.y()*outer.y());
//                std::cout << "innerr, outerr " << innerr << " " << outerr << std::endl;

		// calculate maximal delta phi in [0,2pi]
		double deltaPhiMax = std::abs( std::asin(unitCorrection * B * innerr / ptmin) - std::asin(unitCorrection * B * outerr / ptmin) );
		if ( deltaPhiMax < 0 ) deltaPhiMax = 2*pi - deltaPhiMax;

		if ( deltaPhi <= deltaPhiMax ) {
	      
		  // add dethits passing deltaPhi cut, first inner, second outer
		  TrackingSeed productSeed;
		  productSeed.addHit(&(*innerSeedDetHit));
		  productSeed.addHit(&(*outerSeedDetHit));
	      
		  // add seed to collection
		  output.push_back(productSeed);

		}
	      }
	    }
	  }
	}
      }
    }
  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    std::cout << "[RoadSearchSeedFinderAlgorithm] found " << output.size() << " seeds." << std::endl; 
  }

};

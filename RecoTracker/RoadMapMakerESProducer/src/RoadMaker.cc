//
// Package:         RecoTracker/RoadMapMakerESProducer
// Class:           RoadMaker
// 
// Description:     Creates a Roads object by combining all
//                  inner and outer SeedRings into RoadSeeds
//                  and determines all Rings of the RoadSet
//                  belonging to the RoadSeeds.     
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Jan 12 21:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/30 02:49:39 $
// $Revision: 1.11 $
//

#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>

#include "RecoTracker/RingRecord/interface/SortRingsByZR.h"
#include "RecoTracker/RingRecord/interface/SortLayersByZR.h"

#include "RecoTracker/RoadMapMakerESProducer/interface/RoadMaker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

RoadMaker::RoadMaker(const Rings *rings,
		     RoadMaker::GeometryStructure structure,
		     RoadMaker::SeedingType seedingType) : 
  rings_(rings), 
  structure_(structure),
  seedingType_(seedingType) {

    // define beamspot z extnsion
    zBS_ = 0;
    if(structure_==FullDetector) {
      zBS_ = 3*5.3; // cm
    } else if (structure_==MTCC) {
      zBS_ = 3*75.; // cm
    } else if (structure_==TIF) {
      zBS_ = 3*75.; // cm
    } else if (structure_==TIFTIB) {
      zBS_ = 5*75.; // cm
    } else if (structure_==TIFTOB) {
      zBS_ = 5*75.; // cm
    } else if (structure_==TIFTIBTOB) {
      zBS_ = 5*75.; // cm
    } else if (structure_==TIFTOBTEC) {
      zBS_ = 5*75.; // cm
    }

    roads_ = new Roads();

    constructRoads();

  }

RoadMaker::~RoadMaker() { 
  
}

void RoadMaker::constructRoads() {

  // fill vector of inner Rings
  collectInnerSeedRings();
  collectInnerSeedRings1();
  collectInnerSeedRings2();

  // fill vector of outer Rings
  collectOuterSeedRings();
  collectOuterSeedRings1();

  if ( seedingType_ == TwoRingSeeds ) {
    // loop over inner-outer ring pairs
    for ( std::vector<const Ring*>::iterator innerRing = innerSeedRings_.begin(); 
	  innerRing != innerSeedRings_.end(); 
	  ++innerRing ) {
      // draw lines (z = a*r + b) through all corners and beam spot
      std::vector<std::pair<double,double> > linesInnerSeedRingAndBS = LinesThroughRingAndBS((*innerRing));
      for ( std::vector<const Ring*>::iterator outerRing = outerSeedRings_.begin(); 
	    outerRing != outerSeedRings_.end(); 
	    ++outerRing ) {
	// check if ring is compatible with extrapolation of lines
	if ( CompatibleWithLines(linesInnerSeedRingAndBS, (*outerRing) ) ) {
	  // construct road
	  Roads::RoadSeed seed;

	  // fill first inner seed ring
	  seed.first.push_back((*innerRing));

	  // fill first outer seed ring
	  seed.second.push_back((*outerRing));

	  Roads::RoadSet set = RingsCompatibleWithSeed(seed);

	  // sort seeds
	  std::sort(seed.first.begin(),seed.first.end(),SortRingsByZR());
	  std::sort(seed.second.begin(),seed.second.end(),SortRingsByZR());
	    
	  roads_->insert(seed,set);
	}
      }
    }
    
  } else if (seedingType_ == FourRingSeeds ) {

    // loop over inner-outer ring pairs
    for ( std::vector<const Ring*>::iterator innerRing1 = innerSeedRings1_.begin(); 
	  innerRing1 != innerSeedRings1_.end(); 
	  ++innerRing1 ) {
      
      // draw lines (z = a*r + b) through all corners and beam spot
      std::vector<std::pair<double,double> > linesInnerSeedRing1AndBS = LinesThroughRingAndBS((*innerRing1));
      
      for ( std::vector<const Ring*>::iterator innerRing2 = innerSeedRings2_.begin(); 
	    innerRing2 != innerSeedRings2_.end(); 
	    ++innerRing2 ) {
	
	// check if ring is compatible with extrapolation of lines
	if ( CompatibleWithLines(linesInnerSeedRing1AndBS, (*innerRing2) ) ) {

	  // draw lines (z = a*r + b) through all corners and beam spot
	  std::vector<std::pair<double,double> > linesInnerSeedRing2AndBS = LinesThroughRingAndBS((*innerRing2));

	  // draw lines (z = a*r + b) through all corners of both inner seed rings
	  std::vector<std::pair<double,double> > linesInnerSeedRing1AndInnerSeedRing2 = LinesThroughRings(*innerRing1,*innerRing2);

	  for ( std::vector<const Ring*>::iterator outerRing1 = outerSeedRings1_.begin(); 
		outerRing1 != outerSeedRings1_.end(); 
		++outerRing1 ) {

	    // check if ring is compatible with extrapolation of lines
	    if ( CompatibleWithLines(linesInnerSeedRing1AndBS, (*outerRing1) ) &&
		 CompatibleWithLines(linesInnerSeedRing2AndBS, (*outerRing1) ) &&
		 CompatibleWithLines(linesInnerSeedRing1AndInnerSeedRing2, (*outerRing1) )  ) {

	      std::vector<std::pair<double,double> > linesOuterSeedRing1AndBS = LinesThroughRingAndBS((*outerRing1));
	      std::vector<std::pair<double,double> > linesInnerSeedRing1AndOuterSeedRing1 = LinesThroughRings(*innerRing1,*outerRing1);
	      std::vector<std::pair<double,double> > linesInnerSeedRing2AndOuterSeedRing1 = LinesThroughRings(*innerRing2,*outerRing1);

	      // construct road
	      Roads::RoadSeed seed;

	      // fill first inner seed ring
	      seed.first.push_back((*innerRing1));
	      seed.first.push_back((*innerRing2));

	      // fill first outer seed ring
	      seed.second.push_back((*outerRing1));


	      std::vector<const Ring*> outerRing2Candidates;
	      // try to add second outer seed ring from all outer seed rings if compatible with other three seeds
	      for ( std::vector<const Ring*>::iterator outerRing = outerSeedRings_.begin(); 
		    outerRing != outerSeedRings_.end(); 
		    ++outerRing ) {
		if ( CompatibleWithLines(linesInnerSeedRing1AndBS, (*outerRing) ) &&
		     CompatibleWithLines(linesInnerSeedRing2AndBS, (*outerRing) ) &&
		     CompatibleWithLines(linesOuterSeedRing1AndBS, (*outerRing) ) &&
		     CompatibleWithLines(linesInnerSeedRing1AndInnerSeedRing2, (*outerRing) ) &&
		     CompatibleWithLines(linesInnerSeedRing1AndOuterSeedRing1, (*outerRing) ) &&
		     CompatibleWithLines(linesInnerSeedRing2AndOuterSeedRing1, (*outerRing) ) ) {
		  if ( (*outerRing)->getindex() < (*outerRing1)->getindex() ) {
		    if ( !RingsOnSameLayer((*outerRing),(*outerRing1)) ) {
		      outerRing2Candidates.push_back((*outerRing));
		    }
		  }
		}
	      }

	      unsigned int size = outerRing2Candidates.size();
	      if ( size > 0 ) {
		if ( size == 1 ) {
		  seed.second.push_back(outerRing2Candidates[0]);
		} else {
		  // extrapolate center of outerRing1 throigh 0,0 to candidates
		  // candidate with center clostest to extrapolation willl be taken
		  const Ring *selectedRing = 0;
		  double z_2 = ((*outerRing1)->getzmax()+(*outerRing1)->getzmin())/2;
		  double r_2 = ((*outerRing1)->getrmax()+(*outerRing1)->getrmin())/2;
		  double z_1 = ((*innerRing1)->getzmax()+(*innerRing1)->getzmin())/2;
		  double r_1 = ((*innerRing1)->getrmax()+(*innerRing1)->getrmin())/2;
		  if ( RingInBarrel(*outerRing1) ) {
		    double slope = (z_2-z_1) /(r_2-r_1);
		    double intercept = z_1 - slope * r_1;
		    double zDifference = 999;
		    for ( std::vector<const Ring*>::iterator ring = outerRing2Candidates.begin();
			  ring != outerRing2Candidates.end();
			  ++ring ) {
		      double z = slope * (((*ring)->getrmax()+(*ring)->getrmin())/2) + intercept;
		      double diff = std::abs(z-(((*ring)->getzmax()+(*ring)->getzmin())/2));
		      if ( diff < zDifference ) {
			selectedRing = *ring;
			zDifference = diff;
		      }
		    }
		  } else {
		    double slope = (r_2-r_1) /(z_2-z_1);
		    double intercept = r_1 - slope * z_1;
		    double rDifference = 999;
		    for ( std::vector<const Ring*>::iterator ring = outerRing2Candidates.begin();
			  ring != outerRing2Candidates.end();
			  ++ring ) {
		      double r = slope * (((*ring)->getzmax()+(*ring)->getzmin())/2) + intercept;
		      double diff = std::abs(r-(((*ring)->getrmax()+(*ring)->getrmin())/2));
		      if ( diff < rDifference ) {
			selectedRing = *ring;
			rDifference = diff;
		      }
		    }
		  }
		  seed.second.push_back(selectedRing);
		}
	      }

	      
	      // collect all rings compatible with temporary seed from inner seed ring 1 and outer seed ring 1
	      Roads::RoadSeed tempSeed;
	      tempSeed.first.push_back((*innerRing1));
	      tempSeed.second.push_back((*outerRing1));
	      Roads::RoadSet set = RingsCompatibleWithSeed(tempSeed);

	      // sort seeds
	      std::sort(seed.first.begin(),seed.first.end(),SortRingsByZR());
	      std::sort(seed.second.begin(),seed.second.end(),SortRingsByZR());
	    
	      roads_->insert(seed,set);

	    }
	  }
	}
      }
    }
  }
  
  edm::LogInfo("RoadSearch") << "Constructed " << roads_->size() << " roads.";

}

void RoadMaker::collectInnerSeedRings() {

  if(structure_==FullDetector) {
    collectInnerTIBSeedRings();
    collectInnerTIDSeedRings();
    collectInnerTECSeedRings(); 
  } else if(structure_==MTCC) {
    collectInnerTIBSeedRings();
  } else if(structure_==TIF) {
    collectInnerTIBSeedRings();
    collectInnerTOBSeedRings();
    collectInnerTIDSeedRings();
    collectInnerTECSeedRings(); 
  } else if(structure_==TIFTOB) {
    collectInnerTOBSeedRings();
  } else if(structure_==TIFTIB) {
    collectInnerTIBSeedRings();
    collectInnerTIDSeedRings();
  } else if(structure_==TIFTIBTOB) {
    collectInnerTIBSeedRings();
    collectInnerTIDSeedRings();
    collectInnerTOBSeedRings();
  } else if(structure_==TIFTOBTEC) {
    collectInnerTOBSeedRings();
    collectInnerTECSeedRings();
  }

  LogDebug("RoadSearch") << "collected " << innerSeedRings_.size() << " inner seed rings"; 
  
}

void RoadMaker::collectInnerTIBSeedRings() {

  // TIB
  unsigned int counter = 0, layer_min=0, layer_max=0, fw_bw_min=0, fw_bw_max=0, ext_int_min=0, ext_int_max=0, detector_min = 0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min     = 1;
    layer_max     = 3;
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4;
  } else if(structure_==MTCC) {
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 2;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if (structure_==TIF) { 
    layer_min     = 1;
    layer_max     = 3; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if (structure_==TIFTIB) { 
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if (structure_==TIFTIBTOB) { 
    layer_min     = 1;
    layer_max     = 3; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  }


  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = ext_int_min; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	  const Ring* temp_ring = rings_->getTIBRing(layer,fw_bw,ext_int,detector);
	  innerSeedRings_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TIB inner seed ring with index: " << temp_ring->getindex(); 
	}    
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TIB inner seed rings"; 

}

void RoadMaker::collectInnerTIDSeedRings() {

  // TID
  unsigned int counter = 0;

  unsigned int fw_bw_min =0, fw_bw_max = 0,wheel_min = 0, wheel_max = 0, ring_min=0, ring_max = 0;

  if(structure_==FullDetector) {
    fw_bw_min       = 1;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 3;
  } else if(structure_==TIF) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 3;
  } else if(structure_==TIFTIBTOB) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 3;
  } else if(structure_==TIFTIB) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 2;
  }

  for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = ring_min; ring < ring_max; ++ring ) {
	const Ring* temp_ring = rings_->getTIDRing(fw_bw,wheel,ring);
	innerSeedRings_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TID inner seed ring with index: " << temp_ring->getindex(); 
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TID inner seed rings"; 

}

void RoadMaker::collectInnerTECSeedRings() {

  // TEC
  unsigned int counter = 0;

  unsigned int fw_bw_min       = 1;
  unsigned int fw_bw_max       = 1;
  unsigned int wheel_min       = 1;
  unsigned int wheel_max       = 9;
  unsigned int ring_min[9];
  unsigned int ring_max[9];
  
  if(structure_==FullDetector) {
    fw_bw_max = 3;
    // TEC WHEEL 1
    ring_min[1] = 1;
    ring_max[1] = 3;
    // TEC WHEEL 2
    ring_min[2] = 1;
    ring_max[2] = 3;
    // TEC WHEEL 3
    ring_min[3] = 1;
    ring_max[3] = 3;
    // TEC WHEEL 4
    ring_min[4] = 2;
    ring_max[4] = 3;
    // TEC WHEEL 5
    ring_min[5] = 2;
    ring_max[5] = 3;
    // TEC WHEEL 6
    ring_min[6] = 2;
    ring_max[6] = 3;   
    // TEC WHEEL 7
    ring_min[7] = 1;
    ring_max[7] = 1;
    // TEC WHEEL 8
    ring_min[8] = 1;
    ring_max[8] = 1;
  } else if (structure_==TIFTOBTEC || structure_==TIF) {
    fw_bw_min = 2;
    fw_bw_max = 3;
    // TEC WHEEL 1
    ring_min[1] = 1;
    ring_max[1] = 4;
    // TEC WHEEL 2
    ring_min[2] = 1;
    ring_max[2] = 4;
    // TEC WHEEL 3
    ring_min[3] = 1;
    ring_max[3] = 4;
    // TEC WHEEL 4
    ring_min[4] = 2;
    ring_max[4] = 4;
    // TEC WHEEL 5
    ring_min[5] = 2;
    ring_max[5] = 4;
    // TEC WHEEL 6
    ring_min[6] = 2;
    ring_max[6] = 4;
    // TEC WHEEL 7
    ring_min[7] = 3;
    ring_max[7] = 4;
    // TEC WHEEL 8
    ring_min[8] = 3;
    ring_max[8] = 4;
  }

  for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = ring_min[wheel]; ring < ring_max[wheel]; ++ring ) {
	const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,ring);
	innerSeedRings_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TEC inner seed ring with index: " << temp_ring->getindex(); 
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TEC inner seed rings"; 

}

void RoadMaker::collectInnerTOBSeedRings() {

  // TOB
  unsigned int counter = 0, layer_min=0, layer_max=0, rod_fw_bw_min=0, rod_fw_bw_max=0, detector_min=0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min       = 5;
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==MTCC) { 
    layer_min       = 1; 
    layer_max       = 3;
    rod_fw_bw_min   = 2;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOB || structure_==TIFTIBTOB || structure_==TIF) { 
    layer_min       = 1; 
    layer_max       = 3;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOBTEC) { 
    layer_min       = 1; 
    layer_max       = 3;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  }

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = rod_fw_bw_min; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	const Ring* temp_ring = rings_->getTOBRing(layer,rod_fw_bw,detector);
	innerSeedRings_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
      }    
    }    
  }
  
  if(structure_==FullDetector) { 
    // add most outer rings
    const Ring* temp_ring = rings_->getTOBRing(2,1,6);
    innerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(2,2,6);
    innerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,1,6);
    innerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,2,6);
    innerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,1,6);
    innerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,2,6);
    innerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 

  }

  LogDebug("RoadSearch") << "collected " << counter << " TOB inner seed rings"; 

}

void RoadMaker::collectInnerSeedRings1() {

  if(structure_==FullDetector) {
    collectInnerTIBSeedRings1();
    collectInnerTIDSeedRings1();
    collectInnerTECSeedRings1(); 
  } else if(structure_==MTCC) {
    collectInnerTIBSeedRings1();
  } else if(structure_==TIF) {
    collectInnerTIBSeedRings1();
    collectInnerTOBSeedRings1();
    collectInnerTIDSeedRings1();
    collectInnerTECSeedRings1(); 
 } else if(structure_==TIFTOB) {
    collectInnerTOBSeedRings1();
  } else if(structure_==TIFTIB) {
    collectInnerTIBSeedRings1();
    collectInnerTIDSeedRings1();
  } else if(structure_==TIFTIBTOB) {
    collectInnerTIBSeedRings1();
    collectInnerTIDSeedRings1();
    collectInnerTOBSeedRings1();
  } else if(structure_==TIFTOBTEC) {
    collectInnerTOBSeedRings1();
    collectInnerTECSeedRings1();
   }

  LogDebug("RoadSearch") << "collected " << innerSeedRings1_.size() << " inner seed rings"; 
  
}

void RoadMaker::collectInnerTIBSeedRings1() {

  // TIB
  unsigned int counter = 0, layer_min=0, layer_max=0, fw_bw_min=0, fw_bw_max=0, ext_int_min=0, ext_int_max=0, detector_min=0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min     = 1;
    layer_max     = 2;
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4;
  } else if(structure_==MTCC) {
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 2;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if (structure_==TIF) { 
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if (structure_==TIFTIB) { 
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if (structure_==TIFTIBTOB) { 
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  }

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = ext_int_min; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	  const Ring* temp_ring = rings_->getTIBRing(layer,fw_bw,ext_int,detector);
	  innerSeedRings1_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TIB inner seed ring with index: " << temp_ring->getindex(); 
	}    
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TIB inner seed rings"; 

}

void RoadMaker::collectInnerTIDSeedRings1() {

  // TID
  unsigned int counter = 0;

  unsigned int fw_bw_min = 0, fw_bw_max = 0, wheel_min=0, wheel_max = 0, ring_min=0, ring_max = 0;

  if(structure_==FullDetector) {
    fw_bw_min       = 1;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 2;
  } else if(structure_==TIF) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 2;
  } else if(structure_==TIFTIBTOB) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 2;
  } else if(structure_==TIFTIB) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 1;
    ring_max        = 2;
  }

  for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = ring_min; ring < ring_max; ++ring ) {
	const Ring* temp_ring = rings_->getTIDRing(fw_bw,wheel,ring);
	innerSeedRings1_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TID inner seed ring with index: " << temp_ring->getindex(); 
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TID inner seed rings"; 

}

void RoadMaker::collectInnerTECSeedRings1() {

  // TEC
  unsigned int counter = 0;

  unsigned int fw_bw_min       = 1;
  unsigned int fw_bw_max       = 1;
  unsigned int wheel_min       = 1;
  unsigned int wheel_max       = 4;
  unsigned int ring_min[9];
  unsigned int ring_max[9];

  if(structure_==FullDetector) {
    fw_bw_min = 1;
    fw_bw_max = 3;
    // TEC WHEEL 1
    ring_min[1] = 1;
    ring_max[1] = 2;
    // TEC WHEEL 2
    ring_min[2] = 1;
    ring_max[2] = 2;
    // TEC WHEEL 3
    ring_min[3] = 1;
    ring_max[3] = 2;
    // TEC WHEEL 4
    ring_min[4] = 1;
    ring_max[4] = 1;
    // TEC WHEEL 5
    ring_min[5] = 1;
    ring_max[5] = 1;
    // TEC WHEEL 6
    ring_min[6] = 1;
    ring_max[6] = 1;
    // TEC WHEEL 7
    ring_min[7] = 1;
    ring_max[7] = 1;
    // TEC WHEEL 8
    ring_min[8] = 1;
    ring_max[8] = 1;
  } else if (structure_==TIFTOBTEC || structure_==TIF) {  
    fw_bw_min = 2;
    fw_bw_max = 3;
    // TEC WHEEL 1
    ring_min[1] = 1;
    ring_max[1] = 2;
    // TEC WHEEL 2
    ring_min[2] = 1;
    ring_max[2] = 2;
    // TEC WHEEL 3
    ring_min[3] = 1;
    ring_max[3] = 2;
    // TEC WHEEL 4
    ring_min[4] = 1;
    ring_max[4] = 1;
    // TEC WHEEL 5
    ring_min[5] = 1;
    ring_max[5] = 1;
    // TEC WHEEL 6
    ring_min[6] = 1;
    ring_max[6] = 1;
    // TEC WHEEL 7
    ring_min[7] = 1;
    ring_max[7] = 1;
    // TEC WHEEL 8
    ring_min[8] = 1;
    ring_max[8] = 1;
  }

  for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = ring_min[wheel]; ring < ring_max[wheel]; ++ring ) {
	const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,ring);
	innerSeedRings1_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TEC inner seed ring with index: " << temp_ring->getindex(); 
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TEC inner seed rings"; 

}

void RoadMaker::collectInnerTOBSeedRings1() {

  // TOB
  unsigned int counter = 0, layer_min=0, layer_max=0, rod_fw_bw_min=0, rod_fw_bw_max=0, detector_min=0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min       = 5;
    layer_max       = 6;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==MTCC) { 
    layer_min       = 1; 
    layer_max       = 3;
    rod_fw_bw_min   = 2;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOB || structure_==TIFTIBTOB || structure_==TIF) { 
    layer_min       = 1; 
    layer_max       = 2;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOBTEC) { 
    layer_min       = 1; 
    layer_max       = 2;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  }

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = rod_fw_bw_min; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	const Ring* temp_ring = rings_->getTOBRing(layer,rod_fw_bw,detector);
	innerSeedRings1_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
      }    
    }    
  }
  
  if(structure_==FullDetector) { 
    // add most outer rings
    const Ring* temp_ring = rings_->getTOBRing(2,1,6);
    innerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(2,2,6);
    innerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,1,6);
    innerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,2,6);
    innerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,1,6);
    innerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,2,6);
    innerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 

  }

  LogDebug("RoadSearch") << "collected " << counter << " TOB inner seed rings"; 

}

void RoadMaker::collectInnerSeedRings2() {

  if(structure_==FullDetector) {
    collectInnerTIBSeedRings2();
    collectInnerTIDSeedRings2();
    collectInnerTECSeedRings2(); 
  } else if(structure_==MTCC) {
    collectInnerTIBSeedRings2();
  } else if(structure_==TIF) {
    collectInnerTIBSeedRings2();
    collectInnerTOBSeedRings2();
    collectInnerTIDSeedRings2();
    collectInnerTECSeedRings2(); 
  } else if(structure_==TIFTOB) {
    collectInnerTOBSeedRings2();
  } else if(structure_==TIFTIB) {
    collectInnerTIBSeedRings2();
    collectInnerTIDSeedRings2();
  } else if(structure_==TIFTIBTOB) {
    collectInnerTIBSeedRings2();
    collectInnerTIDSeedRings2();
    collectInnerTOBSeedRings2();
  } else if(structure_==TIFTOBTEC) {
    collectInnerTOBSeedRings2();
    collectInnerTECSeedRings2();
  }


  LogDebug("RoadSearch") << "collected " << innerSeedRings2_.size() << " inner seed rings"; 
  
}

void RoadMaker::collectInnerTIBSeedRings2() {

  // TIB
  unsigned int counter = 0, layer_min = 0, layer_max=0, fw_bw_min=0, fw_bw_max=0, ext_int_min=0, ext_int_max=0, detector_min=0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min     = 2;
    layer_max     = 3;
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4;
  } else if(structure_==MTCC) {
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 2;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if(structure_==TIF) {
    layer_min     = 2;
    layer_max     = 3; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if(structure_==TIFTIB) {
    layer_min     = 1;
    layer_max     = 2; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if(structure_==TIFTIBTOB) {
    layer_min     = 2;
    layer_max     = 3; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  } else if(structure_==TIFTOBTEC) {
    layer_min     = 2;
    layer_max     = 3; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1;
    detector_max  = 4; 
  }

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = ext_int_min; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	  const Ring* temp_ring = rings_->getTIBRing(layer,fw_bw,ext_int,detector);
	  innerSeedRings2_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TIB inner seed ring with index: " << temp_ring->getindex(); 
	}    
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TIB inner seed rings"; 

}

void RoadMaker::collectInnerTIDSeedRings2() {

  // TID
  unsigned int counter = 0;

  unsigned int fw_bw_min = 0, fw_bw_max = 0, wheel_min=0, wheel_max = 0, ring_min = 0, ring_max = 0;

  if(structure_==FullDetector) {
    fw_bw_min       = 1;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 2;
    ring_max        = 3;
  } else if(structure_==TIFTIBTOB || structure_==TIF) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 2;
    ring_max        = 3;
  } else if(structure_==TIFTIB) {
    fw_bw_min       = 2;
    fw_bw_max       = 3;
    wheel_min       = 1;
    wheel_max       = 4;
    ring_min        = 2;
    ring_max        = 2;
  }

  for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = ring_min; ring < ring_max; ++ring ) {
	const Ring* temp_ring = rings_->getTIDRing(fw_bw,wheel,ring);
	innerSeedRings2_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TID inner seed ring with index: " << temp_ring->getindex(); 
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TID inner seed rings"; 

}

void RoadMaker::collectInnerTECSeedRings2() {

  // TEC
  unsigned int counter = 0;

  unsigned int fw_bw_min       = 1;
  unsigned int fw_bw_max       = 1;
  unsigned int wheel_min       = 1;
  unsigned int wheel_max       = 9;
  unsigned int ring_min[9];
  unsigned int ring_max[9];

  if(structure_==FullDetector) {
    fw_bw_min = 1;
    fw_bw_max = 3;
    // TEC WHEEL 1
    ring_min[1] = 2;
    ring_max[1] = 3;
    // TEC WHEEL 2
    ring_min[2] = 2;
    ring_max[2] = 3;
    // TEC WHEEL 3
    ring_min[3] = 2;
    ring_max[3] = 3;
    // TEC WHEEL 4
    ring_min[4] = 2;
    ring_max[4] = 3;
    // TEC WHEEL 5
    ring_min[5] = 2;
    ring_max[5] = 3;
    // TEC WHEEL 6
    ring_min[6] = 2;
    ring_max[6] = 3;
    // TEC WHEEL 7
    ring_min[7] = 1;
    ring_max[7] = 1;
    // TEC WHEEL 8
    ring_min[8] = 1;
    ring_max[8] = 1;
  } else if (structure_==TIFTOBTEC || structure_==TIF) {  
    fw_bw_min = 2;
    fw_bw_max = 3;
    // TEC WHEEL 1
    ring_min[1] = 3;
    ring_max[1] = 4;
    // TEC WHEEL 2
    ring_min[2] = 3;
    ring_max[2] = 4;
    // TEC WHEEL 3
    ring_min[3] = 3;
    ring_max[3] = 4;
    // TEC WHEEL 4
    ring_min[4] = 3;
    ring_max[4] = 4;
    // TEC WHEEL 5
    ring_min[5] = 3;
    ring_max[5] = 4;
    // TEC WHEEL 6
    ring_min[6] = 3;
    ring_max[6] = 4; 
    // TEC WHEEL 6
    ring_min[7] = 3;
    ring_max[7] = 4;
    // TEC WHEEL 7
    ring_min[8] = 3;
    ring_max[8] = 4;
  }

  for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = ring_min[wheel]; ring < ring_max[wheel]; ++ring ) {
	const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,ring);
	innerSeedRings2_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TEC inner seed ring with index: " << temp_ring->getindex(); 
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TEC inner seed rings"; 

}

void RoadMaker::collectInnerTOBSeedRings2() {

  // TOB
  unsigned int counter = 0, layer_min=0, layer_max=0, rod_fw_bw_min=0, rod_fw_bw_max=0, detector_min=0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min       = 5;
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_max    = 1;
    detector_max    = 7;
  } else if (structure_==MTCC) { 
    layer_min       = 1; 
    layer_max       = 3;
    rod_fw_bw_min   = 2;
    rod_fw_bw_max   = 3;
    detector_max    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOB || structure_==TIFTIBTOB || structure_==TIFTOBTEC || structure_==TIF) { 
    layer_min       = 2; 
    layer_max       = 3;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_max    = 1;
    detector_max    = 7;
  }

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = rod_fw_bw_min; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	const Ring* temp_ring = rings_->getTOBRing(layer,rod_fw_bw,detector);
	innerSeedRings2_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
      }    
    }    
  }
  
  if(structure_==FullDetector) { 
    // add most outer rings
    const Ring* temp_ring = rings_->getTOBRing(2,1,6);
    innerSeedRings2_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(2,2,6);
    innerSeedRings2_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,1,6);
    innerSeedRings2_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,2,6);
    innerSeedRings2_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,1,6);
    innerSeedRings2_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,2,6);
    innerSeedRings2_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 

  }

  LogDebug("RoadSearch") << "collected " << counter << " TOB inner seed rings"; 

}

void RoadMaker::collectOuterSeedRings() {
  
  if(structure_==FullDetector) {
    collectOuterTOBSeedRings();
    collectOuterTECSeedRings();
  } else if (structure_==MTCC) {
    collectOuterTOBSeedRings();
  } else if (structure_==TIF) {
    collectOuterTOBSeedRings();
    collectOuterTECSeedRings();
  } else if (structure_==TIFTOB) {
    collectOuterTOBSeedRings();
  } else if (structure_==TIFTIB) {
    collectOuterTIBSeedRings();
  } else if (structure_==TIFTIBTOB) {
    collectOuterTOBSeedRings();
  } else if (structure_==TIFTOBTEC) {
    collectOuterTOBSeedRings();
    collectOuterTECSeedRings();
  }



  LogDebug("RoadSearch") << "collected " << outerSeedRings_.size() << " outer seed rings"; 
}

void RoadMaker::collectOuterTIBSeedRings() {

  // TIB
  unsigned int counter = 0, layer_min=0, layer_max=0, fw_bw_min=0, fw_bw_max=0, ext_int_min=0, ext_int_max=0, detector_min=0, detector_max=0;
  if(structure_==TIFTIB) {
    layer_min     = 3; 
    layer_max     = 5; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1; 
    detector_max  = 4; 
  } 

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = ext_int_min; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	  const Ring* temp_ring = rings_->getTIBRing(layer,fw_bw,ext_int,detector);
	  outerSeedRings_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TIB outer seed ring with index: " << temp_ring->getindex(); 
	}    
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TIB outer seed rings"; 

}

void RoadMaker::collectOuterTOBSeedRings() {

  // TOB
  unsigned int counter = 0, layer_min=0, layer_max=0, rod_fw_bw_min=0, rod_fw_bw_max=0, detector_min=0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min       = 5;
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==MTCC) { 
    layer_min       = 1; 
    layer_max       = 3;
    rod_fw_bw_min   = 2;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIF) { 
    layer_min       = 5; 
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOB) { 
    layer_min       = 5; 
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTIBTOB) { 
    layer_min       = 5; 
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOBTEC) { 
    layer_min       = 5; 
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  }

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = rod_fw_bw_min; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	const Ring* temp_ring = rings_->getTOBRing(layer,rod_fw_bw,detector);
	outerSeedRings_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
      }    
    }    
  }
  
  if(structure_==FullDetector) { 
    // add most outer rings
    const Ring* temp_ring = rings_->getTOBRing(1,1,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(1,2,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(2,1,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(2,2,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,1,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,2,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,1,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,2,6);
    outerSeedRings_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 

  }

  LogDebug("RoadSearch") << "collected " << counter << " TOB outer seed rings"; 

}

void RoadMaker::collectOuterTECSeedRings() {

  // TEC
  unsigned int counter = 0;

  if(structure_==FullDetector) {
    // outer and second outer ring in all wheels except those treated in the following
    unsigned int fw_bw_min       = 1;
    unsigned int fw_bw_max       = 3;
    unsigned int wheel_min       = 1;
    unsigned int wheel_max       = 8;
    
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
	const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,6);
	outerSeedRings_.push_back(temp_ring);
	++counter;
	temp_ring = rings_->getTECRing(fw_bw,wheel,7);
	outerSeedRings_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
      }
    }
    
    // add last two wheels
    fw_bw_min       = 1;
    fw_bw_max       = 3;
    wheel_min       = 1;
    unsigned int wheel_start     = 8;
    wheel_max       = 10;
    unsigned int second_ring_min[10];
    unsigned int second_ring_max[10];
    
    // TEC WHEEL 1
    second_ring_min[1] = 1;
    second_ring_max[1] = 8;
    // TEC WHEEL 2
    second_ring_min[2] = 1;
    second_ring_max[2] = 8;
    // TEC WHEEL 3
    second_ring_min[3] = 1;
    second_ring_max[3] = 8;
    // TEC WHEEL 4
    second_ring_min[4] = 2;
    second_ring_max[4] = 8;
    // TEC WHEEL 5
    second_ring_min[5] = 2;
    second_ring_max[5] = 8;
    // TEC WHEEL 6
    second_ring_min[6] = 2;
    second_ring_max[6] = 8;
    // TEC WHEEL 7
    second_ring_min[7] = 3;
    second_ring_max[7] = 8;
    // TEC WHEEL 8
    second_ring_min[8] = 3;
    second_ring_max[8] = 8;
    // TEC WHEEL 9
    second_ring_min[9] = 4;
    second_ring_max[9] = 8;
    
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int wheel = wheel_start; wheel < wheel_max; ++wheel ) {
	for ( unsigned int second_ring = second_ring_min[wheel]; second_ring < second_ring_max[wheel]; ++second_ring ) {
	  const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,second_ring);
	  outerSeedRings_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
	}    
      }
    }
  } else if (structure_==TIFTOBTEC || structure_==TIF) {
    unsigned int fw_bw_min = 2;
    unsigned int fw_bw_max = 3;
    unsigned int wheel_min = 1;
    unsigned int wheel_max = 10;
    unsigned int ring_min  = 6;
    unsigned int ring_max  = 8;

    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
	for ( unsigned int ring = ring_min; ring < ring_max; ++ring ) {
	  const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,ring);
	  outerSeedRings_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
	}
      }
    }
    
  }

  LogDebug("RoadSearch") << "collected " << counter << " TEC outer seed rings"; 

}

void RoadMaker::collectOuterSeedRings1() {
  
  if(structure_==FullDetector) {
    collectOuterTOBSeedRings1();
    collectOuterTECSeedRings1();
  } else if (structure_==MTCC) {
    collectOuterTOBSeedRings1();
  } else if (structure_==TIF) {
    collectOuterTOBSeedRings1();
    collectOuterTECSeedRings1();
  } else if (structure_==TIFTOB) {
    collectOuterTOBSeedRings1();
  } else if (structure_==TIFTIB) {
    collectOuterTIBSeedRings1();
  } else if (structure_==TIFTIBTOB) {
    collectOuterTOBSeedRings1();
  } else if (structure_==TIFTOBTEC) {
    collectOuterTOBSeedRings1();
    collectOuterTECSeedRings1();

  }

  LogDebug("RoadSearch") << "collected " << outerSeedRings1_.size() << " outer seed rings"; 
}

void RoadMaker::collectOuterTIBSeedRings1() {

  // TIB
  unsigned int counter = 0, layer_min=0, layer_max=0, fw_bw_min=0, fw_bw_max=0, ext_int_min=0, ext_int_max=0, detector_min=0, detector_max=0;
  if(structure_==TIFTIB) {
    layer_min     = 4; 
    layer_max     = 5; 
    fw_bw_min     = 1;
    fw_bw_max     = 3;
    ext_int_min   = 1;
    ext_int_max   = 3;
    detector_min  = 1; 
    detector_max  = 4; 
  } 

  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = ext_int_min; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	  const Ring* temp_ring = rings_->getTIBRing(layer,fw_bw,ext_int,detector);
	  outerSeedRings1_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TIB outer seed ring with index: " << temp_ring->getindex(); 
	}    
      }    
    }
  }

  LogDebug("RoadSearch") << "collected " << counter << " TIB outer seed rings"; 

}

void RoadMaker::collectOuterTOBSeedRings1() {

  // TOB
  unsigned int counter = 0, layer_min=0, layer_max=0, rod_fw_bw_min=0, rod_fw_bw_max=0, detector_min=0, detector_max=0;
  if(structure_==FullDetector) {
    layer_min       = 6;
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==MTCC) { 
    layer_min       = 1; 
    layer_max       = 3;
    rod_fw_bw_min   = 2;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIF) { 
    layer_min       = 6;
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
   } else if (structure_==TIFTOB) { 
    layer_min       = 5;
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTIBTOB) { 
    layer_min       = 6;
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  } else if (structure_==TIFTOBTEC) { 
    layer_min       = 6; 
    layer_max       = 7;
    rod_fw_bw_min   = 1;
    rod_fw_bw_max   = 3;
    detector_min    = 1;
    detector_max    = 7;
  }


  for ( unsigned int layer = layer_min; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = rod_fw_bw_min; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = detector_min; detector < detector_max; ++detector ) {
	const Ring* temp_ring = rings_->getTOBRing(layer,rod_fw_bw,detector);
	outerSeedRings1_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
      }    
    }    
  }
  
  if(structure_==FullDetector) { 
    // add most outer rings
    const Ring* temp_ring = rings_->getTOBRing(1,1,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(1,2,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(2,1,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(2,2,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,1,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(3,2,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,1,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
    
    temp_ring = rings_->getTOBRing(4,2,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 

    temp_ring = rings_->getTOBRing(5,1,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
      
    temp_ring = rings_->getTOBRing(5,2,6);
    outerSeedRings1_.push_back(temp_ring);
    ++counter;
    LogDebug("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 

  }

  LogDebug("RoadSearch") << "collected " << counter << " TOB outer seed rings"; 

}

void RoadMaker::collectOuterTECSeedRings1() {

  // TEC
  unsigned int counter = 0;

  if(structure_==FullDetector) {
    // outer ring in all wheels except those treated in the following
    unsigned int fw_bw_min       = 1;
    unsigned int fw_bw_max       = 3;
    unsigned int wheel_min       = 1;
    unsigned int wheel_max       = 9;
    
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
	const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,7);
	outerSeedRings1_.push_back(temp_ring);
	++counter;
	LogDebug("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
      }
    }
    
    // add last two wheels
    fw_bw_min       = 1;
    fw_bw_max       = 3;
    wheel_min       = 1;
    unsigned int wheel_start     = 9;
    wheel_max       = 10;
    unsigned int second_ring_min[10];
    unsigned int second_ring_max[10];
    
    // TEC WHEEL 1
    second_ring_min[1] = 1;
    second_ring_max[1] = 8;
    // TEC WHEEL 2
    second_ring_min[2] = 1;
    second_ring_max[2] = 8;
    // TEC WHEEL 3
    second_ring_min[3] = 1;
    second_ring_max[3] = 8;
    // TEC WHEEL 4
    second_ring_min[4] = 2;
    second_ring_max[4] = 8;
    // TEC WHEEL 5
    second_ring_min[5] = 2;
    second_ring_max[5] = 8;
    // TEC WHEEL 6
    second_ring_min[6] = 2;
    second_ring_max[6] = 8;
    // TEC WHEEL 7
    second_ring_min[7] = 3;
    second_ring_max[7] = 8;
    // TEC WHEEL 8
    second_ring_min[8] = 3;
    second_ring_max[8] = 8;
    // TEC WHEEL 9
    second_ring_min[9] = 4;
    second_ring_max[9] = 8;
    
    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int wheel = wheel_start; wheel < wheel_max; ++wheel ) {
	for ( unsigned int second_ring = second_ring_min[wheel]; second_ring < second_ring_max[wheel]; ++second_ring ) {
	  const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,second_ring);
	  outerSeedRings1_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
	}    
      }
    }
  } else if (structure_==TIFTOBTEC || structure_==TIF) {
    unsigned int fw_bw_min = 2;
    unsigned int fw_bw_max = 3;
    unsigned int wheel_min = 1;
    unsigned int wheel_max = 10;
    unsigned int ring_min  = 7;
    unsigned int ring_max  = 8;

    for ( unsigned int fw_bw = fw_bw_min; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int wheel = wheel_min; wheel < wheel_max; ++wheel ) {
	for ( unsigned int ring = ring_min; ring < ring_max; ++ring ) {
	  const Ring* temp_ring = rings_->getTECRing(fw_bw,wheel,ring);
	  outerSeedRings1_.push_back(temp_ring);
	  ++counter;
	  LogDebug("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
	}
      }
    }
    
  }
  LogDebug("RoadSearch") << "collected " << counter << " TEC outer seed rings"; 

}

bool RoadMaker::RingsOnSameLayer(const Ring *ring1, const Ring* ring2) {
  //
  // check whether two input rings are on the same layer
  //

  // return value
  bool result = false;
  
  // get first DetId of ring
  const DetId ring1DetId = ring1->getFirst();
  const DetId ring2DetId = ring2->getFirst();

  // check if both rings belong to same subdetector
  if ( (unsigned int)ring1DetId.subdetId() == StripSubdetector::TIB && 
       (unsigned int)ring2DetId.subdetId() == StripSubdetector::TIB ) {
    // make TIBDetId instance
    TIBDetId ring1DetIdTIB(ring1DetId.rawId());
    TIBDetId ring2DetIdTIB(ring2DetId.rawId());
    // check whether both rings are on the same TIB layer
    if ( ring1DetIdTIB.layer() == ring2DetIdTIB.layer() ) {
      result = true;
    }
  } else if ( (unsigned int)ring1DetId.subdetId() == StripSubdetector::TOB &&
	      (unsigned int)ring2DetId.subdetId() == StripSubdetector::TOB ) {
    // make TOBDetId instance
    TOBDetId ring1DetIdTOB(ring1DetId.rawId());
    TOBDetId ring2DetIdTOB(ring2DetId.rawId());
    // check whether both rings are on the same TOB layer
    if ( ring1DetIdTOB.layer() == ring2DetIdTOB.layer() ) {
      result = true;
    }
  } else if ( (unsigned int)ring1DetId.subdetId() == StripSubdetector::TID && 
	      (unsigned int)ring2DetId.subdetId() == StripSubdetector::TID) {
    // make TIDDetId instance
    TIDDetId ring1DetIdTID(ring1DetId.rawId());
    TIDDetId ring2DetIdTID(ring2DetId.rawId());
    // check whether both rings are on the same TID wheel
    if ( ring1DetIdTID.wheel() == ring2DetIdTID.wheel() ) {
      result = true;
    }
  } else if ( (unsigned int)ring1DetId.subdetId() == StripSubdetector::TEC &&
	      (unsigned int)ring2DetId.subdetId() == StripSubdetector::TEC ) {
    // make TECDetId instance
    TECDetId ring1DetIdTEC(ring1DetId.rawId());
    TECDetId ring2DetIdTEC(ring2DetId.rawId());
    // check whether both rings are on the same TEC wheel
    if ( ring1DetIdTEC.wheel() == ring2DetIdTEC.wheel() ) {
      result = true;
    }
  } else if ( (unsigned int)ring1DetId.subdetId() == PixelSubdetector::PixelBarrel && 
	      (unsigned int)ring2DetId.subdetId() == PixelSubdetector::PixelBarrel) {
    // make PXBDetId instance
    PXBDetId ring1DetIdPXB(ring1DetId.rawId());
    PXBDetId ring2DetIdPXB(ring2DetId.rawId());
    // check whether both rings are on the same PXB layer
    if ( ring1DetIdPXB.layer() == ring2DetIdPXB.layer() ) {
      result = true;
    }
  } else if ( (unsigned int)ring1DetId.subdetId() == PixelSubdetector::PixelEndcap &&
	      (unsigned int)ring2DetId.subdetId() == PixelSubdetector::PixelEndcap) {
    // make PXFDetId instance
    PXFDetId ring1DetIdPXF(ring1DetId.rawId());
    PXFDetId ring2DetIdPXF(ring2DetId.rawId());
    // check whether both rings are on the same PXF disk
    if ( ring1DetIdPXF.disk() == ring2DetIdPXF.disk() ) {
      result = true;
    }
  }
  
  return result;
}

bool RoadMaker::RingInBarrel(const Ring *ring) {
  //
  // check if the ring is a TIB or TOB ring
  //

  // return value
  bool result = false;
  
  // get first DetId of ring
  const DetId ringDetId = ring->getFirst();

  // check if both rings belong to same subdetector
  if ( (unsigned int)ringDetId.subdetId() == StripSubdetector::TIB ) {
    result = true;
  } else if ( (unsigned int)ringDetId.subdetId() == StripSubdetector::TOB ) {
    result = true;
  }
  
  return result;
}

bool RoadMaker::RingsOnSameLayer(std::pair<const Ring*,const Ring*> seed1, std::pair<const Ring*,const Ring*> seed2) {
  //
  // check whether two input seeds (two ring seeds) are on the same layer
  //
  
  // return value
  bool result = false;
  
  result = RingsOnSameLayer(seed1.first,seed2.first) &&
    RingsOnSameLayer(seed1.second,seed2.second);
  
  return result;
}
 
std::vector<std::pair<double,double> > RoadMaker::LinesThroughRingAndBS(const Ring* ring ) {
  //
  // calculate lines through all 4 corners of the ring and the BS
  //
   
  // return value
  std::vector<std::pair<double,double> > result;
   
  double z = 0;
  double r = 0;;
  
  for (int ic1 = 0; ic1<4; ++ic1) {
    switch (ic1) {
    case 0: z = ring->getzmin(); r = ring->getrmin(); break;
    case 1: z = ring->getzmin(); r = ring->getrmax(); break;
    case 2: z = ring->getzmax(); r = ring->getrmax(); break;
    case 3: z = ring->getzmax(); r = ring->getrmin(); break;
    }
    for (int ib = 0; ib<2; ++ib) {
      double zb = zBS_*(2*ib-1);
      result.push_back(std::pair<double,double>((z-zb)/r,zb));
    }
  }

  return result;
}
 
std::vector<std::pair<double,double> > RoadMaker::LinesThroughRings(const Ring *ring1,
								    const Ring *ring2) {
  //
  // calculate lines through all 4 corners of the rings
  //

  // return value
  std::vector<std::pair<double,double> > result;

  double z1 = 0;
  double r1 = 0;;
  double z2 = 0;
  double r2 = 0;

  for (int ic1 = 0; ic1<4; ++ic1) {
    switch (ic1) {
    case 0: z1 = ring1->getzmin(); r1 = ring1->getrmin(); break;
    case 1: z1 = ring1->getzmin(); r1 = ring1->getrmax(); break;
    case 2: z1 = ring1->getzmax(); r1 = ring1->getrmax(); break;
    case 3: z1 = ring1->getzmax(); r1 = ring1->getrmin(); break;
    }
    for (int ic2 = 0; ic2<4; ++ic2) {
      switch (ic2) {
      case 0: z2 = ring2->getzmin(); r2 = ring2->getrmin(); break;
      case 1: z2 = ring2->getzmin(); r2 = ring2->getrmax(); break;
      case 2: z2 = ring2->getzmax(); r2 = ring2->getrmax(); break;
      case 3: z2 = ring2->getzmax(); r2 = ring2->getrmin(); break;
      }
      result.push_back(std::pair<double,double>((z2 - z1)/(r2 - r1),(r2*z1 - r1*z2)/(r2 - r1)));
    }
  }
  
  return result;
}

bool RoadMaker::CompatibleWithLines(std::vector<std::pair<double,double> > lines,
				    const Ring* ring) {
  //
  // check compatibility of ring z extension with extrapolation of lines to ring radius
  //

  // return value
  bool result = true;

  // calculate zmin, zmax at the radius of the ring
  double zmin = 999;
  double zmax = -zmin;
  for (int m=0; m<2; ++m) {
    double r = m==0 ? ring->getrmin() : ring->getrmax();
    for (std::vector<std::pair<double,double> >::iterator line = lines.begin();
	 line != lines.end();
	 ++line ) {
      double z = line->first*r + line->second;
      if (zmin>z) zmin = z;
      if (zmax<z) zmax = z;
    }
  }
  if (ring->getzmax()<zmin || ring->getzmin()>zmax) {
    result = false;
  }

  return result;
}

Roads::RoadSet RoadMaker::RingsCompatibleWithSeed(Roads::RoadSeed seed) {
  //
  // collect all rings which are compatible with the seed
  //
  
  // return value
  std::vector<const Ring*> tempRings;

  // calculate lines
  std::vector<std::vector<std::pair<double,double> > > lines;

  for ( std::vector<const Ring*>::iterator innerRing = seed.first.begin();
	innerRing != seed.first.end();
	++innerRing) {
    // calculate lines between inner seed rings and BS
    lines.push_back(LinesThroughRingAndBS(*innerRing));

    for ( std::vector<const Ring*>::iterator outerRing = seed.second.begin();
	  outerRing != seed.second.end();
	  ++outerRing) {
      // calculate lines between inner and outer seed rings
      lines.push_back(LinesThroughRings((*innerRing),(*outerRing)));
    }
  }

  for ( std::vector<const Ring*>::iterator outerRing = seed.second.begin();
	outerRing != seed.second.end();
	++outerRing) {
    // calculate lines between outer seed rings and BS
    lines.push_back(LinesThroughRingAndBS(*outerRing));
  }

  for ( Rings::const_iterator ring = rings_->begin();
	ring != rings_->end();
	++ring ) {
    bool compatible = true;
    for ( std::vector<std::vector<std::pair<double,double> > >::iterator line = lines.begin();
	  line != lines.end();
	  ++line ) {
      if ( !CompatibleWithLines(*line, &(ring->second))) {
	compatible = false;
      }
    }
    if ( compatible ) {
      tempRings.push_back(&(ring->second));
    }
  }

  

  return SortRingsIntoLayers(tempRings);
}


Roads::RoadSeed RoadMaker::CloneSeed(Roads::RoadSeed seed) {
  //
  // clone seed
  //

  Roads::RoadSeed result;

  for ( std::vector<const Ring*>::iterator ring = seed.first.begin();
	ring != seed.first.end();
	++ring ) {
    result.first.push_back((*ring));
  }

  for ( std::vector<const Ring*>::iterator ring = seed.second.begin();
	ring != seed.second.end();
	++ring ) {
    result.second.push_back((*ring));
  }

  return result;
}

bool RoadMaker::AddRoad(Roads::RoadSeed seed,
			Roads::RoadSet set) {
  //
  // add road
  // check if seed rings are included in seeds of other roads or if they are new
  // add road if new
  // take road with larger seed and discard older seed
  // assumption: increasing number of seed rings while adding during the program
  //

  // return value
  bool result = true;

  for ( Roads::iterator existingRoad = roads_->begin();
	existingRoad != roads_->end();
	++existingRoad ) {

    Roads::RoadSeed existingSeed = existingRoad->first;

    // check if existing inner seed is included in new seed
    unsigned int includedInner = 0;
    unsigned int includedOuter = 0;
    for ( std::vector<const Ring*>::iterator existingInnerRing = existingSeed.first.begin();
	  existingInnerRing != existingSeed.first.end();
	  ++existingInnerRing ) {
      bool ringIncluded = false;
      for ( std::vector<const Ring*>::iterator innerRing = seed.first.begin();
	    innerRing != seed.first.end();
	    ++innerRing ) {
	if ( (*existingInnerRing) == (*innerRing) ) {
	  ringIncluded = true;
	}
      }
      if ( ringIncluded ) {
	++includedInner;
      }
    }
    for ( std::vector<const Ring*>::iterator existingOuterRing = existingSeed.second.begin();
	  existingOuterRing != existingSeed.second.end();
	  ++existingOuterRing ) {
      bool ringIncluded = false;
      for ( std::vector<const Ring*>::iterator outerRing = seed.second.begin();
	    outerRing != seed.second.end();
	    ++outerRing ) {
	if ( (*existingOuterRing) == (*outerRing) ) {
	  ringIncluded = true;
	}
      }
      if ( ringIncluded ) {
	++includedOuter;
      }
    }

    if ( includedInner ==  existingSeed.first.size() &&
	 includedOuter ==  existingSeed.second.size() ) {
      // existing road included in new road, remove
      roads_->erase(existingRoad);
    }
  }

  // sort seeds
  std::sort(seed.first.begin(),seed.first.end(),SortRingsByZR());
  std::sort(seed.second.begin(),seed.second.end(),SortRingsByZR());

  roads_->insert(seed,set);
  return result;
}

std::pair<Roads::RoadSeed, Roads::RoadSet> RoadMaker::AddInnerSeedRing(std::pair<Roads::RoadSeed, Roads::RoadSet> input) {
  //
  // add another inner seed ring
  // check for inner seed ring which is in input RoadSet and on another layer than the first seed ring
  for ( std::vector<const Ring*>::iterator innerSeedRing = innerSeedRings_.begin();
	innerSeedRing != innerSeedRings_.end();
	++innerSeedRing ) {
    for ( Roads::RoadSet::iterator roadSetVector = input.second.begin();
	  roadSetVector != input.second.end();
	  ++roadSetVector ) {
      for ( std::vector<const Ring*>::iterator roadSetRing = roadSetVector->begin();
	    roadSetRing != roadSetVector->end();
	    ++roadSetRing) {
	// check for same ring
	if ( (*innerSeedRing) == (*roadSetRing) ) {
	  // check that new ring is not on same layer as previous inner seed rings
	  bool onSameLayer = false;
	  for ( std::vector<const Ring*>::iterator roadSeedRing = input.first.first.begin();
		roadSeedRing != input.first.first.end();
		++roadSeedRing ) {
	    if ( RingsOnSameLayer((*roadSeedRing),(*innerSeedRing)) ) {
	      onSameLayer = true;
	    }
	  }
	  if ( !onSameLayer ) {
		  
	    Roads::RoadSeed seed = CloneSeed(input.first);
	    seed.first.push_back((*innerSeedRing));
		  
	    Roads::RoadSet set = RingsCompatibleWithSeed(seed);
		  
	    if ( SameRoadSet(input.second,set) ) {

	      AddRoad(seed,set);

	      std::pair<Roads::RoadSeed, Roads::RoadSet> result(seed,set);

	      return result;
	    }
	  }
	}
      }
    }
  }

  // if no ring could be added, return input
  return input;

}

std::pair<Roads::RoadSeed, Roads::RoadSet> RoadMaker::AddOuterSeedRing(std::pair<Roads::RoadSeed, Roads::RoadSet> input) {
  //
  // add another outer seed ring
  // check for outer seed ring which is in input RoadSet and on another layer than the first seed ring
  for ( std::vector<const Ring*>::iterator outerSeedRing = outerSeedRings_.begin();
	outerSeedRing != outerSeedRings_.end();
	++outerSeedRing ) {
    for ( Roads::RoadSet::iterator roadSetVector = input.second.begin();
	  roadSetVector != input.second.end();
	  ++roadSetVector ) {
      for ( std::vector<const Ring*>::iterator roadSetRing = roadSetVector->begin();
	    roadSetRing != roadSetVector->end();
	    ++roadSetRing) {
	// check for same ring
	if ( (*outerSeedRing) == (*roadSetRing) ) {
	  // check that new ring is not on same layer as previous outer seed rings
	  bool onSameLayer = false;
	  for ( std::vector<const Ring*>::iterator roadSeedRing = input.first.second.begin();
		roadSeedRing != input.first.second.end();
		++roadSeedRing ) {
	    if ( RingsOnSameLayer((*roadSeedRing),(*outerSeedRing)) ) {
	      onSameLayer = true;
	    }
	  }
	  if ( !onSameLayer ) {
		  
	    Roads::RoadSeed seed = CloneSeed(input.first);
	    seed.second.push_back((*outerSeedRing));
		  
	    Roads::RoadSet set = RingsCompatibleWithSeed(seed);
		  
	    AddRoad(seed,set);

	    std::pair<Roads::RoadSeed, Roads::RoadSet> result(seed,set);

	    return result;
	  }
	}
      }
    }
  }

  // if no ring could be added, return input
  return input;

}

bool RoadMaker::SameRoadSet(Roads::RoadSet set1, Roads::RoadSet set2 ) {
  //
  // check if input roadsets contains exactly the same rings
  //

  // calculate how many rings are in both input sets
  unsigned int nRingsSet1 = 0;
  unsigned int nRingsSet2 = 0;
  for ( Roads::RoadSet::iterator vector1 = set1.begin();
	vector1 != set1.end();
	++vector1 ) {
    nRingsSet1 += vector1->size();
  }

  for ( Roads::RoadSet::iterator vector2 = set2.begin();
	vector2 != set2.end();
	++vector2 ) {
    nRingsSet2 += vector2->size();
  }

  // if one of the input sets has more rings than the other, they cannot be the same
  if ( nRingsSet1 != nRingsSet2 ) {
    return false;
  }

  bool different = false;
  for ( Roads::RoadSet::iterator vector1 = set1.begin();
	vector1 != set1.end();
	++vector1 ) {
    for ( std::vector<const Ring*>::iterator ring1 = vector1->begin();
	  ring1 != vector1->end();
	  ++ring1 ) {
      bool included = false;
      for ( Roads::RoadSet::iterator vector2 = set2.begin();
	    vector2 != set2.end();
	    ++vector2 ) {
	for ( std::vector<const Ring*>::iterator ring2 = vector2->begin();
	      ring2 != vector2->end();
	      ++ring2 ) {
	  if ( (*ring1) == (*ring2) ) {
	    included = true;
	  }
	}
      }
      if ( !included ) {
	different = true;
      }
    }
  }

  return !different;
}

Roads::RoadSet RoadMaker::SortRingsIntoLayers(std::vector<const Ring*> input) {
  //
  // sort rings from input into layer structure of RoadSet
  //

  // return value
  Roads::RoadSet set;

  // sort rings in input by their center in rz, do it twice
  std::sort(input.begin(),input.end(),SortRingsByZR());

  const Ring *reference = (*(input.begin()));
  std::vector<const Ring*> tmp;
  tmp.push_back(reference);
  for (std::vector<const Ring*>::iterator ring = ++input.begin();
       ring != input.end();
       ++ring ) {
    if ( RingsOnSameLayer(reference,(*ring)) ) {
      reference = (*ring);
      tmp.push_back(reference);
    } else {
      set.push_back(tmp);
      tmp.clear();
      reference = (*ring);
      tmp.push_back(reference);
    }
  }

  if ( tmp.size() > 0 ) {
    set.push_back(tmp);
  }

  // order layers in set
  std::sort(set.begin(),set.end(),SortLayersByZR());

//   set.push_back(input);
  return set;
}

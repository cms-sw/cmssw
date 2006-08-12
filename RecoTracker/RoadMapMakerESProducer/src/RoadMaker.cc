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
// $Date: 2006/03/28 22:48:06 $
// $Revision: 1.6 $
//

#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>

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

RoadMaker::RoadMaker(const TrackerGeometry &tracker, unsigned int verbosity) : 
  verbosity_(verbosity) {

    rings_ = new Rings(tracker,verbosity), 

      roads_ = new Roads();

    constructRoads();

  }

RoadMaker::~RoadMaker() { 

}

void RoadMaker::constructRoads() {

  // fill vector of inner Rings
  std::vector<Ring*> innerRings;
  collectInnerSeedRings(innerRings);

  // fill vector of outer Rings
  std::vector<Ring*> outerRings;
  collectOuterSeedRings(outerRings);

  // loop over inner-outer ring pairs
  int npairs = 0;
  int ntriplets = 0;
  
  for ( std::vector<Ring*>::iterator innerRingsIterator = innerRings.begin(); innerRingsIterator != innerRings.end(); ++innerRingsIterator ) {
    Ring* innerRing = *innerRingsIterator;
    // draw lines (z = a*r + b) through all corners and beam spot
    const double sz = 3*5.3; // cm
    double a1[8], b1[8];
    for (int ic1 = 0; ic1<4; ++ic1) {
      double z = 0;
      double r = 0;;
      switch (ic1) {
      case 0: z = innerRing->getzmin(); r = innerRing->getrmin(); break;
      case 1: z = innerRing->getzmin(); r = innerRing->getrmax(); break;
      case 2: z = innerRing->getzmax(); r = innerRing->getrmax(); break;
      case 3: z = innerRing->getzmax(); r = innerRing->getrmin(); break;
      }
      for (int ib = 0; ib<2; ++ib) {
	double zb = sz*(2*ib-1);
	b1[ib*4+ic1] = zb;
	a1[ib*4+ic1] = (z-zb)/r;
      }
    }
    for ( std::vector<Ring*>::iterator outerRingsIterator = outerRings.begin(); outerRingsIterator != outerRings.end(); ++outerRingsIterator ) {
      Ring* outerRing = *outerRingsIterator;
      // calculate zmin, zmax at the radius of the ring
      double zmin = 999;
      double zmax = -zmin;
      for (int m=0; m<2; ++m) {
	double r = m==0 ? outerRing->getrmin() : outerRing->getrmax();
	for (int k=0; k<8; ++k) {
	  double z = a1[k]*r + b1[k];
	  if (zmin>z) zmin = z;
	  if (zmax<z) zmax = z;
	}
      }
      if (outerRing->getzmax()<zmin || outerRing->getzmin()>zmax) continue;
      // good ring pair
      ++npairs;
      Roads::RoadSeed *seed = new Roads::RoadSeed(Ring(innerRing),Ring(outerRing));
      // draw lines (z = a*r + b) through all corners and beam spot
      double a2[8], b2[8];
      for (int ic2 = 0; ic2<4; ++ic2) {
	double z = 0;
	double r = 0;
	switch (ic2) {
	case 0: z = outerRing->getzmin(); r = outerRing->getrmin(); break;
	case 1: z = outerRing->getzmin(); r = outerRing->getrmax(); break;
	case 2: z = outerRing->getzmax(); r = outerRing->getrmax(); break;
	case 3: z = outerRing->getzmax(); r = outerRing->getrmin(); break;
	}
	for (int ib = 0; ib<2; ++ib) {
	  double zb = sz*(2*ib-1);
	  b2[ib*4+ic2] = zb;
	  a2[ib*4+ic2] = (z-zb)/r;
	}
      }
      // draw lines (z = a*r + b) through all corners
      double a3[16], b3[16];
      for (int ic1 = 0; ic1<4; ++ic1) {
	double z1 = 0;
	double r1 = 0;
	switch (ic1) {
	case 0: z1 = innerRing->getzmin(); r1 = innerRing->getrmin(); break;
	case 1: z1 = innerRing->getzmin(); r1 = innerRing->getrmax(); break;
	case 2: z1 = innerRing->getzmax(); r1 = innerRing->getrmax(); break;
	case 3: z1 = innerRing->getzmax(); r1 = innerRing->getrmin(); break;
	}
	for (int ic2 = 0; ic2<4; ++ic2) {
	  double z2 = 0;
	  double r2 = 0;
	  switch (ic2) {
	  case 0: z2 = outerRing->getzmin(); r2 = outerRing->getrmin(); break;
	  case 1: z2 = outerRing->getzmin(); r2 = outerRing->getrmax(); break;
	  case 2: z2 = outerRing->getzmax(); r2 = outerRing->getrmax(); break;
	  case 3: z2 = outerRing->getzmax(); r2 = outerRing->getrmin(); break;
	  }
	  a3[ic1*4+ic2] = (z2 - z1)/(r2 - r1);
	  b3[ic1*4+ic2] = (r2*z1 - r1*z2)/(r2 - r1);
	}
      }
      // probe the rest of the rings for compatibility
      Roads::RoadSet *set = new Roads::RoadSet;
      std::vector<Ring>* ringvector = rings_->getRings();
      std::vector<Ring>::iterator ringvectoriterator     = ringvector->begin();
      std::vector<Ring>::iterator ringvectoriteratorend  = ringvector->end();
      for ( ; ringvectoriterator != ringvectoriteratorend; ++ringvectoriterator ) {
    	Ring* ring = &*ringvectoriterator;
	// is it compatible with beam spot : ring 1 ?
	zmin = 999;
	zmax = -zmin;
	for (int m=0; m<2; ++m) {
	  double r = m==0 ? ring->getrmin() : ring->getrmax();
	  for (int k=0; k<8; ++k) {
	    double z = a1[k]*r + b1[k];
	    if (zmin>z) zmin = z;
	    if (zmax<z) zmax = z;
	  }
	}
	if (ring->getzmax()<zmin || ring->getzmin()>zmax) continue;
	// is it compatible with beam spot : ring 2 ?
	zmin = 999;
	zmax = -zmin;
	for (int m=0; m<2; ++m) {
	  double r = m==0 ? ring->getrmin() : ring->getrmax();
	  for (int k=0; k<8; ++k) {
	    double z = a2[k]*r + b2[k];
	    if (zmin>z) zmin = z;
	    if (zmax<z) zmax = z;
	  }
	}
	if (ring->getzmax()<zmin || ring->getzmin()>zmax) continue;
	// is it compatible with ring 1 : ring 2 ?
	zmin = 999;
	zmax = -zmin;
	for (int m=0; m<2; ++m) {
	  double r = m==0 ? ring->getrmin() : ring->getrmax();
	  for (int k=0; k<16; ++k) {
	    double z = a3[k]*r + b3[k];
	    if (zmin>z) zmin = z;
	    if (zmax<z) zmax = z;
	  }
	}
	if (ring->getzmax()<zmin || ring->getzmin()>zmax) continue;
	// fine, record the result
	set->push_back(Ring(ring));
	++ntriplets;
      }

      roads_->insert(seed,set);
    }
  }

  roads_->setNumberOfLayersPerSubdetector(rings_->getNumberOfLayersPerSubdetector());

  edm::LogInfo("RoadSearch") << "created: " << npairs << " RoadSets";

}

void RoadMaker::collectInnerSeedRings(std::vector<Ring*>& set) {

  collectInnerTIBSeedRings(set);
  collectInnerTIDSeedRings(set);
  collectInnerTECSeedRings(set);

  edm::LogError("RoadSearch") << "collected " << set.size() << " inner seed rings"; 
  
}

void RoadMaker::collectInnerTIBSeedRings(std::vector<Ring*>& set) {

  // TIB
  unsigned int counter     = 0;
  unsigned int layer_max   = 2;
  unsigned int fw_bw_max   = 2;
  unsigned int ext_int_max = 2;
  unsigned int detector_max  = 3;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = 0; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	  Ring* temp_ring = rings_->getTrackerTIBRing(layer,fw_bw,ext_int,detector);
	  set.push_back(temp_ring);
	  ++counter;
	  //edm::LogError("RoadSearch") << "collected TIB inner seed ring with index: " << temp_ring->getindex(); 
	}    
      }    
    }
  }

  edm::LogError("RoadSearch") << "collected " << counter << " TIB inner seed rings"; 

}

void RoadMaker::collectInnerTIDSeedRings(std::vector<Ring*>& set) {

  // TID
  unsigned int counter = 0;

  unsigned int fw_bw_max       = 2;
  unsigned int wheel_max       = 3;
  unsigned int ring_max        = 2;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = 0; ring < ring_max; ++ring ) {
	Ring* temp_ring = rings_->getTrackerTIDRing(fw_bw,wheel,ring);
	//edm::LogError("RoadSearch") << "collected TID inner seed ring with index: " << temp_ring->getindex(); 
	set.push_back(temp_ring);
	++counter;
      }    
    }
  }

  edm::LogError("RoadSearch") << "collected " << counter << " TID inner seed rings"; 

}

void RoadMaker::collectInnerTECSeedRings(std::vector<Ring*>& set) {

  // TEC
  unsigned int counter = 0;

  unsigned int fw_bw_max       = 2;
  unsigned int wheel_max       = 6;
  unsigned int ring_min[6];
  unsigned int ring_max[6];

  // TEC WHEEL 1
  ring_min[0] = 0;
  ring_max[0] = 2;
  // TEC WHEEL 2
  ring_min[1] = 0;
  ring_max[1] = 2;
  // TEC WHEEL 3
  ring_min[2] = 0;
  ring_max[2] = 2;
  // TEC WHEEL 4
  ring_min[3] = 1;
  ring_max[3] = 2;
  // TEC WHEEL 5
  ring_min[4] = 1;
  ring_max[4] = 2;
  // TEC WHEEL 6
  ring_min[5] = 1;
  ring_max[5] = 2;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = ring_min[wheel]; ring < ring_max[wheel]; ++ring ) {
	Ring* temp_ring = rings_->getTrackerTECRing(fw_bw,wheel,ring);
	//edm::LogError("RoadSearch") << "collected TEC inner seed ring with index: " << temp_ring->getindex(); 
	set.push_back(temp_ring);
	++counter;
      }    
    }
  }

  edm::LogError("RoadSearch") << "collected " << counter << " TEC inner seed rings"; 

}

void RoadMaker::collectOuterSeedRings(std::vector<Ring*>& set) {
  
  collectOuterTOBSeedRings(set);
  collectOuterTECSeedRings(set);

  edm::LogError("RoadSearch") << "collected " << set.size() << " outer seed rings"; 
}

void RoadMaker::collectOuterTOBSeedRings(std::vector<Ring*>& set) {

  // TOB
  unsigned int counter = 0;
  unsigned int layer_start     = 4;
  unsigned int layer_max       = 6;
  unsigned int rod_fw_bw_max   = 2;
  unsigned int detector_max      = 6;

  for ( unsigned int layer = layer_start; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = 0; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	Ring* temp_ring = rings_->getTrackerTOBRing(layer,rod_fw_bw,detector);
	//edm::LogError("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
	set.push_back(temp_ring);
	++counter;
      }    
    }    
  }
  
  // add most outer rings
  Ring* temp_ring = rings_->getTrackerTOBRing(1,0,5);
  //edm::LogError("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
  set.push_back(temp_ring);
  ++counter;
  temp_ring = rings_->getTrackerTOBRing(1,1,5);
  //edm::LogError("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
  set.push_back(temp_ring);
  ++counter;
  temp_ring = rings_->getTrackerTOBRing(2,0,5);
  //edm::LogError("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
  set.push_back(temp_ring);
  ++counter;
  temp_ring = rings_->getTrackerTOBRing(2,1,5);
  //edm::LogError("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
  set.push_back(temp_ring);
  ++counter;
  temp_ring = rings_->getTrackerTOBRing(3,0,5);
  //edm::LogError("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
  set.push_back(temp_ring);
  ++counter;
  temp_ring = rings_->getTrackerTOBRing(3,1,5);
  //edm::LogError("RoadSearch") << "collected TOB outer seed ring with index: " << temp_ring->getindex(); 
  set.push_back(temp_ring);
  ++counter;

  edm::LogError("RoadSearch") << "collected " << counter << " TOB outer seed rings"; 

}

void RoadMaker::collectOuterTECSeedRings(std::vector<Ring*>& set) {

  // TEC
  unsigned int counter = 0;

  unsigned int fw_bw_max       = 2;
  unsigned int wheel_max       = 7;
  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      Ring* temp_ring = rings_->getTrackerTECRing(fw_bw,wheel,6);
      //edm::LogError("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
      set.push_back(temp_ring);
      ++counter;
    }
  }

  // add last two wheels
  fw_bw_max       = 2;
  unsigned int wheel_start     = 7;
  wheel_max       = 9;
  unsigned int second_ring_min[9];
  unsigned int second_ring_max[9];

  // TEC WHEEL 1
  second_ring_min[0] = 0;
  second_ring_max[0] = 7;
  // TEC WHEEL 2
  second_ring_min[1] = 0;
  second_ring_max[1] = 7;
  // TEC WHEEL 3
  second_ring_min[2] = 0;
  second_ring_max[2] = 7;
  // TEC WHEEL 4
  second_ring_min[3] = 1;
  second_ring_max[3] = 7;
  // TEC WHEEL 5
  second_ring_min[4] = 1;
  second_ring_max[4] = 7;
  // TEC WHEEL 6
  second_ring_min[5] = 1;
  second_ring_max[5] = 7;
  // TEC WHEEL 7
  second_ring_min[6] = 2;
  second_ring_max[6] = 7;
  // TEC WHEEL 8
  second_ring_min[7] = 2;
  second_ring_max[7] = 7;
  // TEC WHEEL 9
  second_ring_min[8] = 3;
  second_ring_max[8] = 7;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = wheel_start; wheel < wheel_max; ++wheel ) {
      for ( unsigned int second_ring = second_ring_min[wheel]; second_ring < second_ring_max[wheel]; ++second_ring ) {
	Ring* temp_ring = rings_->getTrackerTECRing(fw_bw,wheel,second_ring);
	//edm::LogError("RoadSearch") << "collected TEC outer seed ring with index: " << temp_ring->getindex(); 
	set.push_back(temp_ring);
	++counter;
      }    
    }
  }

  edm::LogError("RoadSearch") << "collected " << counter << " TEC outer seed rings"; 

}

std::string RoadMaker::printTrackerDetUnits(const TrackerGeometry &tracker) {

  std::ostringstream output;

  std::vector<DetId> detIds = tracker.detUnitIds();
  
  for ( std::vector<DetId>::iterator detiterator = detIds.begin(); detiterator != detIds.end(); ++detiterator ) {
    DetId id = *detiterator;

    if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
      TIBDetId tibid(id.rawId()); 
      output << "[RoadMaker] DetUnit for TIB ring Detid: " << id.rawId() 
	     << " layer: " << tibid.layer() 
	     << " fw(0)/bw(1): " << tibid.string()[0]
	     << " ext(0)/int(0): " << tibid.string()[1] 
	     << " string: " << tibid.string()[2] 
	     << " detector: " << tibid.module()
	     << " not stereo(0)/stereo(1): " << tibid.stereo() 
	     << " not glued(0)/glued(1): " << tibid.glued() 
	     << std::endl; 
    } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
      TOBDetId tobid(id.rawId()); 
      output << "[RoadMaker] DetUnit for TOB ring Detid: " << id.rawId() 
	     << " layer: " << tobid.layer() 
	     << " fw(0)/bw(1): " << tobid.rod()[0]
	     << " rod: " << tobid.rod()[1] 
	     << " detector: " << tobid.module()
	     << " not stereo(0)/stereo(1): " << tobid.stereo() 
	     << " not glued(0)/glued(1): " << tobid.glued() 
	     << std::endl; 
    } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
      TIDDetId tidid(id.rawId()); 
      output << "[RoadMaker] DetUnit for TID ring Detid: " << id.rawId() 
	     << " side neg(1)/pos(2): " << tidid.side() 
	     << " wheel: " << tidid.wheel()
	     << " ring: " << tidid.ring()
	     << " detector fw(0)/bw(1): " << tidid.module()[0]
	     << " detector: " << tidid.module()[1] 
	     << " not stereo(0)/stereo(1): " << tidid.stereo() 
	     << " not glued(0)/glued(1): " << tidid.glued() 
	     << std::endl; 
    } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
      TECDetId tecid(id.rawId()); 
      output << "[RoadMaker] DetUnit for TEC ring DetId: " << id.rawId() 
	     << " side neg(1)/pos(2): " << tecid.side() 
	     << " wheel: " << tecid.wheel()
	     << " petal fw(0)/bw(1): " << tecid.petal()[0]
	     << " petal: " << tecid.petal()[1] 
	     << " ring: " << tecid.ring()
	     << " module: " << tecid.module();
      if ( ((int)tecid.partnerDetId()-(int)id.rawId()) == 1 ) {
	output << " stereo: 1";
      } else if ( ((int)tecid.partnerDetId()-(int)id.rawId()) == -1 ) {
	output << " stereo: 2";
      } else if ( tecid.partnerDetId() == 0 ) {
	output << " stereo: 0";
      } else {
	output << " stereo: problem";
      }
      output << std::endl; 
    } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
      PXBDetId pxbid(id.rawId()); 
      output << "[RoadMaker] DetUnit for PXB ring DetId: " << id.rawId() 
	     << " layer: " << pxbid.layer()
	     << " ladder: " << pxbid.ladder()
	     << " detector: " << pxbid.module()
	     << std::endl; 
    } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
      PXFDetId pxfid(id.rawId()); 
      output << "[RoadMaker] DetUnit for PXF ring DetId: " << id.rawId() 
	     << " side: " << pxfid.side()
	     << " disk: " << pxfid.disk()
	     << " blade: " << pxfid.blade()
	     << " detector: " << pxfid.module()
	     << std::endl; 
    }


  }

  return output.str();
  
}

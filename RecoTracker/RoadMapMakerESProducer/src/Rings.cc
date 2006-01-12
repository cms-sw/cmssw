// File: Rings.cc
// Description:  see Rings.h
// Author:  O. Gutsche
// Creation Date:  Oct. 14 2005 Initial version.
//
//--------------------------------------------

#include <iostream>
#include <algorithm>
#include <map>
#include <utility>

#include "RecoTracker/RoadMapMakerESProducer/interface/Rings.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/Surface/interface/TrapezoidalPlaneBounds.h"

Rings::Rings(const TrackingGeometry &tracker, unsigned int verbosity) : verbosity_(verbosity) {

  constructTrackerRings(tracker);

  // make same numbering scheme as ORCA RS
  fixIndexNumberingScheme();

}

Rings::~Rings() { 

}  

void Rings::constructTrackerRings(const TrackingGeometry &tracker) {

  constructTrackerTIBRings(tracker);
  constructTrackerTOBRings(tracker);
  constructTrackerTIDRings(tracker);
  constructTrackerTECRings(tracker);
  constructTrackerPXBRings(tracker);
  //   constructTrackerPXFRings(tracker);

  if ( verbosity_ > 0 ) {
    std::cout << "[Rings] constructed " << rings_.size() << " rings" << std::endl; 
  }

}

void Rings::constructTrackerTIBRings(const TrackingGeometry &tracker) {

  unsigned int counter = 0;
  unsigned int index = 24;

  unsigned int layer_max   = 4;
  unsigned int fw_bw_max   = 2;
  unsigned int ext_int_max = 2;
  unsigned int detector_max  = 3;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = 0; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	  Ring ring = constructTrackerTIBRing(tracker,layer,fw_bw,ext_int,detector);
	  ring.setindex(index++);
	  rings_.push_back(ring);
	  ++counter;
	  if ( verbosity_ > 2 ) {
	    std::cout << "[Rings] constructed TIB ring with index: " << ring.getindex() << " consisting of " << ring.getNumDetIds() << " DetIds" << std::endl; 
	  }
	}    
      }    
    }    
  }

  if ( verbosity_ > 1 ) {
    std::cout << "[Rings] constructed " << counter << " TIB rings" << std::endl; 
  }
  
}

Ring Rings::constructTrackerTIBRing(const TrackingGeometry &tracker,
				    unsigned int layer,
				    unsigned int fw_bw,
				    unsigned int ext_int,
				    unsigned int detector) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int string_max[4][2];

  // TIB 1 internal
  string_max[0][0] = 26;
  // TIB 1 external
  string_max[0][1] = 30;
  // TIB 2 internal
  string_max[1][0] = 34;
  // TIB 2 external
  string_max[1][1] = 38;
  // TIB 3 internal
  string_max[2][0] = 44;
  // TIB 3 external
  string_max[2][1] = 46;
  // TIB 4 internal
  string_max[3][0] = 52;
  // TIB 4 external
  string_max[3][1] = 56;

  Ring ring(Ring::TIBRing);

  if ( (layer == 0) || (layer == 1) ) {
    for ( unsigned int stereo = 1; stereo <= 2; ++stereo) {
      for ( unsigned int string = 0; string < string_max[layer][ext_int]; ++string ) {
	DetId id = constructTrackerTIBDetId(layer,fw_bw,ext_int,string,detector,stereo);
	double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TIBRing);
	ring.addId(phi,id);
      }
    }
  } else {
    for ( unsigned int string = 0; string < string_max[layer][ext_int]; ++string ) {
      DetId id = constructTrackerTIBDetId(layer,fw_bw,ext_int,string,detector,0);
      double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TIBRing);
      ring.addId(phi,id);
    }
  }

  if ( verbosity_ > 2 ) {
    std::cout << "[Rings] Ring with index: " << ring.getindex() << " initialized rmin/rmax/zmin/zmax: " << rmin << "/" << rmax << "/" << zmin << "/" << zmax << std::endl;
  }
    
  ring.initialize(rmin,rmax,zmin,zmax);

  return ring;
}

DetId Rings::constructTrackerTIBDetId(unsigned int layer,
				      unsigned int fw_bw,
				      unsigned int ext_int,
				      unsigned int string,
				      unsigned int detector,
				      unsigned int stereo) {

   TIBDetId id(layer+1,fw_bw,ext_int,string+1,detector+1,stereo);
   if ( verbosity_ > 3 ) {
     std::cout << "[Rings] constructed TIB ring DetId for layer: " << id.layer() << " fw(0)/bw(1): " << id.string()[0]
	       << " ext(0)/int(1): " << id.string()[1] << " string: " << id.string()[2] << " sensor: " << id.det()
	       << " stereo: " << id.stereo() << std::endl; 
   }

  return DetId(id.rawId());
}


void Rings::constructTrackerTOBRings(const TrackingGeometry &tracker) {

  unsigned int counter = 0;
  unsigned int index = 72;

  unsigned int layer_max       = 6;
  unsigned int rod_fw_bw_max   = 2;
  unsigned int detector_max      = 6;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = 0; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	Ring ring = constructTrackerTOBRing(tracker,layer,rod_fw_bw,detector);
	ring.setindex(index++);
	rings_.push_back(ring);
	++counter;
	if ( verbosity_ > 2 ) {
	  std::cout << "[Rings] constructed TOB ring with index: " << ring.getindex() << " consisting of " << ring.getNumDetIds() << " DetIds" << std::endl; 
	}
      }    
    }    
  }

  if ( verbosity_ > 1 ) {
    std::cout << "[Rings] constructed " << counter << " TOB rings" << std::endl; 
  }

}

Ring Rings::constructTrackerTOBRing(const TrackingGeometry &tracker,
				    unsigned int layer,
				    unsigned int rod_fw_bw,
				    unsigned int detector) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int rod_max[6];
  
  // TOB 1
  rod_max[0] = 42;
  // TOB 2
  rod_max[1] = 48;
  // TOB 3
  rod_max[2] = 54;
  // TOB 4
  rod_max[3] = 60;
  // TOB 5
  rod_max[4] = 66;
  // TOB 6
  rod_max[5] = 74;


  Ring ring(Ring::TOBRing);
  
  if ( (layer == 0) || (layer == 1) ) {
    for ( unsigned int stereo = 1; stereo <= 2; ++stereo) {
      for ( unsigned int rod = 0; rod < rod_max[layer]; ++rod ) {
	DetId id = constructTrackerTOBDetId(layer,rod_fw_bw,rod,detector,stereo);
	double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TOBRing);
	ring.addId(phi,id);
      }
    }
  } else {
    for ( unsigned int rod = 0; rod < rod_max[layer]; ++rod ) {
      DetId id = constructTrackerTOBDetId(layer,rod_fw_bw,rod,detector,0);
      double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TOBRing);
      ring.addId(phi,id);
    }
  }

  if ( verbosity_ > 2 ) {
    std::cout << "[Rings] Ring with index: " << ring.getindex() << " initialized rmin/rmax/zmin/zmax: " << rmin << "/" << rmax << "/" << zmin << "/" << zmax << std::endl;
  }
    
  ring.initialize(rmin,rmax,zmin,zmax);

  return ring;
}

DetId Rings::constructTrackerTOBDetId(unsigned int layer,
				      unsigned int rod_fw_bw,
				      unsigned int rod,
				      unsigned int detector,
				      unsigned int stereo) {

  TOBDetId id(layer+1,rod_fw_bw,rod+1,detector+1,stereo);

  if ( verbosity_ > 3 ) {
    std::cout << "[Rings] constructed TOB ring DetId for layer: " << id.layer() << " rod fw(0)/bw(1): " << id.rod()[0] 
	      << " rod: " << id.rod()[1] << " sensor: " << id.det() << " stereo: " << id.stereo() << std::endl; 
  }

  return DetId(id.rawId());
}

void Rings::constructTrackerTIDRings(const TrackingGeometry &tracker) {

  unsigned int counter = 0;
  unsigned int index = 197;

  unsigned int fw_bw_max       = 2;
  unsigned int wheel_max       = 3;
  unsigned int ring_max        = 3;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = 0; ring < ring_max; ++ring ) {
	Ring tempring = constructTrackerTIDRing(tracker,fw_bw,wheel,ring);
	tempring.setindex(index++);
	if ( index == 206 ) {
	  index = 234;
	}
	rings_.push_back(tempring);
	++counter;
	if ( verbosity_ > 2 ) {
	  std::cout << "[Rings] constructed TID ring with index: " << tempring.getindex() << " consisting of " << tempring.getNumDetIds() << " DetIds" << std::endl; 
	}
      }   
    }  
  }
  
  if ( verbosity_ > 1 ) {
    std::cout << "[Rings] constructed " << counter << " TID rings" << std::endl; 
  }

}

Ring Rings::constructTrackerTIDRing(const TrackingGeometry &tracker,
				    unsigned int fw_bw,
				    unsigned int wheel,
				    unsigned int ring) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int detector_fw_bw_max   = 2;
  unsigned int detector_max[3];
  
  // TID 1
  detector_max[0] = 12;
  // TID 2
  detector_max[1] = 12;
  // TID 3
  detector_max[2] = 20;

  Ring tempring(Ring::TIDRing);
  
  if ( (ring==0) || (ring==1) ) {
    for ( unsigned int stereo = 1; stereo <= 2; ++stereo) {
      for ( unsigned int detector_fw_bw = 0; detector_fw_bw < detector_fw_bw_max; ++detector_fw_bw ) {
	for ( unsigned int detector = 0; detector < detector_max[ring]; ++detector ) {
	  DetId id = constructTrackerTIDDetId(fw_bw,wheel,ring,detector_fw_bw,detector,stereo);
	  double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TIDRing);
	  tempring.addId(phi,id);
	}
      }
    }
  } else {
    for ( unsigned int detector_fw_bw = 0; detector_fw_bw < detector_fw_bw_max; ++detector_fw_bw ) {
      for ( unsigned int detector = 0; detector < detector_max[ring]; ++detector ) {
	DetId id = constructTrackerTIDDetId(fw_bw,wheel,ring,detector_fw_bw,detector,0);
	double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TIDRing);
	tempring.addId(phi,id);
      }
    }
  }
	
  if ( verbosity_ > 2 ) {
    std::cout << "[Rings] Ring with index: " << tempring.getindex() << " initialized rmin/rmax/zmin/zmax: " << rmin << "/" << rmax << "/" << zmin << "/" << zmax << std::endl;
  }
    
  tempring.initialize(rmin,rmax,zmin,zmax);

  return tempring;
}

DetId Rings::constructTrackerTIDDetId(unsigned int fw_bw,
				      unsigned int wheel,
				      unsigned int ring,
				      unsigned int detector_fw_bw,
				      unsigned int detector,
				      unsigned int stereo) {

  TIDDetId id(fw_bw+1,wheel+1,ring+1,detector_fw_bw,detector+1,stereo);

  if ( verbosity_ > 3 ) {
    std::cout << "[Rings] constructed TID ring DetId for side: " << id.side() << " wheel: " << id.wheel() 
	      << " ring: " << id.ring() << "detector fw(0)/bw(1): " << id.det()[0] << " detector: " << id.det()[1] 
	      << " stereo: " << id.stereo() << std::endl; 
  }
	
  return DetId(id.rawId());
}

void Rings::constructTrackerTECRings(const TrackingGeometry &tracker) {

  unsigned int counter = 0;
  unsigned int index = 144;

  unsigned int fw_bw_max       = 2;
  unsigned int wheel_max       = 9;
  unsigned int ring_max[9];

  // TEC WHEEL 1
  ring_max[0] = 7;
  // TEC WHEEL 2
  ring_max[1] = 7;
  // TEC WHEEL 3
  ring_max[2] = 7;
  // TEC WHEEL 4
  ring_max[3] = 6;
  // TEC WHEEL 5
  ring_max[4] = 6;
  // TEC WHEEL 6
  ring_max[5] = 6;
  // TEC WHEEL 7
  ring_max[6] = 5;
  // TEC WHEEL 8
  ring_max[7] = 5;
  // TEC WHEEL 9
  ring_max[8] = 4;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = 0; ring < ring_max[wheel]; ++ring ) {
	Ring tempring = constructTrackerTECRing(tracker,fw_bw,wheel,ring);
	tempring.setindex(index++);
	if ( index == 197 ) {
	  index = 243;
	}
	rings_.push_back(tempring);
	++counter;
	if ( verbosity_ > 2 ) {
	  std::cout << "[Rings] constructed TEC ring with index: " << tempring.getindex() << " consisting of " << tempring.getNumDetIds() << " DetIds" << std::endl; 
	}
      }   
    }  
  }

  if ( verbosity_ > 1 ) {
    std::cout << "[Rings] constructed " << counter << " TEC rings" << std::endl; 
  }

}

Ring Rings::constructTrackerTECRing(const TrackingGeometry &tracker,
				    unsigned int fw_bw,
				    unsigned int wheel,
				    unsigned int ring) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int petal_max       = 8;
  unsigned int petal_fw_bw_max = 2;
  unsigned int detector_fw_bw_max = 2;
  unsigned int detector_max[9][7][2][2];

  // ========================================== //

  // TEC Wheel 1 Ring 1
  detector_max[0][0][0][0] = 1;
  detector_max[0][0][0][1] = 1;
  detector_max[0][0][1][0] = 0;
  detector_max[0][0][1][1] = 1;

  // TEC Wheel 1 Ring 2
  detector_max[0][1][0][0] = 1;
  detector_max[0][1][0][1] = 1;
  detector_max[0][1][1][0] = 1;
  detector_max[0][1][1][1] = 0;

  // TEC Wheel 1 Ring 3
  detector_max[0][2][0][0] = 2;
  detector_max[0][2][0][1] = 1;
  detector_max[0][2][1][0] = 1;
  detector_max[0][2][1][1] = 1;

  // TEC Wheel 1 Ring 4
  detector_max[0][3][0][0] = 2;
  detector_max[0][3][0][1] = 2;
  detector_max[0][3][1][0] = 1;
  detector_max[0][3][1][1] = 2;

  // TEC Wheel 1 Ring 5
  detector_max[0][4][0][0] = 1;
  detector_max[0][4][0][1] = 1;
  detector_max[0][4][1][0] = 2;
  detector_max[0][4][1][1] = 1;

  // TEC Wheel 1 Ring 6
  detector_max[0][5][0][0] = 2;
  detector_max[0][5][0][1] = 2;
  detector_max[0][5][1][0] = 1;
  detector_max[0][5][1][1] = 2;

  // TEC Wheel 1 Ring 7
  detector_max[0][6][0][0] = 2;
  detector_max[0][6][0][1] = 3;
  detector_max[0][6][1][0] = 2;
  detector_max[0][6][1][1] = 3;

  // ========================================== //

  // TEC Wheel 2 Ring 1
  detector_max[1][0][0][0] = 1;
  detector_max[1][0][0][1] = 1;
  detector_max[1][0][1][0] = 0;
  detector_max[1][0][1][1] = 1;

  // TEC Wheel 2 Ring 2
  detector_max[1][1][0][0] = 1;
  detector_max[1][1][0][1] = 1;
  detector_max[1][1][1][0] = 1;
  detector_max[1][1][1][1] = 0;

  // TEC Wheel 2 Ring 3
  detector_max[1][2][0][0] = 2;
  detector_max[1][2][0][1] = 1;
  detector_max[1][2][1][0] = 1;
  detector_max[1][2][1][1] = 1;

  // TEC Wheel 2 Ring 4
  detector_max[1][3][0][0] = 2;
  detector_max[1][3][0][1] = 2;
  detector_max[1][3][1][0] = 1;
  detector_max[1][3][1][1] = 2;

  // TEC Wheel 2 Ring 5
  detector_max[1][4][0][0] = 1;
  detector_max[1][4][0][1] = 1;
  detector_max[1][4][1][0] = 2;
  detector_max[1][4][1][1] = 1;

  // TEC Wheel 2 Ring 6
  detector_max[1][5][0][0] = 2;
  detector_max[1][5][0][1] = 2;
  detector_max[1][5][1][0] = 1;
  detector_max[1][5][1][1] = 2;

  // TEC Wheel 2  Ring 7
  detector_max[1][6][0][0] = 2;
  detector_max[1][6][0][1] = 3;
  detector_max[1][6][1][0] = 2;
  detector_max[1][6][1][1] = 3;

  // ========================================== //

  // TEC Wheel 3 Ring 1
  detector_max[2][0][0][0] = 1;
  detector_max[2][0][0][1] = 1;
  detector_max[2][0][1][0] = 0;
  detector_max[2][0][1][1] = 1;

  // TEC Wheel 3 Ring 2
  detector_max[2][1][0][0] = 1;
  detector_max[2][1][0][1] = 1;
  detector_max[2][1][1][0] = 1;
  detector_max[2][1][1][1] = 0;

  // TEC Wheel 3 Ring 3
  detector_max[2][2][0][0] = 2;
  detector_max[2][2][0][1] = 1;
  detector_max[2][2][1][0] = 1;
  detector_max[2][2][1][1] = 1;

  // TEC Wheel 3 Ring 4
  detector_max[2][3][0][0] = 2;
  detector_max[2][3][0][1] = 2;
  detector_max[2][3][1][0] = 1;
  detector_max[2][3][1][1] = 2;

  // TEC Wheel 3 Ring 5
  detector_max[2][4][0][0] = 1;
  detector_max[2][4][0][1] = 1;
  detector_max[2][4][1][0] = 2;
  detector_max[2][4][1][1] = 1;

  // TEC Wheel 3 Ring 6
  detector_max[2][5][0][0] = 2;
  detector_max[2][5][0][1] = 2;
  detector_max[2][5][1][0] = 1;
  detector_max[2][5][1][1] = 2;

  // TEC Wheel 3  Ring 7
  detector_max[2][6][0][0] = 2;
  detector_max[2][6][0][1] = 3;
  detector_max[2][6][1][0] = 2;
  detector_max[2][6][1][1] = 3;

  // ========================================== //

  // TEC Wheel 4 Ring 2
  detector_max[3][0][0][0] = 1;
  detector_max[3][0][0][1] = 1;
  detector_max[3][0][1][0] = 1;
  detector_max[3][0][1][1] = 0;

  // TEC Wheel 4 Ring 3
  detector_max[3][1][0][0] = 2;
  detector_max[3][1][0][1] = 1;
  detector_max[3][1][1][0] = 1;
  detector_max[3][1][1][1] = 1;

  // TEC Wheel 4 Ring 4
  detector_max[3][2][0][0] = 2;
  detector_max[3][2][0][1] = 2;
  detector_max[3][2][1][0] = 1;
  detector_max[3][2][1][1] = 2;

  // TEC Wheel 4 Ring 5
  detector_max[3][3][0][0] = 1;
  detector_max[3][3][0][1] = 1;
  detector_max[3][3][1][0] = 2;
  detector_max[3][3][1][1] = 1;

  // TEC Wheel 4 Ring 6
  detector_max[3][4][0][0] = 2;
  detector_max[3][4][0][1] = 2;
  detector_max[3][4][1][0] = 1;
  detector_max[3][4][1][1] = 2;

  // TEC Wheel 4 Ring 7
  detector_max[3][5][0][0] = 2;
  detector_max[3][5][0][1] = 3;
  detector_max[3][5][1][0] = 2;
  detector_max[3][5][1][1] = 3;

  // ========================================== //

  // TEC Wheel 5 Ring 2
  detector_max[4][0][0][0] = 1;
  detector_max[4][0][0][1] = 1;
  detector_max[4][0][1][0] = 1;
  detector_max[4][0][1][1] = 0;

  // TEC Wheel 5 Ring 3
  detector_max[4][1][0][0] = 2;
  detector_max[4][1][0][1] = 1;
  detector_max[4][1][1][0] = 1;
  detector_max[4][1][1][1] = 1;

  // TEC Wheel 5 Ring 4
  detector_max[4][2][0][0] = 2;
  detector_max[4][2][0][1] = 2;
  detector_max[4][2][1][0] = 1;
  detector_max[4][2][1][1] = 2;

  // TEC Wheel 5 Ring 5
  detector_max[4][3][0][0] = 1;
  detector_max[4][3][0][1] = 1;
  detector_max[4][3][1][0] = 2;
  detector_max[4][3][1][1] = 1;

  // TEC Wheel 5 Ring 6
  detector_max[4][4][0][0] = 2;
  detector_max[4][4][0][1] = 2;
  detector_max[4][4][1][0] = 1;
  detector_max[4][4][1][1] = 2;

  // TEC Wheel 5 Ring 7
  detector_max[4][5][0][0] = 2;
  detector_max[4][5][0][1] = 3;
  detector_max[4][5][1][0] = 2;
  detector_max[4][5][1][1] = 3;

  // ========================================== //

  // TEC Wheel 6 Ring 2
  detector_max[5][0][0][0] = 1;
  detector_max[5][0][0][1] = 1;
  detector_max[5][0][1][0] = 1;
  detector_max[5][0][1][1] = 0;

  // TEC Wheel 6 Ring 3
  detector_max[5][1][0][0] = 2;
  detector_max[5][1][0][1] = 1;
  detector_max[5][1][1][0] = 1;
  detector_max[5][1][1][1] = 1;

  // TEC Wheel 6 Ring 4
  detector_max[5][2][0][0] = 2;
  detector_max[5][2][0][1] = 2;
  detector_max[5][2][1][0] = 1;
  detector_max[5][2][1][1] = 2;

  // TEC Wheel 6 Ring 5
  detector_max[5][3][0][0] = 1;
  detector_max[5][3][0][1] = 1;
  detector_max[5][3][1][0] = 2;
  detector_max[5][3][1][1] = 1;

  // TEC Wheel 6 Ring 6
  detector_max[5][4][0][0] = 2;
  detector_max[5][4][0][1] = 2;
  detector_max[5][4][1][0] = 1;
  detector_max[5][4][1][1] = 2;

  // TEC Wheel 6 Ring 7
  detector_max[5][5][0][0] = 2;
  detector_max[5][5][0][1] = 3;
  detector_max[5][5][1][0] = 2;
  detector_max[5][5][1][1] = 3;

  // ========================================== //

  // TEC Wheel 7 Ring 3
  detector_max[6][0][0][0] = 2;
  detector_max[6][0][0][1] = 1;
  detector_max[6][0][1][0] = 1;
  detector_max[6][0][1][1] = 1;

  // TEC Wheel 7 Ring 4
  detector_max[6][1][0][0] = 2;
  detector_max[6][1][0][1] = 2;
  detector_max[6][1][1][0] = 1;
  detector_max[6][1][1][1] = 2;

  // TEC Wheel 7 Ring 5
  detector_max[6][2][0][0] = 1;
  detector_max[6][2][0][1] = 1;
  detector_max[6][2][1][0] = 2;
  detector_max[6][2][1][1] = 1;

  // TEC Wheel 7 Ring 6
  detector_max[6][3][0][0] = 2;
  detector_max[6][3][0][1] = 2;
  detector_max[6][3][1][0] = 1;
  detector_max[6][3][1][1] = 2;

  // TEC Wheel 7 Ring 7
  detector_max[6][4][0][0] = 2;
  detector_max[6][4][0][1] = 3;
  detector_max[6][4][1][0] = 2;
  detector_max[6][4][1][1] = 3;

  // ========================================== //

  // TEC Wheel 8 Ring 3
  detector_max[7][0][0][0] = 2;
  detector_max[7][0][0][1] = 1;
  detector_max[7][0][1][0] = 1;
  detector_max[7][0][1][1] = 1;

  // TEC Wheel 8 Ring 4
  detector_max[7][1][0][0] = 2;
  detector_max[7][1][0][1] = 2;
  detector_max[7][1][1][0] = 1;
  detector_max[7][1][1][1] = 2;

  // TEC Wheel 8 Ring 5
  detector_max[7][2][0][0] = 1;
  detector_max[7][2][0][1] = 1;
  detector_max[7][2][1][0] = 2;
  detector_max[7][2][1][1] = 1;

  // TEC Wheel 8 Ring 6
  detector_max[7][3][0][0] = 2;
  detector_max[7][3][0][1] = 2;
  detector_max[7][3][1][0] = 1;
  detector_max[7][3][1][1] = 2;

  // TEC Wheel 8 Ring 7
  detector_max[7][4][0][0] = 2;
  detector_max[7][4][0][1] = 3;
  detector_max[7][4][1][0] = 2;
  detector_max[7][4][1][1] = 3;

  // ========================================== //

  // TEC Wheel 9 Ring 4
  detector_max[8][0][0][0] = 2;
  detector_max[8][0][0][1] = 2;
  detector_max[8][0][1][0] = 1;
  detector_max[8][0][1][1] = 2;

  // TEC Wheel 9 Ring 5
  detector_max[8][1][0][0] = 1;
  detector_max[8][1][0][1] = 1;
  detector_max[8][1][1][0] = 2;
  detector_max[8][1][1][1] = 1;

  // TEC Wheel 9 Ring 6
  detector_max[8][2][0][0] = 2;
  detector_max[8][2][0][1] = 2;
  detector_max[8][2][1][0] = 1;
  detector_max[8][2][1][1] = 2;

  // TEC Wheel 9 Ring 7
  detector_max[8][3][0][0] = 2;
  detector_max[8][3][0][1] = 3;
  detector_max[8][3][1][0] = 2;
  detector_max[8][3][1][1] = 3;

  // ========================================== //

  bool stereo = false;
  if ( (wheel==0)||(wheel==1)||(wheel==2) ) {
    if ( (ring==0) || (ring==1) || (ring==4) ) {
      stereo = true;
    } 
  } else if ( (wheel==3)||(wheel==4)||(wheel==5) ) {
    if ( (ring==0) || (ring==3) ) {
      stereo = true;
    } 
  } else if ( (wheel==6)||(wheel==7) ) {
    if ( (ring==2) ) {
      stereo = true;
    } 
  } else if ( (wheel==8) ) {
    if ( (ring==1) ) {
      stereo = true;
    } 
  }
  
  Ring tempring(Ring::TECRing);
	
  if ( stereo ) {
    for ( unsigned int stereo = 1; stereo <= 2; ++stereo) {
      for ( unsigned int petal = 0; petal < petal_max; ++petal ) {
	for ( unsigned int petal_fw_bw = 0; petal_fw_bw < petal_fw_bw_max; ++petal_fw_bw ) {
	  for ( unsigned int detector_fw_bw = 0; detector_fw_bw < detector_fw_bw_max; ++detector_fw_bw ) {
	    for ( unsigned int detector = 0; detector < detector_max[wheel][ring][petal_fw_bw][detector_fw_bw]; ++detector ) {
	      DetId id = constructTrackerTECDetId(fw_bw,wheel,petal_fw_bw,petal,ring,detector_fw_bw,detector,stereo);
	      double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TECRing);
	      tempring.addId(phi,id);
	    }
	  }
	}
      }
    }
  } else {
    for ( unsigned int petal = 0; petal < petal_max; ++petal ) {
      for ( unsigned int petal_fw_bw = 0; petal_fw_bw < petal_fw_bw_max; ++petal_fw_bw ) {
	for ( unsigned int detector_fw_bw = 0; detector_fw_bw < detector_fw_bw_max; ++detector_fw_bw ) {
	  for ( unsigned int detector = 0; detector < detector_max[wheel][ring][petal_fw_bw][detector_fw_bw]; ++detector ) {
	    DetId id = constructTrackerTECDetId(fw_bw,wheel,petal_fw_bw,petal,ring,detector_fw_bw,detector,0);
	    double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::TECRing);
	    tempring.addId(phi,id);
	  }
	}
      }
    }
  }

  if ( verbosity_ > 2 ) {
    std::cout << "[Rings] Ring with index: " << tempring.getindex() << " initialized rmin/rmax/zmin/zmax: " << rmin << "/" << rmax << "/" << zmin << "/" << zmax << std::endl;
  }
    
  tempring.initialize(rmin,rmax,zmin,zmax);

  return tempring;
}

DetId Rings::constructTrackerTECDetId(unsigned int fw_bw,
				      unsigned int wheel,
				      unsigned int petal_fw_bw,
				      unsigned int petal,
				      unsigned int ring,
				      unsigned int detector_fw_bw,
				      unsigned int detector,
				      unsigned int stereo) {

  TECDetId id(fw_bw+1,wheel+1,petal_fw_bw,petal+1,ring+1,detector_fw_bw,detector+1,stereo);
  
  if ( verbosity_ > 3 ) {
    std::cout << "[Rings] constructed TEC ring DetId for side: " << id.side() << " wheel: " << id.wheel() 
	      << " ring: " << id.ring() << " petal fw(0)/bw(0): " << id.petal()[0] << " petal: " << id.petal()[1] 
	      << "detector fw(0)/bw(1): " << id.det()[0] << " detector: " << id.det()[1] << " stereo: " << id.stereo() << std::endl; 
  }

  return DetId(id.rawId());
}

void Rings::constructTrackerPXBRings(const TrackingGeometry &tracker) {

  unsigned int counter = 0;
  unsigned int index = 0;

  unsigned int layer_max   = 3;
  unsigned int detector_max  = 8;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
      Ring ring = constructTrackerPXBRing(tracker,layer,detector);
      ring.setindex(index++);
      rings_.push_back(ring);
      ++counter;
      if ( verbosity_ > 2 ) {
	std::cout << "[Rings] constructed PXB ring with index: " << ring.getindex() << " consisting of " << ring.getNumDetIds() << " DetIds" << std::endl; 
      }
    }    
  }    

  if ( verbosity_ > 1 ) {
    std::cout << "[Rings] constructed " << counter << " PXB rings" << std::endl; 
  }
  
}

Ring Rings::constructTrackerPXBRing(const TrackingGeometry &tracker,
				    unsigned int layer,
				    unsigned int detector) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int ladder_max[3];

  // PXB 1
  ladder_max[0] = 20;
  // PXB 2
  ladder_max[1] = 32;
  // PXB 3
  ladder_max[2] = 44;

  Ring ring(Ring::PXBRing);

  for ( unsigned int ladder = 0; ladder < ladder_max[layer]; ++ladder ) {
    DetId id = constructTrackerPXBDetId(layer,ladder,detector);
    double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::PXBRing);
    ring.addId(phi,id);
  }
  
  if ( verbosity_ > 2 ) {
    std::cout << "[Rings] Ring with index: " << ring.getindex() << " initialized rmin/rmax/zmin/zmax: " << rmin << "/" << rmax << "/" << zmin << "/" << zmax << std::endl;
  }
    
  ring.initialize(rmin,rmax,zmin,zmax);

  return ring;
}

DetId Rings::constructTrackerPXBDetId(unsigned int layer,
				      unsigned int ladder,
				      unsigned int detector) {
  PXBDetId id(layer+1,ladder+1,detector+1);
	
  if ( verbosity_ > 3 ) {
    std::cout << "[Rings] constructed PXB ring DetId for layer: " << id.layer() << " ladder: " << id.ladder() 
	      << " detector: " << id.det() << std::endl; 
  }

  return DetId(id.rawId());
}

void Rings::constructTrackerPXFRings(const TrackingGeometry &tracker) {

  unsigned int counter = 0;
  unsigned int index = 206;

  unsigned int fw_bw_max   = 2;
  unsigned int disk_max    = 2;
  unsigned int detector_max  = 7;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int disk = 0; disk < disk_max; ++disk ) {
      for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	Ring ring = constructTrackerPXFRing(tracker,fw_bw,disk,detector);
	ring.setindex(index++);
	rings_.push_back(ring);
	++counter;
	if ( verbosity_ > 2 ) {
	  std::cout << "[Rings] constructed PXF ring with index: " << ring.getindex() << " consisting of " << ring.getNumDetIds() << " DetIds" << std::endl; 
	}
      }
    }    
  }    

  if ( verbosity_ > 1 ) {
    std::cout << "[Rings] constructed " << counter << " PXF rings" << std::endl; 
  }
  
}

Ring Rings::constructTrackerPXFRing(const TrackingGeometry &tracker,
				    unsigned int fw_bw,
				    unsigned int disk,
				    unsigned int detector) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int blade_max   = 24;

  Ring ring(Ring::PXFRing);

  for ( unsigned int blade = 0; blade < blade_max; ++blade ) {
    DetId id = constructTrackerPXFDetId(fw_bw,disk,blade,detector);
    double phi = determineExtensions(tracker,id,rmin,rmax,zmin,zmax,Ring::PXFRing);
    ring.addId(phi,id);
  }

  if ( verbosity_ > 2 ) {
    std::cout << "[Rings] Ring with index: " << ring.getindex() << " initialized rmin/rmax/zmin/zmax: " << rmin << "/" << rmax << "/" << zmin << "/" << zmax << std::endl;
  }
    
  ring.initialize(rmin,rmax,zmin,zmax);

  return ring;
}

DetId Rings::constructTrackerPXFDetId(unsigned int fw_bw,
				      unsigned int disk,
				      unsigned int blade,
				      unsigned int detector) {

  PXFDetId id(fw_bw+1,disk+1,blade+1,detector+1);

  if ( verbosity_ > 3 ) {
    std::cout << "[Rings] constructed PXF ring DetId for fw_bw: " << id.side() << " disk: " << id.disk() 
	      << " blade: " << id.blade() << " detector: " << id.det() << std::endl; 
  }
	
  return DetId(id.rawId());
}

Ring* Rings::getTrackerRing(DetId id) {
  
  for ( iterator ring = rings_.begin(); ring != rings_.end(); ++ring ) {
    Ring *temp = &*ring;
    if ( temp->containsDetId(id)) {
      return temp;
    }
  }
  
  std::cout << "[Rings] could not find Ring with DetId: " << id.rawId() << std::endl; 
  
  return 0;
}

Ring* Rings::getTrackerTIBRing(unsigned int layer,
			       unsigned int fw_bw,
			       unsigned int ext_int,
			       unsigned int detector) {

  // construct DetID from info using else the first of all entities and return Ring

  unsigned int stereo = 0;
  if ( (layer == 0) || (layer == 1) ) {
    stereo = 1;
  }

  TIBDetId id(layer+1,fw_bw,ext_int,1,detector+1,stereo);

  return getTrackerRing(DetId(id.rawId()));
}

Ring* Rings::getTrackerTIDRing(unsigned int fw_bw,
			       unsigned int wheel,
			       unsigned int ring) {

  // construct DetID from info using else the first of all entities and return Ring

  unsigned int stereo = 0;
  if ( (ring==0) || (ring==1) ) {
    stereo = 1;
  }

  TIDDetId id(fw_bw+1,wheel+1,ring+1,0,1,stereo);

  return getTrackerRing(DetId(id.rawId()));
}

Ring* Rings::getTrackerTECRing(unsigned int fw_bw,
			       unsigned int wheel,
			       unsigned int ring) {

  // construct DetID from info using else the first of all entities and return Ring
  bool stereo_flag = false;
  if ( (wheel==0)||(wheel==1)||(wheel==2) ) {
    if ( (ring==0) || (ring==1) || (ring==4) ) {
      stereo_flag = true;
    } 
  } else if ( (wheel==3)||(wheel==4)||(wheel==5) ) {
    if ( (ring==0) || (ring==3) ) {
      stereo_flag = true;
    } 
  } else if ( (wheel==6)||(wheel==7) ) {
    if ( (ring==2) ) {
      stereo_flag = true;
    } 
  } else if ( (wheel==8) ) {
    if ( (ring==1) ) {
      stereo_flag = true;
    } 
  }

  unsigned int stereo = 0;
  if ( stereo_flag ) {
    stereo = 1;
  }

  TECDetId id(fw_bw+1,wheel+1,0,1,ring+1,0,1,stereo);

  return getTrackerRing(DetId(id.rawId()));
}

Ring* Rings::getTrackerTOBRing(unsigned int layer,
			       unsigned int rod_fw_bw,
			       unsigned int detector) {

  // construct DetID from info using else the first of all entities and return Ring
  unsigned int stereo = 0;
  if ( (layer == 0) || (layer == 1) ) {
    stereo = 1;
  }

  TOBDetId id(layer+1,rod_fw_bw,1,detector+1,stereo);

  return getTrackerRing(DetId(id.rawId()));
}

Ring* Rings::getTrackerPXBRing(unsigned int layer,
			       unsigned int detector) {

  // construct DetID from info using else the first of all entities and return Ring
  unsigned int ladder = 0;

  PXBDetId id(layer+1,ladder+1,detector+1);

  return getTrackerRing(DetId(id.rawId()));
}

Ring* Rings::getTrackerPXFRing(unsigned int fw_bw,
			       unsigned int disk,
			       unsigned int detector) {

  // construct DetID from info using else the first of all entities and return Ring
  unsigned int blade = 0;

  PXFDetId id(fw_bw+1,disk+1,blade+1,detector+1);

  return getTrackerRing(DetId(id.rawId()));
}

void Rings::fixIndexNumberingScheme() {

  unsigned int counter = 0;

  for ( iterator ring = rings_.begin(); ring != rings_.end(); ++ring ) {
    
    unsigned int index = (*ring).getindex();

    if ( index == 29 ) (*ring).setindex(24);
    if ( index == 26 ) (*ring).setindex(25);
    if ( index == 28 ) (*ring).setindex(26);
    if ( index == 25 ) (*ring).setindex(27);
    if ( index == 27 ) (*ring).setindex(28);
    if ( index == 24 ) (*ring).setindex(29);
    if ( index == 33 ) (*ring).setindex(30);
    if ( index == 30 ) (*ring).setindex(31);
    if ( index == 34 ) (*ring).setindex(32);
    if ( index == 31 ) (*ring).setindex(33);
    if ( index == 35 ) (*ring).setindex(34);
    if ( index == 32 ) (*ring).setindex(35);
    if ( index == 38 ) (*ring).setindex(36);
    if ( index == 41 ) (*ring).setindex(37);
    if ( index == 37 ) (*ring).setindex(38);
    if ( index == 40 ) (*ring).setindex(39);
    if ( index == 36 ) (*ring).setindex(40);
    if ( index == 39 ) (*ring).setindex(41);
    if ( index == 42 ) (*ring).setindex(42);
    if ( index == 45 ) (*ring).setindex(43);
    if ( index == 43 ) (*ring).setindex(44);
    if ( index == 46 ) (*ring).setindex(45);
    if ( index == 44 ) (*ring).setindex(46);
    if ( index == 47 ) (*ring).setindex(47);
    if ( index == 53 ) (*ring).setindex(48);
    if ( index == 50 ) (*ring).setindex(49);
    if ( index == 52 ) (*ring).setindex(50);
    if ( index == 49 ) (*ring).setindex(51);
    if ( index == 51 ) (*ring).setindex(52);
    if ( index == 48 ) (*ring).setindex(53);
    if ( index == 57 ) (*ring).setindex(54);
    if ( index == 54 ) (*ring).setindex(55);
    if ( index == 58 ) (*ring).setindex(56);
    if ( index == 55 ) (*ring).setindex(57);
    if ( index == 59 ) (*ring).setindex(58);
    if ( index == 56 ) (*ring).setindex(59);
    if ( index == 62 ) (*ring).setindex(60);
    if ( index == 65 ) (*ring).setindex(61);
    if ( index == 61 ) (*ring).setindex(62);
    if ( index == 64 ) (*ring).setindex(63);
    if ( index == 60 ) (*ring).setindex(64);
    if ( index == 63 ) (*ring).setindex(65);
    if ( index == 66 ) (*ring).setindex(66);
    if ( index == 69 ) (*ring).setindex(67);
    if ( index == 67 ) (*ring).setindex(68);
    if ( index == 70 ) (*ring).setindex(69);
    if ( index == 68 ) (*ring).setindex(70);
    if ( index == 71 ) (*ring).setindex(71);
    if ( index == 77 ) (*ring).setindex(72);
    if ( index == 76 ) (*ring).setindex(73);
    if ( index == 75 ) (*ring).setindex(74);
    if ( index == 74 ) (*ring).setindex(75);
    if ( index == 73 ) (*ring).setindex(76);
    if ( index == 72 ) (*ring).setindex(77);
    if ( index == 78 ) (*ring).setindex(78);
    if ( index == 79 ) (*ring).setindex(79);
    if ( index == 80 ) (*ring).setindex(80);
    if ( index == 81 ) (*ring).setindex(81);
    if ( index == 82 ) (*ring).setindex(82);
    if ( index == 83 ) (*ring).setindex(83);
    if ( index == 89 ) (*ring).setindex(84);
    if ( index == 88 ) (*ring).setindex(85);
    if ( index == 87 ) (*ring).setindex(86);
    if ( index == 86 ) (*ring).setindex(87);
    if ( index == 85 ) (*ring).setindex(88);
    if ( index == 84 ) (*ring).setindex(89);
    if ( index == 90 ) (*ring).setindex(90);
    if ( index == 91 ) (*ring).setindex(91);
    if ( index == 92 ) (*ring).setindex(92);
    if ( index == 93 ) (*ring).setindex(93);
    if ( index == 94 ) (*ring).setindex(94);
    if ( index == 95 ) (*ring).setindex(95);
    if ( index == 101 ) (*ring).setindex(96);
    if ( index == 100 ) (*ring).setindex(97);
    if ( index == 99 ) (*ring).setindex(98);
    if ( index == 98 ) (*ring).setindex(99);
    if ( index == 97 ) (*ring).setindex(100);
    if ( index == 96 ) (*ring).setindex(101);
    if ( index == 102 ) (*ring).setindex(102);
    if ( index == 103 ) (*ring).setindex(103);
    if ( index == 104 ) (*ring).setindex(104);
    if ( index == 105 ) (*ring).setindex(105);
    if ( index == 106 ) (*ring).setindex(106);
    if ( index == 107 ) (*ring).setindex(107);
    if ( index == 113 ) (*ring).setindex(108);
    if ( index == 112 ) (*ring).setindex(109);
    if ( index == 111 ) (*ring).setindex(110);
    if ( index == 110 ) (*ring).setindex(111);
    if ( index == 109 ) (*ring).setindex(112);
    if ( index == 108 ) (*ring).setindex(113);
    if ( index == 114 ) (*ring).setindex(114);
    if ( index == 115 ) (*ring).setindex(115);
    if ( index == 116 ) (*ring).setindex(116);
    if ( index == 117 ) (*ring).setindex(117);
    if ( index == 118 ) (*ring).setindex(118);
    if ( index == 119 ) (*ring).setindex(119);
    if ( index == 125 ) (*ring).setindex(120);
    if ( index == 124 ) (*ring).setindex(121);
    if ( index == 123 ) (*ring).setindex(122);
    if ( index == 122 ) (*ring).setindex(123);
    if ( index == 121 ) (*ring).setindex(124);
    if ( index == 120 ) (*ring).setindex(125);
    if ( index == 126 ) (*ring).setindex(126);
    if ( index == 127 ) (*ring).setindex(127);
    if ( index == 128 ) (*ring).setindex(128);
    if ( index == 129 ) (*ring).setindex(129);
    if ( index == 130 ) (*ring).setindex(130);
    if ( index == 131 ) (*ring).setindex(131);
    if ( index == 137 ) (*ring).setindex(132);
    if ( index == 136 ) (*ring).setindex(133);
    if ( index == 135 ) (*ring).setindex(134);
    if ( index == 134 ) (*ring).setindex(135);
    if ( index == 133 ) (*ring).setindex(136);
    if ( index == 132 ) (*ring).setindex(137);
    if ( index == 138 ) (*ring).setindex(138);
    if ( index == 139 ) (*ring).setindex(139);
    if ( index == 140 ) (*ring).setindex(140);
    if ( index == 141 ) (*ring).setindex(141);
    if ( index == 142 ) (*ring).setindex(142);
    if ( index == 143 ) (*ring).setindex(143);
    if ( index == 0 ) (*ring).setindex(0);
    if ( index == 1 ) (*ring).setindex(1);
    if ( index == 2 ) (*ring).setindex(2);
    if ( index == 3 ) (*ring).setindex(3);
    if ( index == 4 ) (*ring).setindex(4);
    if ( index == 5 ) (*ring).setindex(5);
    if ( index == 6 ) (*ring).setindex(6);
    if ( index == 7 ) (*ring).setindex(7);
    if ( index == 8 ) (*ring).setindex(8);
    if ( index == 9 ) (*ring).setindex(9);
    if ( index == 10 ) (*ring).setindex(10);
    if ( index == 11 ) (*ring).setindex(11);
    if ( index == 12 ) (*ring).setindex(12);
    if ( index == 13 ) (*ring).setindex(13);
    if ( index == 14 ) (*ring).setindex(14);
    if ( index == 15 ) (*ring).setindex(15);
    if ( index == 16 ) (*ring).setindex(16);
    if ( index == 17 ) (*ring).setindex(17);
    if ( index == 18 ) (*ring).setindex(18);
    if ( index == 19 ) (*ring).setindex(19);
    if ( index == 20 ) (*ring).setindex(20);
    if ( index == 21 ) (*ring).setindex(21);
    if ( index == 22 ) (*ring).setindex(22);
    if ( index == 23 ) (*ring).setindex(23);
    if ( index == 203 ) (*ring).setindex(197);
    if ( index == 204 ) (*ring).setindex(198);
    if ( index == 205 ) (*ring).setindex(199);
    if ( index == 200 ) (*ring).setindex(200);
    if ( index == 201 ) (*ring).setindex(201);
    if ( index == 202 ) (*ring).setindex(202);
    if ( index == 197 ) (*ring).setindex(203);
    if ( index == 198 ) (*ring).setindex(204);
    if ( index == 199 ) (*ring).setindex(205);
    if ( index == 234 ) (*ring).setindex(234);
    if ( index == 235 ) (*ring).setindex(235);
    if ( index == 236 ) (*ring).setindex(236);
    if ( index == 237 ) (*ring).setindex(237);
    if ( index == 238 ) (*ring).setindex(238);
    if ( index == 239 ) (*ring).setindex(239);
    if ( index == 240 ) (*ring).setindex(240);
    if ( index == 241 ) (*ring).setindex(241);
    if ( index == 242 ) (*ring).setindex(242);
    if ( index == 193 ) (*ring).setindex(144);
    if ( index == 194 ) (*ring).setindex(145);
    if ( index == 195 ) (*ring).setindex(146);
    if ( index == 196 ) (*ring).setindex(147);
    if ( index == 188 ) (*ring).setindex(148);
    if ( index == 189 ) (*ring).setindex(149);
    if ( index == 190 ) (*ring).setindex(150);
    if ( index == 191 ) (*ring).setindex(151);
    if ( index == 192 ) (*ring).setindex(152);
    if ( index == 183 ) (*ring).setindex(153);
    if ( index == 184 ) (*ring).setindex(154);
    if ( index == 185 ) (*ring).setindex(155);
    if ( index == 186 ) (*ring).setindex(156);
    if ( index == 187 ) (*ring).setindex(157);
    if ( index == 177 ) (*ring).setindex(158);
    if ( index == 178 ) (*ring).setindex(159);
    if ( index == 179 ) (*ring).setindex(160);
    if ( index == 180 ) (*ring).setindex(161);
    if ( index == 181 ) (*ring).setindex(162);
    if ( index == 182 ) (*ring).setindex(163);
    if ( index == 171 ) (*ring).setindex(164);
    if ( index == 172 ) (*ring).setindex(165);
    if ( index == 173 ) (*ring).setindex(166);
    if ( index == 174 ) (*ring).setindex(167);
    if ( index == 175 ) (*ring).setindex(168);
    if ( index == 176 ) (*ring).setindex(169);
    if ( index == 165 ) (*ring).setindex(170);
    if ( index == 166 ) (*ring).setindex(171);
    if ( index == 167 ) (*ring).setindex(172);
    if ( index == 168 ) (*ring).setindex(173);
    if ( index == 169 ) (*ring).setindex(174);
    if ( index == 170 ) (*ring).setindex(175);
    if ( index == 158 ) (*ring).setindex(176);
    if ( index == 159 ) (*ring).setindex(177);
    if ( index == 160 ) (*ring).setindex(178);
    if ( index == 161 ) (*ring).setindex(179);
    if ( index == 162 ) (*ring).setindex(180);
    if ( index == 163 ) (*ring).setindex(181);
    if ( index == 164 ) (*ring).setindex(182);
    if ( index == 151 ) (*ring).setindex(183);
    if ( index == 152 ) (*ring).setindex(184);
    if ( index == 153 ) (*ring).setindex(185);
    if ( index == 154 ) (*ring).setindex(186);
    if ( index == 155 ) (*ring).setindex(187);
    if ( index == 156 ) (*ring).setindex(188);
    if ( index == 157 ) (*ring).setindex(189);
    if ( index == 144 ) (*ring).setindex(190);
    if ( index == 145 ) (*ring).setindex(191);
    if ( index == 146 ) (*ring).setindex(192);
    if ( index == 147 ) (*ring).setindex(193);
    if ( index == 148 ) (*ring).setindex(194);
    if ( index == 149 ) (*ring).setindex(195);
    if ( index == 150 ) (*ring).setindex(196);
    if ( index == 243 ) (*ring).setindex(243);
    if ( index == 244 ) (*ring).setindex(244);
    if ( index == 245 ) (*ring).setindex(245);
    if ( index == 246 ) (*ring).setindex(246);
    if ( index == 247 ) (*ring).setindex(247);
    if ( index == 248 ) (*ring).setindex(248);
    if ( index == 249 ) (*ring).setindex(249);
    if ( index == 250 ) (*ring).setindex(250);
    if ( index == 251 ) (*ring).setindex(251);
    if ( index == 252 ) (*ring).setindex(252);
    if ( index == 253 ) (*ring).setindex(253);
    if ( index == 254 ) (*ring).setindex(254);
    if ( index == 255 ) (*ring).setindex(255);
    if ( index == 256 ) (*ring).setindex(256);
    if ( index == 257 ) (*ring).setindex(257);
    if ( index == 258 ) (*ring).setindex(258);
    if ( index == 259 ) (*ring).setindex(259);
    if ( index == 260 ) (*ring).setindex(260);
    if ( index == 261 ) (*ring).setindex(261);
    if ( index == 262 ) (*ring).setindex(262);
    if ( index == 263 ) (*ring).setindex(263);
    if ( index == 264 ) (*ring).setindex(264);
    if ( index == 265 ) (*ring).setindex(265);
    if ( index == 266 ) (*ring).setindex(266);
    if ( index == 267 ) (*ring).setindex(267);
    if ( index == 268 ) (*ring).setindex(268);
    if ( index == 269 ) (*ring).setindex(269);
    if ( index == 270 ) (*ring).setindex(270);
    if ( index == 271 ) (*ring).setindex(271);
    if ( index == 272 ) (*ring).setindex(272);
    if ( index == 273 ) (*ring).setindex(273);
    if ( index == 274 ) (*ring).setindex(274);
    if ( index == 275 ) (*ring).setindex(275);
    if ( index == 276 ) (*ring).setindex(276);
    if ( index == 277 ) (*ring).setindex(277);
    if ( index == 278 ) (*ring).setindex(278);
    if ( index == 279 ) (*ring).setindex(279);
    if ( index == 280 ) (*ring).setindex(280);
    if ( index == 281 ) (*ring).setindex(281);
    if ( index == 282 ) (*ring).setindex(282);
    if ( index == 283 ) (*ring).setindex(283);
    if ( index == 284 ) (*ring).setindex(284);
    if ( index == 285 ) (*ring).setindex(285);
    if ( index == 286 ) (*ring).setindex(286);
    if ( index == 287 ) (*ring).setindex(287);
    if ( index == 288 ) (*ring).setindex(288);
    if ( index == 289 ) (*ring).setindex(289);
    if ( index == 290 ) (*ring).setindex(290);
    if ( index == 291 ) (*ring).setindex(291);
    if ( index == 292 ) (*ring).setindex(292);
    if ( index == 293 ) (*ring).setindex(293);
    if ( index == 294 ) (*ring).setindex(294);
    if ( index == 295 ) (*ring).setindex(295);

    ++counter;

  }

  if ( verbosity_ > 1 ) {
    std::cout << "[Rings] fixed the index numbering scheme for " << counter << " rings" << std::endl; 
  }

}

double Rings::determineExtensions(const TrackingGeometry &tracker, DetId id, float &rmin, float &rmax, float &zmin, float& zmax, Ring::type type) {

  const GeomDetUnit *det = tracker.idToDet(id);
	
  double phi = 0.;

  if ( det != 0 ) {

    GlobalPoint p[8];
    float r[8],z[8];
    float local_rmin = 1200.;
    float local_rmax = 0.;
    float local_zmin = 2800.;
    float local_zmax = -2800.;
    
    // calculate global position of center
    GlobalPoint center = det->surface().toGlobal(LocalPoint(0,0,0)); 
    phi = center.phi();

    // convert to 0 to 2pi
    double pi = 3.14159265358979312;
    if ( phi < 0 ) phi = 2*pi + phi;

    if ( (type == Ring::TIBRing) || (type == Ring::TOBRing) || (type == Ring::PXBRing) || (type == Ring::PXFRing) ) {

      float length = det->surface().bounds().length();
      float width = det->surface().bounds().width();
      float thickness = det->surface().bounds().thickness();
	  
      p[0] = det->surface().toGlobal(LocalPoint(width/2,length/2,thickness/2)); 
      p[1] = det->surface().toGlobal(LocalPoint(width/2,-length/2,thickness/2)); 
      p[2] = det->surface().toGlobal(LocalPoint(-width/2,length/2,thickness/2)); 
      p[3] = det->surface().toGlobal(LocalPoint(-width/2,-length/2,thickness/2)); 
      p[4] = det->surface().toGlobal(LocalPoint(width/2,length/2,-thickness/2)); 
      p[5] = det->surface().toGlobal(LocalPoint(width/2,-length/2,-thickness/2)); 
      p[6] = det->surface().toGlobal(LocalPoint(-width/2,length/2,-thickness/2)); 
      p[7] = det->surface().toGlobal(LocalPoint(-width/2,-length/2,-thickness/2)); 
      
    } else if ( (type == Ring::TIDRing) || (type == Ring::TECRing) ) {
      
      std::vector<float> parameters = ((TrapezoidalPlaneBounds&)(det->surface().bounds())).parameters();
      
      p[0] = det->surface().toGlobal(LocalPoint(parameters[0],-parameters[3],parameters[2])); 
      p[1] = det->surface().toGlobal(LocalPoint(-parameters[0],-parameters[3],parameters[2])); 
      p[2] = det->surface().toGlobal(LocalPoint(parameters[1],parameters[3],parameters[2])); 
      p[3] = det->surface().toGlobal(LocalPoint(-parameters[1],parameters[3],parameters[2])); 
      p[4] = det->surface().toGlobal(LocalPoint(parameters[0],-parameters[3],-parameters[2])); 
      p[5] = det->surface().toGlobal(LocalPoint(-parameters[0],-parameters[3],-parameters[2])); 
      p[6] = det->surface().toGlobal(LocalPoint(parameters[1],parameters[3],-parameters[2])); 
      p[7] = det->surface().toGlobal(LocalPoint(-parameters[1],parameters[3],-parameters[2])); 

    } 
    
    for ( int i = 0; i < 8; ++i ) {
      r[i] = sqrt(p[i].x()*p[i].x() + p[i].y()*p[i].y());
      z[i] = p[i].z();
      if ( r[i] < local_rmin ) local_rmin = r[i];
      if ( r[i] > local_rmax ) local_rmax = r[i];
      if ( z[i] < local_zmin ) local_zmin = z[i];
      if ( z[i] > local_zmax ) local_zmax = z[i];
      }
    
    if ( local_rmin < rmin ) rmin = local_rmin;
    if ( local_rmax > rmax ) rmax = local_rmax;
    if ( local_zmin < zmin ) zmin = local_zmin;
    if ( local_zmax > zmax ) zmax = local_zmax;
    
  } else {
    
    if ( type == Ring::TIBRing ) {
      TIBDetId tibid(id.rawId());
      std::cout << "[Rings] problem resolving DetUnit for TIB ring Detid: " << id.rawId() 
		<< " layer: " << tibid.layer() 
		<< " fw(0)/bw(1): " << tibid.string()[0]
		<< " ext(0)/int(0): " << tibid.string()[1] 
		<< " string: " << tibid.string()[2] 
		<< " detector: " << tibid.det()
		<< " not stereo(0)/stereo(1): " << tibid.stereo() 
		<< " not glued(0)/glued(1): " << tibid.glued() 
		<< std::endl; 
    } else if ( type == Ring::TOBRing ) {
      TOBDetId tobid(id.rawId()); 
      std::cout << "[Rings] problem resolving DetUnit for TOB ring Detid: " << id.rawId() 
		<< " layer: " << tobid.layer() 
		<< " fw(0)/bw(1): " << tobid.rod()[0]
		<< " rod: " << tobid.rod()[1] 
		<< " detector: " << tobid.det()
		<< " not stereo(0)/stereo(1): " << tobid.stereo() 
		<< " not glued(0)/glued(1): " << tobid.glued() 
		<< std::endl; 
    } else if ( type == Ring::TIDRing ) {
      TIDDetId tidid(id.rawId()); 
      std::cout << "[Rings] problem resolving DetUnit for TID ring Detid: " << id.rawId() 
		<< " side neg(1)/pos(2): " << tidid.side() 
		<< " wheel: " << tidid.wheel()
		<< " ring: " << tidid.ring()
		<< " detector fw(0)/bw(1): " << tidid.det()[0]
		<< " detector: " << tidid.det()[1] 
		<< " not stereo(0)/stereo(1): " << tidid.stereo() 
		<< " not glued(0)/glued(1): " << tidid.glued() 
		<< std::endl; 
    } else if ( type == Ring::TECRing ) {
      TECDetId tecid(id.rawId()); 
      std::cout << "[Rings] problem resolving DetUnit for TEC ring DetId: " << id.rawId() 
		<< " side neg(1)/pos(2): " << tecid.side() 
		<< " wheel: " << tecid.wheel()
		<< " petal fw(0)/bw(1): " << tecid.petal()[0]
		<< " petal: " << tecid.petal()[1] 
		<< " ring: " << tecid.ring()
		<< " detector fw(0)/bw(1): " << tecid.det()[0]
		<< " detector: " << tecid.det()[1] 
		<< " not stereo(0)/stereo(1): " << tecid.stereo() 
		<< " not glued(0)/glued(1): " << tecid.glued() 
		<< std::endl; 
    } else if ( type == Ring::PXBRing ) {
      PXBDetId pxbid(id.rawId()); 
      std::cout << "[Rings] problem resolving DetUnit for PXB ring DetId: " << id.rawId() 
		<< " layer: " << pxbid.layer()
		<< " ladder: " << pxbid.ladder()
		<< " detector: " << pxbid.det()
		<< std::endl; 
    } else if ( type == Ring::PXFRing ) {
      PXFDetId pxfid(id.rawId()); 
      std::cout << "[Rings] problem resolving DetUnit for PXF ring DetId: " << id.rawId() 
		<< " side: " << pxfid.side()
		<< " disk: " << pxfid.disk()
		<< " blade: " << pxfid.blade()
		<< " detector: " << pxfid.det()
		<< std::endl; 
    }
  }

  return phi;

}

std::vector<unsigned int> Rings::dumpOldStyle(std::string ascii_filename, bool writeFile) {
  // defined order: TIB, TOB, TID, TEC, PXB, PXF
  // change if road is changed

  unsigned int layersTIB = 0;
  unsigned int layersTOB = 0;
  unsigned int layersTID = 0;
  unsigned int layersTEC = 0;
  unsigned int layersPXB = 0;
  unsigned int layersPXF = 0;

  std::string tib = dumpOldStyleTIB(layersTIB);
  std::string tob = dumpOldStyleTOB(layersTOB);
  std::string tid = dumpOldStyleTID(layersTID);
  std::string tec = dumpOldStyleTEC(layersTEC);
  std::string pxb = dumpOldStylePXB(layersPXB);
//   std::string pxf = dumpOldStylePXF(layersPXF);

  unsigned int nLayers = layersTIB + layersTOB +
    layersTID + layersTEC +
    layersPXB + layersPXF;

  if ( writeFile ) {

    std::ofstream stream(ascii_filename.c_str());

    stream << nLayers << std::endl;

    stream << tib;
    stream << tob;
    stream << tid;
    stream << tec;
    stream << pxb;
    //   stream << pxf;

    if ( verbosity_ > 0 ) {
      std::cout << "[Rings] wrote out rings for " << nLayers << " layers in old style." << std::endl; 
    }
  }

  std::vector<unsigned int> layers;
  layers.push_back(layersTIB);
  layers.push_back(layersTOB);
  layers.push_back(layersTID);
  layers.push_back(layersTEC);
  layers.push_back(layersPXB);
  layers.push_back(layersPXF);

  return layers;

}

std::string Rings::dumpOldStyleTIB(unsigned int &nLayers) {

  std::ostringstream stream;

  unsigned int layer_max   = 4;
  unsigned int fw_bw_max   = 2;
  unsigned int ext_int_max = 2;
  unsigned int detector_max  = 3;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    ++nLayers;
    std::ostringstream tempstream;
    unsigned int nRings = 0;
    for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = 0; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	  ++nRings;
	  Ring *ring = getTrackerTIBRing(layer,fw_bw,ext_int,detector);
	  tempstream << ring->getrmin() << " "
		     << ring->getrmax() << " "
		     << ring->getzmin() << " "
		     << ring->getzmax() << " " << std::endl;
	}    
      }    
    }
    stream << "0 0.0" << std::endl;
    stream << nRings << std::endl;
    stream << tempstream.str();

    if ( verbosity_ > 1 ) {
      std::cout << "[Rings] wrote out " << nRings << " TIB rings in old style." << std::endl; 
    }

  }

  return stream.str();

}

std::string Rings::dumpOldStyleTOB(unsigned int &nLayers) {

  std::ostringstream stream;

  unsigned int layer_max       = 6;
  unsigned int rod_fw_bw_max   = 2;
  unsigned int detector_max      = 6;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    ++nLayers;
    std::ostringstream tempstream;
    unsigned int nRings = 0;
    for ( unsigned int rod_fw_bw = 0; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	++nRings;
	Ring *ring = getTrackerTOBRing(layer,rod_fw_bw,detector);
	tempstream << ring->getrmin() << " "
		   << ring->getrmax() << " "
		   << ring->getzmin() << " "
		   << ring->getzmax() << " " << std::endl;
      }    
    }
    stream << "0 0.0" << std::endl;
    stream << nRings << std::endl;
    stream << tempstream.str();

    if ( verbosity_ > 1 ) {
      std::cout << "[Rings] wrote out " << nRings << " TOB rings in old style." << std::endl; 
    }
    
  }
  
  return stream.str();
  
}

std::string Rings::dumpOldStyleTID(unsigned int &nLayers) {

  std::ostringstream stream;

  unsigned int fw_bw_max       = 2;
  unsigned int wheel_max       = 3;
  unsigned int ring_max        = 3;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      ++nLayers;
      std::ostringstream tempstream;
      unsigned int nRings = 0;
      for ( unsigned int ring = 0; ring < ring_max; ++ring ) {
	++nRings;
	Ring *tempring = getTrackerTIDRing(fw_bw,wheel,ring);
	tempstream << tempring->getrmin() << " "
		   << tempring->getrmax() << " "
		   << tempring->getzmin() << " "
		   << tempring->getzmax() << " " << std::endl;
      }    
      stream << "0 0.0" << std::endl;
      stream << nRings << std::endl;
      stream << tempstream.str();
      
      if ( verbosity_ > 1 ) {
	std::cout << "[Rings] wrote out " << nRings << " TID rings in old style." << std::endl; 
      }
    
    }
  }
  
  return stream.str();
  
}

std::string Rings::dumpOldStyleTEC(unsigned int &nLayers) {

  std::ostringstream stream;

  unsigned int fw_bw_max       = 2;
  unsigned int wheel_max       = 9;
  unsigned int ring_max[9];

  // TEC WHEEL 1
  ring_max[0] = 7;
  // TEC WHEEL 2
  ring_max[1] = 7;
  // TEC WHEEL 3
  ring_max[2] = 7;
  // TEC WHEEL 4
  ring_max[3] = 6;
  // TEC WHEEL 5
  ring_max[4] = 6;
  // TEC WHEEL 6
  ring_max[5] = 6;
  // TEC WHEEL 7
  ring_max[6] = 5;
  // TEC WHEEL 8
  ring_max[7] = 5;
  // TEC WHEEL 9
  ring_max[8] = 4;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      ++nLayers;
      std::ostringstream tempstream;
      unsigned int nRings = 0;
      for ( unsigned int ring = 0; ring < ring_max[wheel]; ++ring ) {
	++nRings;
	Ring *tempring = getTrackerTECRing(fw_bw,wheel,ring);
	tempstream << tempring->getrmin() << " "
		   << tempring->getrmax() << " "
		   << tempring->getzmin() << " "
		   << tempring->getzmax() << " " << std::endl;
      }    
      stream << "0 0.0" << std::endl;
      stream << nRings << std::endl;
      stream << tempstream.str();

      if ( verbosity_ > 1 ) {
	std::cout << "[Rings] wrote out " << nRings << " TEC rings in old style." << std::endl; 
      }
    
    }
  }
  
  return stream.str();
  
}

std::string Rings::dumpOldStylePXB(unsigned int &nLayers) {

  std::ostringstream stream;

  unsigned int layer_max   = 3;
  unsigned int detector_max  = 8;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    ++nLayers;
    std::ostringstream tempstream;
    unsigned int nRings = 0;
    for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
      ++nRings;
      Ring *ring = getTrackerPXBRing(layer,detector);
      tempstream << ring->getrmin() << " "
		 << ring->getrmax() << " "
		 << ring->getzmin() << " "
		 << ring->getzmax() << " " << std::endl;
    }
    stream << "0 0.0" << std::endl;
    stream << nRings << std::endl;
    stream << tempstream.str();

    if ( verbosity_ > 1 ) {
      std::cout << "[Rings] wrote out " << nRings << " PXB rings in old style." << std::endl; 
    }
    
  }
  
  return stream.str();
  
}

std::string Rings::dumpOldStylePXF(unsigned int &nLayers) {

  std::ostringstream stream;

  unsigned int fw_bw_max   = 2;
  unsigned int disk_max    = 2;
  unsigned int detector_max  = 7;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int disk = 0; disk < disk_max; ++disk ) {
      ++nLayers;
      std::ostringstream tempstream;
      unsigned int nRings = 0;
      for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	++nRings;
	Ring *ring = getTrackerPXFRing(fw_bw,disk,detector);
	tempstream << ring->getrmin() << " "
		   << ring->getrmax() << " "
		   << ring->getzmin() << " "
		   << ring->getzmax() << " " << std::endl;
      }
      stream << "0 0.0" << std::endl;
      stream << nRings << std::endl;
      stream << tempstream.str();

      if ( verbosity_ > 1 ) {
	std::cout << "[Rings] wrote out " << nRings << " PXF rings in old style." << std::endl; 
      }
    
    }
  }
  
  return stream.str();
  
}

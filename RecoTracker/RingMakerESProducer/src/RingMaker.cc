//
// Package:         RecoTracker/RingMakerESProducer
// Class:           RingMaker
// 
// Description:     The RingMaker object creates and povides
//                  all Rings in the detector.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Tue Oct  3 23:51:34 UTC 2006
//
// $Author: burkett $
// $Date: 2008/03/19 15:54:19 $
// $Revision: 1.5 $
//

#include <iostream>
#include <algorithm>
#include <map>
#include <utility>
#include <fstream>
#include <sstream>

#include "RecoTracker/RingMakerESProducer/interface/RingMaker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/Pi.h"

RingMaker::RingMaker(const TrackerGeometry *tracker,
		     std::string configuration) 
  : tracker_(tracker), configuration_(configuration)
{

  rings_ = new Rings();

  fillTIBGeometryArray();
  fillTOBGeometryArray();
  fillTIDGeometryArray();
  fillTECGeometryArray();
  fillPXBGeometryArray();
  fillPXFGeometryArray();

  constructRings();
  
}

RingMaker::~RingMaker() { 

}  

void RingMaker::constructRings() {

  unsigned int index = 0;

  if ( configuration_ == "TIF") {
    index = 52;
    constructTIBRings(index);
    constructTIDRings(index);
    constructTOBRings(index);
    constructTECRings(index);
  } else if ( configuration_ == "TIFTOB") {
    index = 118;
    constructTOBRings(index);
  } else if ( configuration_ == "TIFTIB") {
    index = 52;
    constructTIBRings(index);
    constructTIDRings(index);
  } else if ( configuration_ == "TIFTIBTOB") {
    index = 52;
    constructTIBRings(index);
    constructTIDRings(index);
    index = 118;
    constructTOBRings(index);
 } else if ( configuration_ == "TIFTOBTEC") {
    index = 118;
    constructTOBRings(index);
    index = 190;
    constructTECRings(index);
  } else {
    constructPXBRings(index);
    constructPXFRings(index);
    constructTIBRings(index);
    constructTIDRings(index);
    constructTOBRings(index);
    constructTECRings(index);
  }
  
  edm::LogInfo("RoadSearch") << "Constructed " << index << " rings."; 

}

void RingMaker::constructTIBRings(unsigned int &index) {

  unsigned int counter = 0;

  unsigned int layer_max   = 5;
  unsigned int fw_bw_max   = 3;
  unsigned int ext_int_max = 3;
  unsigned int detector_max  = 4;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
      for ( unsigned int ext_int = 0; ext_int < ext_int_max; ++ext_int ) {
	for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
	  Ring ring = constructTIBRing(layer,fw_bw,ext_int,detector);
	  if ( ring.getNumDetIds() > 0 ) {
	    ring.setindex(++index);
	    double center_z = ring.getzmin() + ((ring.getzmax()-ring.getzmin())/2);
	    rings_->insert(center_z,ring);
	    ++counter;
	    LogDebug("RoadSearch") << "constructed TIB ring with index: " << ring.getindex() 
				   << " consisting of " << ring.getNumDetIds() << " DetIds"; 
	  }
	}    
      }    
    }    
  }
  LogDebug("RoadSearch") << "constructed " << counter << " TIB rings"; 
  
}

Ring RingMaker::constructTIBRing(unsigned int layer,
				 unsigned int fw_bw,
				 unsigned int ext_int,
				 unsigned int detector) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int string_max = 57;

  Ring ring(Ring::TIBRing);

  for ( unsigned int string = 0; string < string_max; ++string) {
    // only fill r/phi sensor id's
    // first try r/phi of double sided layer (stereo = 2)
    // then try r/phi of single sided layer (stereo = 0)
    if ( tib_[layer][fw_bw][ext_int][string][detector][2] > 0 ) {
      DetId id = constructTIBDetId(layer,fw_bw,ext_int,string,detector,2);
      double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TIBRing);
      ring.addId(phi,id);
    } else if ( tib_[layer][fw_bw][ext_int][string][detector][0] > 0 ) {
      DetId id = constructTIBDetId(layer,fw_bw,ext_int,string,detector,0);
      double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TIBRing);
      ring.addId(phi,id);
    }
  }

  LogDebug("RoadSearch") << "TIB ring initialized rmin/rmax/zmin/zmax: " << rmin 
			 << "/" << rmax 
			 << "/" << zmin 
			 << "/" << zmax;
    
  ring.initialize(rmin,rmax,zmin,zmax);

  return ring;
}

DetId RingMaker::constructTIBDetId(unsigned int layer,
				   unsigned int fw_bw,
				   unsigned int ext_int,
				   unsigned int string,
				   unsigned int detector,
				   unsigned int stereo) {

  TIBDetId id(layer+1,fw_bw,ext_int,string+1,detector+1,stereo);
  LogDebug("RoadSearch") << "constructed TIB ring DetId for layer: " << id.layer() 
			 << " fw(0)/bw(1): " << id.string()[0]
			 << " ext(0)/int(1): " << id.string()[1] 
			 << " string: " << id.string()[2] 
			 << " module: " << id.module()
			 << " stereo: " << id.stereo(); 

  return DetId(id.rawId());
}


void RingMaker::constructTOBRings(unsigned int &index) {

  unsigned int counter = 0;
 
  unsigned int layer_max       = 7;
  unsigned int rod_fw_bw_max   = 3;
  unsigned int module_max      = 7; 

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    for ( unsigned int rod_fw_bw = 0; rod_fw_bw < rod_fw_bw_max; ++rod_fw_bw ) {
      for ( unsigned int module = 0; module < module_max; ++module ) {
	Ring ring = constructTOBRing(layer,rod_fw_bw,module);
	if ( ring.getNumDetIds() > 0 ) {
	  ring.setindex(++index);
	  double center_z = ring.getzmin() + ((ring.getzmax()-ring.getzmin())/2);
	  rings_->insert(center_z,ring);
	  ++counter;
	  LogDebug("RoadSearch") << "constructed TOB ring with index: " << ring.getindex() 
				 << " consisting of " << ring.getNumDetIds() << " DetIds"; 
	}
      }    
    }    
  }

  LogDebug("RoadSearch") << "constructed " << counter << " TOB rings"; 

}

Ring RingMaker::constructTOBRing(unsigned int layer,
				 unsigned int rod_fw_bw,
				 unsigned int module) {
  
  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int rod_max = 75;
  Ring ring(Ring::TOBRing);

  for ( unsigned int rod = 0; rod < rod_max; ++rod ) {
    // only fill r/phi sensor id's
    // first try r/phi of double sided layer (stereo = 2)
    // then try r/phi of single sided layer (stereo = 0)
    if ( tob_[layer][rod_fw_bw][rod][module][2] > 0 ) {
      DetId id = constructTOBDetId(layer,rod_fw_bw,rod,module,2);
      double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TOBRing);
      ring.addId(phi,id);
    } else if ( tob_[layer][rod_fw_bw][rod][module][0] > 0 ) {
      DetId id = constructTOBDetId(layer,rod_fw_bw,rod,module,0);
      double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TOBRing);
      ring.addId(phi,id);
    }
  }
  
  
  LogDebug("RoadSearch") << "TOB ring initialized rmin/rmax/zmin/zmax: " << rmin 
			 << "/" << rmax 
			 << "/" << zmin 
			 << "/" << zmax;
    
  ring.initialize(rmin,rmax,zmin,zmax);
  
  return ring;
}

DetId RingMaker::constructTOBDetId(unsigned int layer,
				   unsigned int rod_fw_bw,
				   unsigned int rod,
				   unsigned int detector,
				   unsigned int stereo) {

  TOBDetId id(layer+1,rod_fw_bw,rod+1,detector+1,stereo);

  LogDebug("RoadSearch") << "constructed TOB ring DetId for layer: " << id.layer() 
			 << " rod fw(0)/bw(1): " << id.rod()[0] 
			 << " rod: " << id.rod()[1] 
			 << " module: " << id.module() 
			 << " stereo: " << id.stereo() << std::endl; 

  return DetId(id.rawId());
}

void RingMaker::constructTIDRings(unsigned int &index) {

  unsigned int counter = 0;

  unsigned int fw_bw_max       = 3;
  unsigned int wheel_max       = 4;
  unsigned int ring_max        = 4;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = 0; ring < ring_max; ++ring ) {
	Ring tempring = constructTIDRing(fw_bw,wheel,ring);
	if ( tempring.getNumDetIds() > 0 ) {
	  tempring.setindex(++index);
	  double center_z = tempring.getzmin() + ((tempring.getzmax()-tempring.getzmin())/2);
	  rings_->insert(center_z,tempring);
	  ++counter;
	  LogDebug("RoadSearch") << "constructed TID ring with index: " << tempring.getindex() 
				 << " consisting of " << tempring.getNumDetIds() << " DetIds"; 
	}
      }   
    }  
  }
  
  LogDebug("RoadSearch") << "constructed " << counter << " TID rings"; 

}

Ring RingMaker::constructTIDRing(unsigned int fw_bw,
				 unsigned int wheel,
				 unsigned int ring) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int detector_fw_bw_max   = 3;
  unsigned int detector_max = 21;
  
  Ring tempring(Ring::TIDRing);
  
  for ( unsigned int detector_fw_bw = 0; detector_fw_bw < detector_fw_bw_max; ++detector_fw_bw ) {
    for ( unsigned int detector = 0; detector < detector_max; ++detector ) {
      if ( tid_[fw_bw][wheel][ring][detector_fw_bw][detector][2] > 0 ) {
	DetId id = constructTIDDetId(fw_bw,wheel,ring,detector_fw_bw,detector,2);
	double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TIDRing);
	tempring.addId(phi,id);
      } else if ( tid_[fw_bw][wheel][ring][detector_fw_bw][detector][0] > 0 ) {
	DetId id = constructTIDDetId(fw_bw,wheel,ring,detector_fw_bw,detector,0);
	double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TIDRing);
	tempring.addId(phi,id);
      }
    }
  }
  	
  LogDebug("RoadSearch") << "TID ring initialized rmin/rmax/zmin/zmax: " << rmin 
			 << "/" << rmax 
			 << "/" << zmin 
			 << "/" << zmax;
    
  tempring.initialize(rmin,rmax,zmin,zmax);

  return tempring;
}

DetId RingMaker::constructTIDDetId(unsigned int fw_bw,
				   unsigned int wheel,
				   unsigned int ring,
				   unsigned int module_fw_bw,
				   unsigned int module,
				   unsigned int stereo) {

  TIDDetId id(fw_bw+1,wheel+1,ring+1,module_fw_bw,module+1,stereo);

  LogDebug("RoadSearch") << "constructed TID ring DetId for side: " << id.side() 
			 << " wheel: " << id.wheel() 
			 << " ring: " << id.ring() 
			 << " module_fw_bw: " << id.module()[0] 
			 << " module: " << id.module()[1] 
			 << " stereo: " << id.stereo(); 
	
  return DetId(id.rawId());
}

void RingMaker::constructTECRings(unsigned int &index) {

  unsigned int counter = 0;

  unsigned int fw_bw_max       = 3;
  unsigned int wheel_max       = 10;
  unsigned int ring_max        = 8;

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int wheel = 0; wheel < wheel_max; ++wheel ) {
      for ( unsigned int ring = 0; ring < ring_max; ++ring ) {
	if ( tec2_[fw_bw][wheel][ring] > 0 ) {
	  Ring tempring = constructTECRing(fw_bw,wheel,ring);
	  if ( tempring.getNumDetIds() > 0 ) {
	    tempring.setindex(++index);
	    double center_z = tempring.getzmin() + ((tempring.getzmax()-tempring.getzmin())/2);
	    rings_->insert(center_z,tempring);
	    ++counter;
	    LogDebug("RoadSearch") << "constructed TEC ring with index: " << tempring.getindex() 
				   << " consisting of " << tempring.getNumDetIds() << " DetIds"; 
	  }
	}
      }   
    }  
  }

  LogDebug("RoadSearch") << "constructed " << counter << " TEC rings"; 

}

Ring RingMaker::constructTECRing(unsigned int fw_bw,
				 unsigned int wheel,
				 unsigned int ring) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int petal_max       = 9;
  unsigned int petal_fw_bw_max = 3;
  unsigned int module_max      = 21;

  Ring tempring(Ring::TECRing);
	
  for ( unsigned int petal = 0; petal < petal_max; ++petal ) {
    for ( unsigned int petal_fw_bw = 0; petal_fw_bw < petal_fw_bw_max; ++petal_fw_bw ) {
      for ( unsigned int module = 0; module < module_max; ++module ) {
	// only fill r/phi sensor id's
	// first try r/phi of double sided layer (stereo = 2)
	// then try r/phi of single sided layer (stereo = 0)
	if ( tec_[fw_bw][wheel][petal_fw_bw][petal][ring][module][2] > 0 ) {
	  DetId id = constructTECDetId(fw_bw,wheel,petal_fw_bw,petal,ring,module,2);
	  double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TECRing);
	  tempring.addId(phi,id);
	} else if ( tec_[fw_bw][wheel][petal_fw_bw][petal][ring][module][0] > 0 ) {
	  DetId id = constructTECDetId(fw_bw,wheel,petal_fw_bw,petal,ring,module,0);
	  double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::TECRing);
	  tempring.addId(phi,id);
	}
      }
    }
  }

  LogDebug("RoadSearch") << "TEC ring initialized rmin/rmax/zmin/zmax: " << rmin 
			 << "/" << rmax 
			 << "/" << zmin 
			 << "/" << zmax;
    
  tempring.initialize(rmin,rmax,zmin,zmax);

  return tempring;
}

DetId RingMaker::constructTECDetId(unsigned int fw_bw,
				   unsigned int wheel,
				   unsigned int petal_fw_bw,
				   unsigned int petal,
				   unsigned int ring,
				   unsigned int module,
				   unsigned int stereo) {

  TECDetId id(fw_bw+1,wheel+1,petal_fw_bw,petal+1,ring+1,module+1,stereo);
  
  LogDebug("RoadSearch") << "constructed TEC ring DetId for side: " << id.side() 
			 << " wheel: " << id.wheel() 
			 << " ring: " << id.ring() 
			 << " petal fw(0)/bw(0): " << id.petal()[0] 
			 << " petal: " << id.petal()[1] 
			 << " module: " << id.module() 
			 << " stereo: " << id.stereo(); 

  return DetId(id.rawId());
}

void RingMaker::constructPXBRings(unsigned int &index) {

  unsigned int counter = 0;

  unsigned int layer_max   = 3;
  unsigned int module_max  = 8;

  for ( unsigned int layer = 0; layer < layer_max; ++layer ) {
    for ( unsigned int module = 0; module < module_max; ++module ) {
      Ring ring = constructPXBRing(layer,module);
      if ( ring.getNumDetIds() > 0 ) {
	ring.setindex(++index);
	double center_z = ring.getzmin() + ((ring.getzmax()-ring.getzmin())/2);
	rings_->insert(center_z,ring);
	++counter;
	LogDebug("RoadSearch") << "constructed PXB ring with index: " << ring.getindex() 
			       << " consisting of " << ring.getNumDetIds() << " DetIds"; 
      }
    }    
  }    

  LogDebug("RoadSearch") << "constructed " << counter << " PXB rings"; 
  
}

Ring RingMaker::constructPXBRing(unsigned int layer,
				 unsigned int module) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int ladder_max = 44;

  Ring ring(Ring::PXBRing);

  for ( unsigned int ladder = 0; ladder < ladder_max; ++ladder ) {
    if ( pxb_[layer][ladder][module] > 0 ) {
      DetId id = constructPXBDetId(layer,ladder,module);
      double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::PXBRing);
      ring.addId(phi,id);
    }
  }
  
  LogDebug("RoadSearch") << "PXB ring initialized rmin/rmax/zmin/zmax: " << rmin 
			 << "/" << rmax 
			 << "/" << zmin 
			 << "/" << zmax;
    
  ring.initialize(rmin,rmax,zmin,zmax);

  return ring;
}

DetId RingMaker::constructPXBDetId(unsigned int layer,
				   unsigned int ladder,
				   unsigned int module) {
  PXBDetId id(layer+1,ladder+1,module+1);
	
  LogDebug("RoadSearch") << "constructed PXB ring DetId for layer: " << id.layer() 
			 << " ladder: " << id.ladder() 
			 << " module: " << id.det(); 

  return DetId(id.rawId());
}

void RingMaker::constructPXFRings(unsigned int &index) {

  unsigned int counter = 0;

  unsigned int fw_bw_max   = 2;
  unsigned int disk_max    = 2; // 2 disks
  unsigned int panel_max   = 2; // 2 sided panel on each blade
  unsigned int module_max  = 4; // 3-4 sensor arrays on each panel

  for ( unsigned int fw_bw = 0; fw_bw < fw_bw_max; ++fw_bw ) {
    for ( unsigned int disk = 0; disk < disk_max; ++disk ) {
      for ( unsigned int panel = 0; panel < panel_max; ++panel ) {
	for ( unsigned int module = 0; module < module_max; ++module ) {
	  if ( pxf2_[fw_bw][disk][panel][module] > 0 ) {
	    Ring ring = constructPXFRing(fw_bw,disk,panel,module);
	    if ( ring.getNumDetIds() > 0 ) {
	      ring.setindex(++index);
	      double center_z = ring.getzmin() + ((ring.getzmax()-ring.getzmin())/2);
	      rings_->insert(center_z,ring);
	      ++counter;
	    }
	  }
	}
      }
    }    
  }    

  LogDebug("RoadSearch") << "constructed " << counter << " PXF rings"; 
  
}

Ring RingMaker::constructPXFRing(unsigned int fw_bw,
				 unsigned int disk,
				 unsigned int panel,
				 unsigned int module) {

  // variables for determinaton of rmin, rmax, zmin, zmax
  float rmin = 1200.;
  float rmax = 0.;
  float zmin = 2800.;
  float zmax = -2800.;

  unsigned int blade_max   = 24;

  Ring ring(Ring::PXFRing);

  for ( unsigned int blade = 0; blade < blade_max; ++blade ) {
    if ( pxf_[fw_bw][disk][blade][panel][module] > 0 ) {
      DetId id = constructPXFDetId(fw_bw,disk,blade,panel,module);
      double phi = determineExtensions(id,rmin,rmax,zmin,zmax,Ring::PXFRing);
      ring.addId(phi,id);
    } 
  }

  LogDebug("RoadSearch") << "PXF ring initialized rmin/rmax/zmin/zmax: " << rmin 
			 << "/" << rmax 
			 << "/" << zmin 
			 << "/" << zmax;
    
  ring.initialize(rmin,rmax,zmin,zmax);

  return ring;
}

DetId RingMaker::constructPXFDetId(unsigned int fw_bw,
				   unsigned int disk,
				   unsigned int blade,
				   unsigned int panel,
				   unsigned int module) {

  PXFDetId id(fw_bw+1,disk+1,blade+1,panel+1,module+1);

  LogDebug("RoadSearch") << "constructed PXF ring DetId for fw_bw: " << id.side() 
			 << " disk: " << id.disk() 
			 << " blade: " << id.blade() 
			 << " panel: " << id.panel() 
			 << " module: " << id.module(); 
	
  return DetId(id.rawId());
}

double RingMaker::determineExtensions(DetId id, float &rmin, float &rmax, float &zmin, float& zmax, Ring::type type) {

  //solution for double modules: loop over r-phi and stereo sensors is required
  std::vector<unsigned int> UseRingIds;

  if ( type == Ring::TOBRing ) {
    TOBDetId tob_axial(id.rawId());
    UseRingIds.push_back(tob_axial.rawId());
    if ( tob_axial.partnerDetId() != 0 ) {
      TOBDetId tob_stereo(tob_axial.partnerDetId());
      UseRingIds.push_back(tob_stereo.rawId());
    }
  }
  if ( type == Ring::TIDRing ) {
    TIDDetId tid_axial(id.rawId());
    UseRingIds.push_back(tid_axial.rawId());
    if ( tid_axial.partnerDetId() != 0 ) {
      TIDDetId tid_stereo(tid_axial.partnerDetId());
      UseRingIds.push_back(tid_stereo.rawId());
    }
  }
  if ( type == Ring::TECRing ) {
    TECDetId tec_axial(id.rawId());
    UseRingIds.push_back(tec_axial.rawId());
    if ( tec_axial.partnerDetId() != 0 ) {
      TECDetId tec_stereo(tec_axial.partnerDetId());
      UseRingIds.push_back(tec_stereo.rawId());
    }
  }
  if ( type == Ring::PXBRing ) {
    PXBDetId pxb_axial(id.rawId());
    UseRingIds.push_back(pxb_axial.rawId());
  }
  if ( type == Ring::PXFRing ) {
    PXFDetId pxf_axial(id.rawId());
    UseRingIds.push_back(pxf_axial.rawId());
  }


  if ( type == Ring::TIBRing ) {
    TIBDetId tib_axial(id.rawId());
    UseRingIds.push_back(tib_axial.rawId());
    if ( tib_axial.partnerDetId() != 0 ) {
      TIBDetId tib_stereo(tib_axial.partnerDetId());
      UseRingIds.push_back(tib_stereo.rawId());
    }
  }
  double phi = 0.;
  for ( std::vector<unsigned int>::iterator it = UseRingIds.begin(); it != UseRingIds.end(); ++it) {

    const GeomDetUnit *det = tracker_->idToDetUnit(DetId(*it));
	
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
      
	const std::vector<float> &parameters = ((const TrapezoidalPlaneBounds&)(det->surface().bounds())).parameters();
      
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
	edm::LogError("RoadSearch") << "problem resolving DetUnit for TIB ring Detid: " << id.rawId() 
				    << " layer: " << tibid.layer() 
				    << " fw(0)/bw(1): " << tibid.string()[0]
				    << " ext(0)/int(0): " << tibid.string()[1] 
				    << " string: " << tibid.string()[2] 
				    << " module: " << tibid.module()
				    << " not stereo(0)/stereo(1): " << tibid.stereo() 
				    << " not glued(0)/glued(1): " << tibid.glued(); 
      } else if ( type == Ring::TOBRing ) {
	TOBDetId tobid(id.rawId()); 
	edm::LogError("RoadSearch") << "problem resolving DetUnit for TOB ring Detid: " << id.rawId() 
				    << " layer: " << tobid.layer() 
				    << " fw(0)/bw(1): " << tobid.rod()[0]
				    << " rod: " << tobid.rod()[1] 
				    << " detector: " << tobid.det()
				    << " not stereo(0)/stereo(1): " << tobid.stereo() 
				    << " not glued(0)/glued(1): " << tobid.glued(); 
      } else if ( type == Ring::TIDRing ) {
	TIDDetId tidid(id.rawId()); 
	edm::LogError("RoadSearch") << "problem resolving DetUnit for TID ring Detid: " << id.rawId() 
				    << " side neg(1)/pos(2): " << tidid.side() 
				    << " wheel: " << tidid.wheel()
				    << " ring: " << tidid.ring()
				    << " module fw(0)/bw(1): " << tidid.module()[0]
				    << " module: " << tidid.module()[1] 
				    << " not stereo(0)/stereo(1): " << tidid.stereo() 
				    << " not glued(0)/glued(1): " << tidid.glued(); 
      } else if ( type == Ring::TECRing ) {
	TECDetId tecid(id.rawId()); 
	edm::LogError("RoadSearch") << "problem resolving DetUnit for TEC ring DetId: " << id.rawId() 
				    << " side neg(1)/pos(2): " << tecid.side() 
				    << " wheel: " << tecid.wheel()
				    << " petal fw(0)/bw(1): " << tecid.petal()[0]
				    << " petal: " << tecid.petal()[1] 
				    << " ring: " << tecid.ring()
				    << " module: " << tecid.module()
				    << " not stereo(0)/stereo(1): " << tecid.stereo() 
				    << " not glued(0)/glued(1): " << tecid.glued(); 
      } else if ( type == Ring::PXBRing ) {
	PXBDetId pxbid(id.rawId()); 
	edm::LogError("RoadSearch") << "problem resolving DetUnit for PXB ring DetId: " << id.rawId() 
				    << " layer: " << pxbid.layer()
				    << " ladder: " << pxbid.ladder()
				    << " module: " << pxbid.module(); 
      } else if ( type == Ring::PXFRing ) {
	PXFDetId pxfid(id.rawId()); 
	edm::LogError("RoadSearch") << "problem resolving DetUnit for PXF ring DetId: " << id.rawId() 
				    << " side: " << pxfid.side()
				    << " disk: " << pxfid.disk()
				    << " blade: " << pxfid.blade()
				    << " panel: " << pxfid.panel()
				    << " module: " << pxfid.module(); 
      }
    }
  }
  LogDebug("RoadSearch") << "id/rmin/rmax/zmin/zmax " << id.rawId() << "/" << rmin <<"/"<<rmax<<"/"<<zmin<<"/"<<zmax;
  return phi;
}

void RingMaker::fillTIBGeometryArray() {
  // fill hardcoded TIB geometry array
  // tib[layer][str_fw_bw][str_int_ext][str][module][stereo]
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
	for (int l = 0; l < 57; ++l) {
	  for (int m = 0; m < 4; ++m) {
	    for (int n =0; n < 3; ++n) {
	      tib_[i][j][k][l][m][n] = 0;
	    }
	  }
	}
      }
    }
  }

  std::vector<DetId> detIds = tracker_->detUnitIds();
  
  for ( std::vector<DetId>::iterator detiterator = detIds.begin(); detiterator != detIds.end(); ++detiterator ) {
    DetId id = *detiterator;

    if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
      TIBDetId tibid(id.rawId());
      
      if( (int)tibid.rawId()-(int)tibid.glued() == (int)tibid.rawId()) {
	//single sided: r-phi 
	//double sided: matched
	tib_[tibid.layer()-1][tibid.string()[0]][tibid.string()[1]][tibid.string()[2]-1][tibid.module()-1][0] += 1;
      } else if ( (int)tibid.rawId()-(int)tibid.glued() == 1) {
	//double sided: stereo
	tib_[tibid.layer()-1][tibid.string()[0]][tibid.string()[1]][tibid.string()[2]-1][tibid.module()-1][1] += 1;
      } else if ( (int)tibid.rawId()-(int)tibid.glued() == 2) { 
	//double sided: r-phi
	tib_[tibid.layer()-1][tibid.string()[0]][tibid.string()[1]][tibid.string()[2]-1][tibid.module()-1][2] += 1;
      } else {
	edm::LogError("RoadSearch") << "stereo of TIBId: " << id.rawId() << " could not be determined." << (int)tibid.glued(); 
      }

    }
  }
}

void RingMaker::fillTIDGeometryArray() {
  // fills hardcoded TID geometry array
  // tid[side][wheel][ring][module_fw_bw][module][stereo]
  // where stereo gives the int of the last constructor parameter
  // the content inidicates if detector with combination exists (>0) or not (==0)

  for (int i = 0; i < 3; ++i ) {
    for (int j = 0; j < 4; ++j ) {
      for (int k = 0; k < 4; ++k ) {
	for (int l = 0; l < 3; ++l ) {
	  for (int m = 0; m < 21; ++m ) {
	    for (int n = 0; n < 3; ++n ) {
	      tid_[i][j][k][l][m][n] = 0;
	    }
	  }
	}
      }
    }
  }

  std::vector<DetId> detIds = tracker_->detUnitIds();
  
  for ( std::vector<DetId>::iterator detiterator = detIds.begin(); detiterator != detIds.end(); ++detiterator ) {
    DetId id = *detiterator;

    if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
      TIDDetId tidid(id.rawId());

      if( (int)tidid.rawId()-(int)tidid.glued() == (int)tidid.rawId()) {
	//single sided: r-phi 
	//double sided: matched
	tid_[tidid.side()-1][tidid.wheel()-1][tidid.ring()-1][tidid.module()[0]][tidid.module()[1]-1][0] += 1;
      } else if ( (int)tidid.rawId()-(int)tidid.glued() == 1) {
	//double sided: stereo
	tid_[tidid.side()-1][tidid.wheel()-1][tidid.ring()-1][tidid.module()[0]][tidid.module()[1]-1][1] += 1;
      } else if ( (int)tidid.rawId()-(int)tidid.glued() == 2) { 
	//double sided: r-phi
	tid_[tidid.side()-1][tidid.wheel()-1][tidid.ring()-1][tidid.module()[0]][tidid.module()[1]-1][2] += 1;
      } else {
	edm::LogError("RoadSearch") << "stereo of TIDId: " << id.rawId() << " could not be determined."; 
      }

    }
  }
}

void RingMaker::fillTOBGeometryArray() {
  // fills hardcoded TOB geometry array
  // tob[layer][rod_fw_bw][rod][module][stereo]
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 75; ++k) {
	for (int l = 0; l < 7; ++l) {
	  for (int m = 0; m < 3; ++m) {
	    tob_[i][j][k][l][m] = 0;
	  }
	}
      }
    }
  }

  std::vector<DetId> detIds = tracker_->detUnitIds();
  
  for ( std::vector<DetId>::iterator detiterator = detIds.begin(); detiterator != detIds.end(); ++detiterator ) {
    DetId id = *detiterator;

    if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
      TOBDetId tobid(id.rawId());

      if( (int)tobid.rawId()-(int)tobid.glued() == (int)tobid.rawId()) {
	//single sided: r-phi 
	//double sided: matched
	tob_[tobid.layer()-1][tobid.rod()[0]][tobid.rod()[1]-1][tobid.module()-1][0] += 1;
      } else if ( (int)tobid.rawId()-(int)tobid.glued() == 1) {
	//double sided: stereo
	tob_[tobid.layer()-1][tobid.rod()[0]][tobid.rod()[1]-1][tobid.module()-1][1] += 1;
      } else if ( (int)tobid.rawId()-(int)tobid.glued() == 2) { 
	//double sided: r-phi
	tob_[tobid.layer()-1][tobid.rod()[0]][tobid.rod()[1]-1][tobid.module()-1][2] += 1;
      } else {
	edm::LogError("RoadSearch") << "stereo of TOBId: " << id.rawId() << " could not be determined."; 
      }

    }
  }
}

void RingMaker::fillTECGeometryArray() {
  // fills hardcoded TEC geometry array
  // tec[side][wheel][petal_fw_bw][petal][ring][module][stereo]
  // where stereo gives the int of the last constructor parameter
  // the content inidicates if detector with combination exists (>0) or not (==0)

  // fill two arrays to restrict first loop (number of rings dependent on wheel)

  for (int i = 0; i < 3; ++i ) {
    for (int j = 0; j < 10; ++j ) {
      for (int k = 0; k < 3; ++k ) {
	for (int l = 0; l < 9; ++l ) {
	  for (int m = 0; m < 8; ++m ) {
	    for (int n = 0; n < 21; ++n ) {
	      for (int o = 0; o < 3; ++o ) {
		tec_[i][j][k][l][m][n][o] = 0;
	      }
	    }	
	  }
	}
      }
    }
  }

  for (int i = 0; i < 3; ++i ) {
    for (int j = 0; j < 10; ++j ) {
      for (int k = 0; k < 8; ++k ) {
	tec2_[i][j][k] = 0;
      }
    }	
  }

  std::vector<DetId> detIds = tracker_->detUnitIds();
  
  for ( std::vector<DetId>::iterator detiterator = detIds.begin(); detiterator != detIds.end(); ++detiterator ) {
    DetId id = *detiterator;

    if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
      TECDetId tecid(id.rawId());

      if( (int)tecid.rawId()-(int)tecid.glued() == (int)tecid.rawId()) {
	//single sided: r-phi 
	//double sided: matched
	tec_[tecid.side()-1][tecid.wheel()-1][tecid.petal()[0]][tecid.petal()[1]-1][tecid.ring()-1][tecid.module()-1][0] += 1;
	tec2_[tecid.side()-1][tecid.wheel()-1][tecid.ring()-1] += 1;
      } else if ( (int)tecid.rawId()-(int)tecid.glued() == 1) {
	//double sided: stereo
	tec_[tecid.side()-1][tecid.wheel()-1][tecid.petal()[0]][tecid.petal()[1]-1][tecid.ring()-1][tecid.module()-1][1] += 1;
	tec2_[tecid.side()-1][tecid.wheel()-1][tecid.ring()-1] += 1;
      } else if ( (int)tecid.rawId()-(int)tecid.glued() == 2) { 
	//double sided: r-phi
	tec_[tecid.side()-1][tecid.wheel()-1][tecid.petal()[0]][tecid.petal()[1]-1][tecid.ring()-1][tecid.module()-1][2] += 1;
	tec2_[tecid.side()-1][tecid.wheel()-1][tecid.ring()-1] += 1;
      } else {
	edm::LogError("RoadSearch") << "stereo of TECId: " << id.rawId() << " could not be determined."; 
      }

    }
  }
}

void RingMaker::fillPXBGeometryArray() {
  // fills hardcoded PXB geometry array: pxb[layer][ladder][module]
  // module gives the int of the last constructor parameter
  // content of [module] indicates if module with combination exists (>0) or not (==0)

  for (int i = 0; i < 3; ++i ) {
    for (int j = 0; j < 44; ++j ) {
      for (int k = 0; k < 8; ++k ) {
	pxb_[i][j][k] = 0;
      }
    }
  }

  std::vector<DetId> detIds = tracker_->detUnitIds();
  
  for ( std::vector<DetId>::iterator detiterator = detIds.begin(); detiterator != detIds.end(); ++detiterator ) {
    DetId id = *detiterator;

    if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
      PXBDetId pxbid(id.rawId());

      //sanity check with partner ID not possible
      pxb_[pxbid.layer()-1][pxbid.ladder()-1][pxbid.module()-1] += 1;
      
    }
  }
}

void RingMaker::fillPXFGeometryArray() {
  // fills hardcoded PXF geometry array: pxf[side][disk][blade][panel][module]
  // module gives the int of the last constructor parameter
  // content of [module] indicates if module with combination exists (>0) or not (==0)

  for (int i = 0; i < 2; ++i ) {
    for (int j = 0; j < 2; ++j ) {
      for (int k = 0; k < 24; ++k ) {
	for (int l = 0; l < 2; ++l ) {
	  for (int m = 0; m < 4; ++m ) {
	    pxf_[i][j][k][l][m] = 0;
	  }
	}
      }
    }
  }

  for (int i = 0; i < 2; ++i ) {
    for (int j = 0; j < 2; ++j ) {
	for (int k = 0; k < 2; ++k ) {
	  for (int l = 0; l < 4; ++l ) {
	    pxf2_[i][j][k][l] = 0;
	  }
	}
    }
  }

  std::vector<DetId> detIds = tracker_->detUnitIds();
  
  for ( std::vector<DetId>::iterator detiterator = detIds.begin(); detiterator != detIds.end(); ++detiterator ) {
    DetId id = *detiterator;

    if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
      PXFDetId pxfid(id.rawId());

      //sanity check with partner ID not possible
      pxf_[pxfid.side()-1][pxfid.disk()-1][pxfid.blade()-1][pxfid.panel()-1][pxfid.module()-1] += 1;
      pxf2_[pxfid.side()-1][pxfid.disk()-1][pxfid.panel()-1][pxfid.module()-1] += 1;
      
    }
  }
}

bool RingMaker::dumpDetIdsIntoFile(std::string filename) {
  //
  // dumps all tracker detids in geometry
  //

  // return value
  bool result = true;

  std::ofstream output(filename.c_str());

  output << dumpDetIds();

  return result;

}

std::string RingMaker::dumpDetIds() {
  //
  // dumps all tracker detids in geometry
  //

  std::ostringstream output;

  std::vector<DetId> detIds = tracker_->detUnitIds();

  output << std::endl << "[RoadMaker] Total number of DETECTOR = " << detIds.size() << std::endl;

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
    } else {
      output << "[RoadMaker] DetUnit for unknown subdetector: " << id.rawId() << std::endl;
    }

  }

  return output.str();
  
}

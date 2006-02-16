#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"
#include "Geometry/CommonDetUnit/interface/ModifiedSurfaceGenerator.h"
#include "Geometry/CommonDetAlgo/interface/BoundingBox.h"

//#include "CommonDet/BasicDet/interface/DetUnit.h"
//#include "CommonDet/PatternPrimitives/interface/Propagator.h"


ForwardDetLayer::~ForwardDetLayer() {}


//--- GeometricSearchDet interface
const BoundSurface& ForwardDetLayer::surface() const { 
  return *theDisk;
}


//--- Extension of the interface
void ForwardDetLayer::setSurface( BoundDisk* cp) { 
  theDisk = cp;
}

bool ForwardDetLayer::contains(const Local3DPoint& p) const {
  return surface().bounds().inside(p);
}


void ForwardDetLayer::initialize() 
{
  setSurface( computeSurface());
}


//typedef std::vector<const GeometricSearchDet*>  DetContainer;

BoundDisk* ForwardDetLayer::computeSurface() {

  vector<const GeometricSearchDet*>::const_iterator ifirst = components().begin();
  vector<const GeometricSearchDet*>::const_iterator ilast  = components().end();

  // Find extension in R
  // float tolerance = 1.; // cm
  theRmin = (**ifirst).position().perp(); theRmax = theRmin;
  theZmin = (**ifirst).position().z(); theZmax = theZmin;
  for ( vector<const GeometricSearchDet*>::const_iterator deti = ifirst;
	deti != ilast; deti++) {
    vector<GlobalPoint> corners = 
      BoundingBox().corners( dynamic_cast<const BoundPlane&>((**deti).surface()));
    for (vector<GlobalPoint>::const_iterator ic = corners.begin();
	 ic != corners.end(); ic++) {
      float r = ic->perp();
      float z = ic->z();
      theRmin = min( theRmin, r);
      theRmax = max( theRmax, r);
      theZmin = min( theZmin, z);
      theZmax = max( theZmax, z);
    }

    // in addition to the corners we have to check the middle of the 
    // det +/- length/2
    // , since the min (max) radius for typical fw dets is reached there
    float rdet = (**deti).position().perp();
    float len = (**deti).surface().bounds().length();
    theRmin = min( theRmin, rdet-len/2.F);
    theRmax = max( theRmax, rdet+len/2.F);
  }

#ifdef DEBUG_GEOM
  cout << "creating SimpleDiskBounds with r range" << theRmin << " " 
       << theRmax << " and z range " << theZmin << " " << theZmax << endl;
#endif

  // By default the forward layers are positioned around the z axis of the
  // global frame, and the axes of their local frame coincide with 
  // those of the global grame (z along the disk axis)
  float zPos = (theZmax+theZmin)/2.;
  PositionType pos(0.,0.,zPos);
  RotationType rot;

  return new BoundDisk( pos, rot, 
			SimpleDiskBounds( theRmin, theRmax, 
					  theZmin-zPos, theZmax-zPos));
}  



#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"
#include "Geometry/CommonDetUnit/interface/ModifiedSurfaceGenerator.h"
#include "Geometry/Surface/interface/BoundingBox.h"

using namespace std;

ForwardDetLayer::ForwardDetLayer() : 
  theDisk(0),
  theRmin(0), theRmax(0), theZmin(0), theZmax(0)
{}

ForwardDetLayer::ForwardDetLayer( float initPos) : 
  theDisk(0),
  theRmin(0), theRmax(0), theZmin(0), theZmax(0)
{}


ForwardDetLayer::~ForwardDetLayer() {
}


//--- Extension of the interface
void ForwardDetLayer::setSurface( BoundDisk* cp) { 
  theDisk = cp;
}

bool ForwardDetLayer::contains(const Local3DPoint& p) const {
  return surface().bounds().inside(p);
}


void ForwardDetLayer::initialize() {
  setSurface( computeSurface());
}


BoundDisk* ForwardDetLayer::computeSurface() {

  // FIXME: it could work (faster) with components() instead of basicComponents()
  vector<const GeomDet*> comps= basicComponents();

  vector<const GeomDet*>::const_iterator ifirst = comps.begin();
  vector<const GeomDet*>::const_iterator ilast  = comps.end();

  // Find extension in R
  // float tolerance = 1.; // cm
  theRmin = basicComponents().front()->position().perp(); 
  theRmax = theRmin;
  theZmin = basicComponents().back()->position().z();
  theZmax = theZmin;
  for ( vector<const GeomDet*>::const_iterator deti = ifirst;
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


  LogDebug("DetLayers") << "creating SimpleDiskBounds with r range" << theRmin << " " 
			<< theRmax << " and z range " << theZmin << " " << theZmax ;


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


pair<bool, TrajectoryStateOnSurface>
ForwardDetLayer::compatible( const TrajectoryStateOnSurface& ts, 
			     const Propagator& prop, 
			     const MeasurementEstimator& est) const
{
  TrajectoryStateOnSurface myState = prop.propagate( ts, specificSurface());
  if ( !myState.isValid()) return make_pair( false, myState);

  // take into account the thickness of the layer
  float deltaR = surface().bounds().thickness()/2. *
    fabs( tan( myState.localDirection().theta()));

  // take into account the error on the predicted state
  const float nSigma = 3.;
  if (myState.hasError()) {
    LocalError err = myState.localError().positionError();
    // ignore correlation for the moment...
    deltaR += nSigma * sqrt(err.xx() + err.yy());
  }

  float zPos = (theZmax+theZmin)/2.;
  SimpleDiskBounds tmp( theRmin-deltaR, theRmax+deltaR, 
			theZmin-zPos, theZmax-zPos);

  return make_pair( tmp.inside(myState.localPosition()), myState);
}

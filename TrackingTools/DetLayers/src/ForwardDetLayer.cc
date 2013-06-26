#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"

using namespace std;


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
  LogDebug("DetLayers") << "ForwaLayer::computeSurface callded" ;
  vector<const GeomDet*> comps= basicComponents();

  vector<const GeomDet*>::const_iterator ifirst = comps.begin();
  vector<const GeomDet*>::const_iterator ilast  = comps.end();

  // Find extension in R
  float theRmin = components().front()->position().perp(); 
  float theRmax = theRmin;
  float theZmin = components().back()->position().z();
  float theZmax = theZmin;
  for ( vector<const GeomDet*>::const_iterator deti = ifirst;
	deti != ilast; deti++) {
    vector<GlobalPoint> corners = 
      BoundingBox().corners( dynamic_cast<const Plane&>((**deti).surface()));
    for (vector<GlobalPoint>::const_iterator ic = corners.begin();
	 ic != corners.end(); ic++) {
      float r = ic->perp();
      LogDebug("DetLayers") << "corner.perp(): " << r ;
      float z = ic->z();
      theRmin = min( theRmin, r);
      theRmax = max( theRmax, r);
      theZmin = min( theZmin, z);
      theZmax = max( theZmax, z);
    }

    // in addition to the corners we have to check the middle of the 
    // det +/- length/2
    // , since the min (max) radius for typical fw dets is reached there

    float rdet  = (**deti).position().perp();
    float len   = (**deti).surface().bounds().length();
    float width = (**deti).surface().bounds().width();

    GlobalVector xAxis = (**deti).toGlobal(LocalVector(1,0,0));
    GlobalVector yAxis = (**deti).toGlobal(LocalVector(0,1,0));
    GlobalVector perpDir = GlobalVector( (**deti).position() - GlobalPoint(0,0,(**deti).position().z()) );

    double xAxisCos = xAxis.unit().dot(perpDir.unit());
    double yAxisCos = yAxis.unit().dot(perpDir.unit());

    LogDebug("DetLayers") << "in ForwardDetLayer::computeSurface(),xAxisCos,yAxisCos: " << xAxisCos << " , " << yAxisCos ;
    LogDebug("DetLayers") << "det pos.perp,length,width: " 
			  << rdet << " , " 
			  << len  << " , "
			  << width ;

    if( fabs(xAxisCos) > fabs(yAxisCos) ) {
      theRmin = min( theRmin, rdet-width/2.F);
      theRmax = max( theRmax, rdet+width/2.F);
    }else{
      theRmin = min( theRmin, rdet-len/2.F);
      theRmax = max( theRmax, rdet+len/2.F);
    }
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
			new SimpleDiskBounds( theRmin, theRmax, 
			      		      theZmin-zPos, theZmax-zPos));
}  


pair<bool, TrajectoryStateOnSurface>
ForwardDetLayer::compatible( const TrajectoryStateOnSurface& ts, 
			     const Propagator& prop, 
			     const MeasurementEstimator&) const
{
  if unlikely(theDisk == 0)  edm::LogError("DetLayers") 
    << "ERROR: BarrelDetLayer::compatible() is used before the layer surface is initialized" ;
  // throw an exception? which one?

  TrajectoryStateOnSurface myState = prop.propagate( ts, specificSurface());
  if unlikely( !myState.isValid()) return make_pair( false, myState);

  // take into account the thickness of the layer
  float deltaR = 0.5f*surface().bounds().thickness() *
    myState.localDirection().perp()/std::abs(myState.localDirection().z());

  // take into account the error on the predicted state
  const float nSigma = 3.;
  if (myState.hasError()) {
    LocalError err = myState.localError().positionError();
    // ignore correlation for the moment...
    deltaR += nSigma * sqrt(err.xx() + err.yy());
  }

  float zPos = 0.5f*(zmax()+zmin());
  SimpleDiskBounds tmp( rmin()-deltaR, rmax()+deltaR, 
			zmin()-zPos, zmax()-zPos);

  return make_pair( tmp.inside(myState.localPosition()), myState);
}

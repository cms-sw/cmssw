#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "Geometry/CommonDetAlgo/interface/BoundingBox.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"
typedef GeometricSearchDet::DetWithState DetWithState;

PixelForwardLayer::PixelForwardLayer(vector<const PixelBlade*>& blades):
  theBlades(blades.begin(),blades.end())
{
  for(vector<const PixelBlade*>::const_iterator it=theBlades.begin();
      it!=theBlades.end();it++){
    theComps.push_back(*it);
  }

  //They should be already phi-ordered. TO BE CHECKED!!
  //sort( theBlades.begin(), theBlades.end(), PhiLess());
  setSurface( computeDisk(theBlades) );
  

  /*--------- DEBUG INFO --------------
  cout << "DEBUG INFO for PixelForwardLayer" << endl;
  cout << "PixelForwardLayer.surfcace.z(): " 
       << this->surface().position().z() << endl;
  cout << "PixelForwardLayer.surfcace.innerR(): " 
       << this->specificSurface().innerRadius() << endl;
  cout << "PixelForwardLayer.surfcace.outerR(): " 
       << this->specificSurface().outerRadius() << endl;
  //cout << "PixelForwardLayer.surfcace.thickness(): " << specificSurface().thickness << end
  for(vector<const PixelBlade*>::const_iterator it=theBlades.begin(); 
      it!=theBlades.end(); it++){
    cout << "blades.phi: " << (*it)->surface().position().phi() << endl;
  }
  -----------------------------------*/

    
}

PixelForwardLayer::~PixelForwardLayer(){
  vector<const PixelBlade*>::const_iterator i;
  for (i=theBlades.begin(); i!=theBlades.end(); i++) {
    delete *i;
  }
} 

vector<const GeomDet*> 
PixelForwardLayer::basicComponents() const{
  cout << "temporary dummy implementation of PixelForwardLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
PixelForwardLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of PixelForwardLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
PixelForwardLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop, 
		      const MeasurementEstimator& est) const{

  // standard implementation of compatibleDets() for class which have 
  // groupedCompatibleDets implemented.
  // This code should be moved in a common place intead of being 
  // copied many times.
  
  vector<DetWithState> result;  
  vector<DetGroup> vectorGroups = groupedCompatibleDets(startingState,prop,est);
  for(vector<DetGroup>::const_iterator itDG=vectorGroups.begin();
      itDG!=vectorGroups.end();itDG++){
    for(vector<DetGroupElement>::const_iterator itDGE=itDG->begin();
	itDGE!=itDG->end();itDGE++){
      result.push_back(DetWithState(itDGE->det(),itDGE->trajectoryState()));
    }
  }
  return result;  
}


vector<DetGroup> 
PixelForwardLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
					  const Propagator& prop,
					  const MeasurementEstimator& est) const{  
  return vector<DetGroup>();
}



BoundDisk* 
PixelForwardLayer::computeDisk(const vector<const PixelBlade*>& blades) const
{
  vector<const PixelBlade*>::const_iterator ifirst = blades.begin();
  vector<const PixelBlade*>::const_iterator ilast  = blades.end();

  // Find extension in R
  // float tolerance = 1.; // cm
  float theRmin = (**ifirst).position().perp(); float theRmax = theRmin;
  float theZmin = (**ifirst).position().z(); float theZmax = theZmin;
  for ( vector<const PixelBlade*>::const_iterator deti = ifirst;
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
  Surface::PositionType pos(0.,0.,zPos);
  Surface::RotationType rot;

  return new BoundDisk( pos, rot, 
			SimpleDiskBounds( theRmin, theRmax, 
					  theZmin-zPos, theZmax-zPos));
}    

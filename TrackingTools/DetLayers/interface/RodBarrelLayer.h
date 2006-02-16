#ifndef DetLayers_RodBarrelLayer_H
#define DetLayers_RodBarrelLayer_H

/** \class RodBarrelLayer
 *  Abstract class for a cylinder composed of rods of detetors.
 */

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

class DetRod;
class MeasurementEstimator;

class RodBarrelLayer : public BarrelDetLayer {

 public:

  RodBarrelLayer() {};

  RodBarrelLayer( vector<const Det*>::const_iterator firstRod,
		  vector<const Det*>::const_iterator lastRod);

  RodBarrelLayer( const vector<const Det*>& theRods);
  
  virtual ~RodBarrelLayer();

  
  //--- GeometricSearchDet interface
  virtual vector<const GeometricSearchDet*> components() const {return theDets;}
  

  
  //--- Extension of the interface
  /*
  virtual Module module() const;

  //Should become pure virtual!!!
  virtual Det* operator()( double x, double phi) const {return 0;}

  virtual void addDets( detunit_p_iter ifirst, detunit_p_iter ilast);
  */

  
 private:
  vector<const Det*> theDets;
};

#endif

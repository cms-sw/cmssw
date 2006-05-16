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

//   RodBarrelLayer( vector<const GeomDet*>::const_iterator first,
// 		     vector<const GeomDet*>::const_iterator last);

//   RodBarrelLayer( const vector<const GeomDet*>& dets);
  
  virtual ~RodBarrelLayer();

  
  //--- GeometricSearchDet interface
//   virtual vector<const GeometricSearchDet*> components() const {return theDets;}
  
  //--- Extension of the interface
//   virtual GeomDet* operator()( double x, double phi) const = 0;

//   virtual void addDets( detunit_p_iter ifirst, detunit_p_iter ilast);

  
 private:
  //  vector<const GeomDet*> theDets;
};

#endif

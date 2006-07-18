#ifndef DetLayers_ForwardDetLayer_H
#define DetLayers_ForwardDetLayer_H

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/Surface/interface/ReferenceCounted.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include <vector>
#include <algorithm>


/** A specialization of the DetLayer interface for forward layers.
 *  Forward layers are disks with their axes parallel to 
 *  the global Z axis.
 *  The methods that have a common implementation for all ForwardDetLayers
 *  are implemented in this class,
 *  but some methods are left abstract.
 */

class ForwardDetLayer : public DetLayer {
public:

  ForwardDetLayer();

  ForwardDetLayer( float initPos);

  virtual ~ForwardDetLayer();

  // GeometricSearchDet interface
  virtual const BoundSurface&  surface() const { return *theDisk;}

  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&, 
	      const MeasurementEstimator&) const;

  // DetLayer interface
  virtual Part   part()   const { return forward;}

  // Extension of the interface
  virtual const BoundDisk&    specificSurface() const { return *theDisk;}

  void setSurface( BoundDisk* cp);
  
  bool contains( const Local3DPoint& p) const;  
  
 protected:

  virtual void initialize();

  float rmin() const { return theRmin;}
  float rmax() const { return theRmax;}
  virtual BoundDisk* computeSurface();

 private:
  
  ReferenceCountingPointer<BoundDisk> theDisk;

  float theRmin, theRmax, theZmin, theZmax;
};


#endif 

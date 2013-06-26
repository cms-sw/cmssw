#ifndef DetLayers_ForwardDetLayer_H
#define DetLayers_ForwardDetLayer_H

/** A specialization of the DetLayer interface for forward layers.
 *  Forward layers are disks with their axes parallel to 
 *  the global Z axis.
 *  The methods that have a common implementation for all ForwardDetLayers
 *  are implemented in this class,
 *  but some methods are left abstract.
 */

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include <vector>
#include <algorithm>

class ForwardDetLayer : public DetLayer {
public:

  ForwardDetLayer();

  virtual ~ForwardDetLayer();

  // GeometricSearchDet interface
  virtual const BoundSurface&  surface() const { return *theDisk;}

  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&, 
	      const MeasurementEstimator&) const;

  // DetLayer interface
  virtual Location   location()   const {return GeomDetEnumerators::endcap;}

  // Extension of the interface
  virtual const BoundDisk&    specificSurface() const { return *theDisk;}

  bool contains( const Local3DPoint& p) const;  
  
 protected:
  void setSurface( BoundDisk* cp);

  virtual void initialize();

  float rmin() const { return theDisk->innerRadius();}
  float rmax() const { return theDisk->outerRadius();}
  float zmin() const { return (theDisk->position().z() - theDisk->bounds().thickness()/2);}
  float zmax() const { return (theDisk->position().z() + theDisk->bounds().thickness()/2);}


  virtual BoundDisk* computeSurface();


 private:
  //float theRmin, theRmax, theZmin, theZmax;
  ReferenceCountingPointer<BoundDisk> theDisk;


};


#endif 

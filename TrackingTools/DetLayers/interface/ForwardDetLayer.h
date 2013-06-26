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

  ForwardDetLayer(): DetLayer(false),  theDisk(0){}

  virtual ~ForwardDetLayer();

  // GeometricSearchDet interface
  virtual const BoundSurface&  surface() const GCC11_FINAL { return *theDisk;}

  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&, 
	      const MeasurementEstimator&) const;

  // DetLayer interface
  virtual Location location() const  GCC11_FINAL {return GeomDetEnumerators::endcap;}

  // Extension of the interface
  virtual const BoundDisk& specificSurface() const  GCC11_FINAL { return *theDisk;}

  bool contains( const Local3DPoint& p) const;  
  
 protected:
  void setSurface( BoundDisk* cp);

  virtual void initialize();

  float rmin() const { return theDisk->innerRadius();}
  float rmax() const { return theDisk->outerRadius();}
  float zmin() const { return (theDisk->position().z() - theDisk->bounds().thickness()*0.5f);}
  float zmax() const { return (theDisk->position().z() + theDisk->bounds().thickness()*0.5f);}


  virtual BoundDisk* computeSurface();


 private:
  ReferenceCountingPointer<BoundDisk> theDisk;


};


#endif 

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
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"


#include <vector>
#include <algorithm>

class ForwardDetLayer : public DetLayer {
public:

  ForwardDetLayer(bool doHaveGroups): DetLayer(doHaveGroups,false) {}

  ~ForwardDetLayer() override;

  // GeometricSearchDet interface
  const BoundSurface&  surface() const final { return *theDisk;}

  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&, 
	      const MeasurementEstimator&) const override;

  // DetLayer interface
  Location location() const  final {return GeomDetEnumerators::endcap;}

  // Extension of the interface
  virtual const BoundDisk& specificSurface() const  final { return *theDisk;}

  bool contains( const Local3DPoint& p) const;  
  
 protected:

  virtual void initialize();


  float rmin() const { return theDisk->innerRadius();}
  float rmax() const { return theDisk->outerRadius();}
  float zmin() const { return (theDisk->position().z() - bounds().thickness()*0.5f);}
  float zmax() const { return (theDisk->position().z() + bounds().thickness()*0.5f);}

  void setSurface( BoundDisk* cp);
  virtual BoundDisk* computeSurface();

  SimpleDiskBounds const & bounds() const { return static_cast<SimpleDiskBounds const &>(theDisk->bounds());} 

 private:
  ReferenceCountingPointer<BoundDisk> theDisk;


};


#endif 

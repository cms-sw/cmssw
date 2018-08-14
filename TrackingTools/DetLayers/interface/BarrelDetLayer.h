#ifndef DetLayers_BarrelDetLayer_H
#define DetLayers_BarrelDetLayer_H

/** \class BarrelDetLayer
 *  A specialization of the DetLayer interface for barrel layers.
 *  Barrel layers are cylinders with their axes parallel to 
 *  the global Z axis.
 *  The methods that have a common implementation for all BarrelDetLayers
 *  are implemented in this class,
 *  but some methods are left abstract.
 *
 */

#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include <vector>
#include <algorithm>


class BarrelDetLayer : public DetLayer {
 public:

  BarrelDetLayer(bool doHaveGroup) : DetLayer(doHaveGroup,true),
    theCylinder(nullptr){}
  
  ~BarrelDetLayer() override;

  /// GeometricSearchDet interface
  const BoundSurface&  surface() const  final { return *theCylinder;}

  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const final;

  /// DetLayer interface
  Location location() const final {return GeomDetEnumerators::barrel;}


  /// Extension of the interface
  virtual const BoundCylinder&  specificSurface() const final { return *theCylinder;}

  bool contains( const Local3DPoint& p) const;



protected:

  virtual void initialize();

  void setSurface( BoundCylinder* cp);
  virtual BoundCylinder* computeSurface();

  SimpleCylinderBounds const & bounds() const { return static_cast<SimpleCylinderBounds const &>(theCylinder->bounds());} 


private:
  //float theRmin, theRmax, theZmin, theZmax;
  ReferenceCountingPointer<BoundCylinder>  theCylinder;

};

#endif 

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
 *  $Date: 2012/12/14 08:16:36 $
 *  $Revision: 1.10 $
 */

#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include <vector>
#include <algorithm>


class BarrelDetLayer : public DetLayer {
 public:

  BarrelDetLayer() : DetLayer(true),
    theCylinder(0){}
  
  virtual ~BarrelDetLayer();

  /// GeometricSearchDet interface
  virtual const BoundSurface&  surface() const  GCC11_FINAL { return *theCylinder;}

  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const GCC11_FINAL;

  /// DetLayer interface
  virtual Location location() const GCC11_FINAL {return GeomDetEnumerators::barrel;}


  /// Extension of the interface
  virtual const BoundCylinder&  specificSurface() const GCC11_FINAL { return *theCylinder;}

  bool contains( const Local3DPoint& p) const;



protected:
  void setSurface( BoundCylinder* cp);

  virtual void initialize();

  virtual BoundCylinder* computeSurface();


private:
  //float theRmin, theRmax, theZmin, theZmax;
  ReferenceCountingPointer<BoundCylinder>  theCylinder;

};

#endif 

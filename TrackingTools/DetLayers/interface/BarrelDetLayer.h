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
 *  $Date: 2007/03/07 16:28:39 $
 *  $Revision: 1.7 $
 */

#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include <vector>
#include <algorithm>


class BarrelDetLayer : public DetLayer {
 public:

  BarrelDetLayer() : 
    theCylinder(0){}
  
  virtual ~BarrelDetLayer();

  /// GeometricSearchDet interface
  virtual const BoundSurface&  surface() const { return *theCylinder;}

  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  /// DetLayer interface
  virtual Location location()   const {return GeomDetEnumerators::barrel;}


  /// Extension of the interface
  virtual const BoundCylinder&  specificSurface() const { return *theCylinder;}

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

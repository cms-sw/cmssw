#ifndef DetLayers_BarrelDetLayer_H
#define DetLayers_BarrelDetLayer_H

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Surface/interface/ReferenceCounted.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include <vector>
#include <algorithm>


/** A specialization of the DetLayer interface for barrel layers.
 *  Barrel layers are cylinders with their axes parallel to 
 *  the global Z axis.
 *  The methods that have a common implementation for all BarrelDetLayers
 *  are implemented in this class,
 *  but some methods are left abstract.
 */

class BarrelDetLayer : public DetLayer {
 public:

  BarrelDetLayer() : 
    theCylinder(0), theInitialPosition(0) {}
  
  BarrelDetLayer( float initPos) : 
    theCylinder(0), theInitialPosition(initPos) {}
  

  virtual ~BarrelDetLayer();

  // GeometricSearchDet interface
  virtual const BoundSurface&  surface() const { return *theCylinder;}


  // DetLayer interface
  virtual Part   part()   const { return barrel;}


  // Extension of the interface
  virtual const BoundCylinder&  specificSurface() const { return *theCylinder;}

  void setSurface( BoundCylinder* cp);

  bool contains( const Local3DPoint& p) const;

  virtual float initialPosition() const { return theInitialPosition;}
  
  virtual void initialize();
  


protected:

  virtual BoundCylinder* computeSurface();

private:

  ReferenceCountingPointer<BoundCylinder>  theCylinder;
  float theInitialPosition;
  
  float theRmin, theRmax, theZmin, theZmax;

};

#endif 

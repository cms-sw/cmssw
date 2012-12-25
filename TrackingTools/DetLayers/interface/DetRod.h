#ifndef DetLayers_DetRod_H
#define DetLayers_DetRod_H

/** \class DetRod
 *  Abstract interface for a rod of detectors sitting on a Plane.
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

class MeasurementEstimator;

class DetRod : public virtual GeometricSearchDet   {
 public:
  
  virtual ~DetRod();
  
 
  virtual const BoundSurface& surface() const GCC11_FINAL {return *thePlane;}


  //--- Extension of the interface
  
  /// Return the rod surface as a Plane
  virtual const Plane& specificSurface() const GCC11_FINAL {return *thePlane;}


protected:
  /// Set the rod's plane
  void setPlane( Plane* plane) { thePlane = plane;}

  //obsolete?
  // Return the range in Z to be checked for compatibility
  //float zError( const TrajectoryStateOnSurface& tsos,
  //		const MeasurementEstimator& est) const;

 private:
  ReferenceCountingPointer<Plane>  thePlane;

};

#endif

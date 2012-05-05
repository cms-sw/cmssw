#ifndef DetLayers_DetRod_H
#define DetLayers_DetRod_H

/** \class DetRod
 *  Abstract interface for a rod of detectors sitting on a BoundPlane.
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

class MeasurementEstimator;

class DetRod : public virtual GeometricSearchDet   {
 public:
  
  virtual ~DetRod();
  
 
  virtual const BoundSurface& surface() const GCC11_FINAL {return *thePlane;}


  //--- Extension of the interface
  
  /// Return the rod surface as a BoundPlane
  virtual const BoundPlane& specificSurface() const GCC11_FINAL {return *thePlane;}


protected:
  /// Set the rod's plane
  void setPlane( BoundPlane* plane) { thePlane = plane;}

  //obsolete?
  // Return the range in Z to be checked for compatibility
  //float zError( const TrajectoryStateOnSurface& tsos,
  //		const MeasurementEstimator& est) const;

 private:
  ReferenceCountingPointer<BoundPlane>  thePlane;

};

#endif

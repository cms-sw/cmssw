#ifndef ForwardDetRing_H
#define ForwardDetRing_H

/** \class ForwardDetRing
 *  Abstract interface for a ring of detectors sitting on a BoundDisk.
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

class ForwardDetRing : public GeometricSearchDet {
 public:

  using GeometricSearchDet::GeometricSearchDet;


  ~ForwardDetRing() override;

  
  void
  compatibleDetsV( const TrajectoryStateOnSurface& startingState,
		   const Propagator& prop, 
		   const MeasurementEstimator& est,
		   std::vector<DetWithState>& result) const override;
  
  const BoundSurface& surface() const final {return *theDisk;}

  
  //--- Extension of the interface

  /// Return the ring surface as a BoundDisk
  const BoundDisk& specificSurface() const {return *theDisk;}


protected:

  /// Set the rod's disk
  void setDisk( BoundDisk* disk) { theDisk = disk;}

  
 private:
  ReferenceCountingPointer<BoundDisk>  theDisk;

};
#endif


#ifndef DetLayers_DetRodOneR_H
#define DetLayers_DetRodOneR_H

/** \class DetRodOneR
 *  A rod of detectors, all having the same Plane.
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

class MeasurementEstimator;

class DetRodOneR : public DetRod {
 public: 
  typedef std::vector<GeometricSearchDet*> DetContainer;

  /// Dummy constructor
  DetRodOneR(){};

  /// Construct from iterators on GeomDet*
  DetRodOneR( std::vector<const GeomDet*>::const_iterator first,
	      std::vector<const GeomDet*>::const_iterator last);

  /// Construct from a std::vector of GeomDet*
  DetRodOneR( const std::vector<const GeomDet*>& dets);

  virtual ~DetRodOneR();

  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}



protected:
  /// Query detector idet for compatible and add the output to result.
  
  bool add( int idet, std::vector<DetWithState>& result,
	    const TrajectoryStateOnSurface& startingState,
	    const Propagator& prop, 
	    const MeasurementEstimator& est) const;

  std::vector<const GeomDet*>     theDets;
  
  void initialize();

};

#endif

#ifndef DetLayers_ForwardDetRingOneZ_H
#define DetLayers_ForwardDetRingOneZ_H

/** \class ForwardDetRingOneZ
 *  A ring of detectors, all having the same BoundDisk.
 */

#include "TrackingTools/DetLayers/interface/ForwardDetRing.h"


class ForwardDetRingOneZ : public ForwardDetRing {
public:

  /// Dummy constructor
  ForwardDetRingOneZ(){};

  /// Construct from iterators on Det*.
  ForwardDetRingOneZ( std::vector<const GeomDet*>::const_iterator first,
		      std::vector<const GeomDet*>::const_iterator last);

  // Construct from a std::vector of Det*.
  ForwardDetRingOneZ( const std::vector<const GeomDet*>& dets);

  virtual ~ForwardDetRingOneZ();
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}

protected:

  bool add( int idet, std::vector<DetWithState>& result,
	    const TrajectoryStateOnSurface& tsos,
	    const Propagator& prop,
 	    const MeasurementEstimator& est) const;

private:
  std::vector<const GeomDet*> theDets;

  void initialize();

};
#endif


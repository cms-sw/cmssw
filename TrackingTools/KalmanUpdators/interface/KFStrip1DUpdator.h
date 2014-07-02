#ifndef CD_KFStrip1DUpdator_H_
#define CD_KFStrip1DUpdator_H_

/** \class KFStrip1DUpdator
 *  A Kalman Updator that works in the measurement frame and uses
 *  only the X coordinate (the one perpendicular to the strip). Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

class KFStrip1DUpdator GCC11_FINAL : public TrajectoryStateUpdator {

private:
  
  typedef TrajectoryStateOnSurface TSOS;
  typedef LocalTrajectoryParameters LTP;
  typedef LocalTrajectoryError LTE;
  
public:

  KFStrip1DUpdator() {}

  ~KFStrip1DUpdator() {}

  virtual TSOS update(const TSOS& aTsos, const TrackingRecHit& aHit) const;

  virtual KFStrip1DUpdator * clone() const 
  {
    return new KFStrip1DUpdator(*this);
  }

};

#endif// CD_KFStrip1DUpdator_H_

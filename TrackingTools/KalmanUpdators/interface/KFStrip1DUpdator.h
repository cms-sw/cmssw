#ifndef CD_KFStrip1DUpdator_H_
#define CD_KFStrip1DUpdator_H_

/** \class KFStrip1DUpdator
 *  A Kalman Updator that works in the measurement frame and uses
 *  only the X coordinate (the one perpendicular to the strip). Ported from ORCA.
 *
 *  $Date: 2007/05/09 13:50:25 $
 *  $Revision: 1.3 $
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

  virtual TSOS update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const;

  virtual KFStrip1DUpdator * clone() const 
  {
    return new KFStrip1DUpdator(*this);
  }

};

#endif// CD_KFStrip1DUpdator_H_

#ifndef CD_KFStripUpdator_H_
#define CD_KFStripUpdator_H_

/** \class KFStripUpdator
 *  A Kalman Updator that works in the measurement frame and uses
 *  both hit coordinates. Ported from ORCA.
 *
 *  $Date: 2007/05/09 13:50:25 $
 *  $Revision: 1.3 $
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

class KFStripUpdator : public TrajectoryStateUpdator {

private:
  
  typedef TrajectoryStateOnSurface TSOS;
  typedef LocalTrajectoryParameters LTP;
  typedef LocalTrajectoryError LTE;
  
public:

  KFStripUpdator() {}

  ~KFStripUpdator() {}

  virtual TSOS update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const;

  virtual KFStripUpdator * clone() const 
  {
    return new KFStripUpdator(*this);
  }

};

#endif// CD_KFStripUpdator_H_

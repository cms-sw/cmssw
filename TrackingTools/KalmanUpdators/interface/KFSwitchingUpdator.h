#ifndef KFSwitchingUpdator_H_
#define KFSwitchingUpdator_H_

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFStripUpdator.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointerByClone.h"

/** A Kalman Updator that uses a KFUpdator for pixel and matched hits,
 *  and a KFStripUpdator for simple strip hits.
 */

class KFSwitchingUpdator : public TrajectoryStateUpdator {

private:
  typedef TrajectoryStateOnSurface TSOS;
  
public:

  KFSwitchingUpdator() : theLocalUpdator(new KFUpdator()),
			 theStripUpdator(new KFStripUpdator()) {}

  ~KFSwitchingUpdator() {}

  /// update with a hit
  virtual TSOS update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const;

#ifndef CMS_NO_RELAXED_RETURN_TYPE
  virtual KFSwitchingUpdator * clone() const 
#else
  virtual TrajectoryStateUpdator * clone() const 
#endif
  {
    return new KFSwitchingUpdator(*this);
  }

private:
  /// updator for 2D hits (matched or pixel)
  const KFUpdator& localUpdator() const {return *theLocalUpdator;}
  /// updator for non-matched strip hits
  const KFStripUpdator& stripUpdator() const {return *theStripUpdator;}

private:
  DeepCopyPointer<const KFUpdator> theLocalUpdator;
  DeepCopyPointer<const KFStripUpdator> theStripUpdator;

};

#endif// KFSwitchingUpdator_H_

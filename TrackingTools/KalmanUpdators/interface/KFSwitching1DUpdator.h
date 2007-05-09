#ifndef KFSwitching1DUpdator_H_
#define KFSwitching1DUpdator_H_

/** \class KFSwitching1DUpdator
 *  A Kalman Updator that uses a KFUpdator for pixel and matched hits,
 *  and a KFStrip1DUpdator for simple strip hits. Ported from ORCA.
 *
 *  $Date: 2007/05/09 13:11:43 $
 *  $Revision: 1.3.2.1 $
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFStrip1DUpdator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

class KFSwitching1DUpdator : public TrajectoryStateUpdator {

private:
  typedef TrajectoryStateOnSurface TSOS;
  
public:

  KFSwitching1DUpdator() : theLocalUpdator(new KFUpdator()),
			   theStripUpdator(new KFStrip1DUpdator()) {}

  ~KFSwitching1DUpdator() {}

  /// update with a hit
  virtual TSOS update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const;

  virtual KFSwitching1DUpdator * clone() const 
  {
    return new KFSwitching1DUpdator(*this);
  }

private:
  /// updator for 2D hits (matched or pixel)
  const KFUpdator& localUpdator() const {return *theLocalUpdator;}
  /// updator for non-matched strip hits
  const KFStrip1DUpdator& stripUpdator() const {return *theStripUpdator;}

private:
  DeepCopyPointerByClone<const KFUpdator> theLocalUpdator;
  DeepCopyPointerByClone<const KFStrip1DUpdator> theStripUpdator;

};

#endif// KFSwitching1DUpdator_H_

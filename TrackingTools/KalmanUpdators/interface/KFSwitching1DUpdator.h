#ifndef KFSwitching1DUpdator_H_
#define KFSwitching1DUpdator_H_

/** \class KFSwitching1DUpdator
 *  A Kalman Updator that uses a KFUpdator for pixel and matched hits,
 *  and a KFStrip1DUpdator for simple strip hits. Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFStrip1DUpdator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class KFSwitching1DUpdator final : public TrajectoryStateUpdator {

private:
  typedef TrajectoryStateOnSurface TSOS;
  
public:

  KFSwitching1DUpdator(const edm::ParameterSet * pset=nullptr) : theDoEndCap(false) {
    if (pset){
      theDoEndCap=pset->getParameter<bool>("doEndCap");
    }
  }
  ~KFSwitching1DUpdator() override {}

  /// update with a hit
  TSOS update(const TSOS& aTsos, const TrackingRecHit& aHit) const override;

  KFSwitching1DUpdator * clone() const override 
  {
    return new KFSwitching1DUpdator(*this);
  }

private:
  /// updator for 2D hits (matched or pixel)
  const KFUpdator& localUpdator() const {return theLocalUpdator;}
  /// updator for non-matched strip hits
  const KFStrip1DUpdator& stripUpdator() const {return theStripUpdator;}

private:
  const KFUpdator theLocalUpdator;
  const KFStrip1DUpdator theStripUpdator;

  bool theDoEndCap;
};

#endif// KFSwitching1DUpdator_H_

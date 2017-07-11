#ifndef RecTracksDistanceMatrix_H
#define RecTracksDistanceMatrix_H

#include <vector>
#include <map>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
// #include "CommonReco/PatternTools/interface/RecTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
// #include "Utilities/GenUtil/interface/ReferenceCounted.h"

/** \class RecTracksDistanceMatrix
 *  Calculates all distances between a given bunch of reco::TransientTracks at once,
 *  stores the results. CrossingPoints can optionally be calculated and 
 *  stored, as well.
 */

class RecTracksDistanceMatrix { // : public ReferenceCounted {

public:
  virtual const std::vector < reco::TransientTrack > * tracks() const = 0;
  virtual ~RecTracksDistanceMatrix() {};

  virtual double distance ( reco::TransientTrack , reco::TransientTrack ) const = 0;
  virtual double weightedDistance ( reco::TransientTrack , reco::TransientTrack ) const = 0;

  virtual GlobalPoint crossingPoint ( reco::TransientTrack , reco::TransientTrack ) const = 0;

  virtual std::pair < GlobalPoint, GlobalPoint > pointsOfClosestApproach (
              reco::TransientTrack, reco::TransientTrack ) const = 0;

  virtual bool hasDistances()      const =0;
  virtual bool hasWeightedDistances()      const =0;
  virtual bool hasCrossingPoints() const =0;
  virtual bool hasPCAs()           const =0;
};

#endif

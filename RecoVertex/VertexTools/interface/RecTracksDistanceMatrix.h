#ifndef RecTracksDistanceMatrix_H
#define RecTracksDistanceMatrix_H

#include <vector>
#include <map>
#include "Geometry/Vector/interface/GlobalPoint.h"
// #include "CommonReco/PatternTools/interface/RecTrack.h"
#include "RecoVertex/VertexPrimitives/interface/DummyRecTrack.h"
// #include "Utilities/GenUtil/interface/ReferenceCounted.h"

/** \class RecTracksDistanceMatrix
 *  Calculates all distances between a given bunch of DummyRecTracks at once,
 *  stores the results. CrossingPoints can optionally be calculated and 
 *  stored, as well.
 */

class RecTracksDistanceMatrix { // : public ReferenceCounted {

public:
  virtual const vector < DummyRecTrack > * tracks() const = 0;
  virtual ~RecTracksDistanceMatrix() {};

  virtual double distance ( const DummyRecTrack , const DummyRecTrack ) const = 0;
  virtual double weightedDistance ( const DummyRecTrack , const DummyRecTrack ) const = 0;

  virtual GlobalPoint crossingPoint ( const DummyRecTrack , const DummyRecTrack ) const = 0;

  virtual pair < GlobalPoint, GlobalPoint > pointsOfClosestApproach (
              const DummyRecTrack, const DummyRecTrack ) const = 0;

  virtual bool hasDistances()      const =0;
  virtual bool hasWeightedDistances()      const =0;
  virtual bool hasCrossingPoints() const =0;
  virtual bool hasPCAs()           const =0;
};

#endif

#ifndef DummyTrackToTrackCovCalculator_H
#define DummyTrackToTrackCovCalculator_H

#include "RecoVertex/VertexPrimitives/interface/TrackToTrackCovCalculator.h"

/** \class DummyTrackToTrackCovCalculator 
 *  Dummy TrackToTrackCovCalculator. 
 *  Returns empty TrackToTrackMap. 
 */

class CachingVertex;

class DummyTrackToTrackCovCalculator : public TrackToTrackCovCalculator {

public:

  virtual TrackToTrackMap operator() (const CachingVertex &) const;
  virtual DummyTrackToTrackCovCalculator * clone() const;

};

#endif

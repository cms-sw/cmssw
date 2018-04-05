#ifndef DummyTrackToTrackCovCalculator_H
#define DummyTrackToTrackCovCalculator_H

#include "RecoVertex/VertexPrimitives/interface/TrackToTrackCovCalculator.h"

/** \class DummyTrackToTrackCovCalculator 
 *  Dummy TrackToTrackCovCalculator. 
 *  Returns empty TrackToTrackMap. 
 */

template <unsigned int N>
class DummyTrackToTrackCovCalculator : public TrackToTrackCovCalculator<N> {

public:

  typename CachingVertex<N>::TrackToTrackMap operator() (const CachingVertex<N> &) const override;
  DummyTrackToTrackCovCalculator * clone() const override;

};

#endif

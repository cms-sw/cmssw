#include "RecoVertex/VertexTools/interface/DummyTrackToTrackCovCalculator.h"


TrackToTrackMap 
DummyTrackToTrackCovCalculator::operator() (const CachingVertex &) const
{
  return TrackToTrackMap();
}


DummyTrackToTrackCovCalculator * DummyTrackToTrackCovCalculator::clone() const
{
  return new DummyTrackToTrackCovCalculator(*this);
}


#include "TMTrackTrigger/VertexFinder/interface/VertexFinder.h"

namespace vertexFinder {

L1fittedTrack::L1fittedTrack(const TTTrack<Ref_Phase2TrackerDigi_>& aTrack) :
  track_(aTrack)
{
}

L1fittedTrack::~L1fittedTrack()
{
}

float L1fittedTrack::eta() const
{
  return track_.getMomentum().eta();
}

float L1fittedTrack::phi0() const
{
  return track_.getMomentum().phi();
}

float L1fittedTrack::pt() const
{
  return track_.getMomentum().transverse();
}

float L1fittedTrack::z0() const
{
  return track_.getPOCA().z();
}

} // end ns vertexFinder
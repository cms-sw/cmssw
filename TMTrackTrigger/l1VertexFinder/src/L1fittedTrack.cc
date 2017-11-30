
#include "TMTrackTrigger/l1VertexFinder/interface/VertexFinder.h"


#include "TMTrackTrigger/l1VertexFinder/interface/Stub.h"
#include "TMTrackTrigger/l1VertexFinder/interface/utility.h"



namespace l1tVertexFinder {

L1fittedTrack::L1fittedTrack(const TTTrack<Ref_Phase2TrackerDigi_>& aTrack, const Settings& aSettings, const TrackerGeometry*  trackerGeometry, const TrackerTopology*  trackerTopology, const std::map<edm::Ptr< TrackingParticle >, const TP* >& translateTP, edm::Handle<TTStubAssMap> mcTruthTTStubHandle, edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle) :
  track_(aTrack)
{
  std::vector<Stub*> stubs;
  for(const auto& stubRef : aTrack.getStubRefs()) {
    stubs.push_back(new Stub(stubRef, 0, &aSettings, trackerGeometry, trackerTopology) );
    stubs.back()->fillTruth(translateTP, mcTruthTTStubHandle, mcTruthTTClusterHandle);
  }
  matchedTP_ = utility::matchingTP(&aSettings, stubs, nMatchedLayers_, matchedStubs_);

  numStubs = stubs.size();
  for (const auto& stub : stubs)
  	delete stub;
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

float L1fittedTrack::chi2dof() const
{
  // FIXME: Double check nPar=4 is correct
  return track_.getChi2Red();
}

const TP* L1fittedTrack::getMatchedTP() const
{
  return matchedTP_;
}

} // end ns l1tVertexFinder

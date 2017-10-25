#ifndef TTHelper_s
#define TTHelper_s

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

namespace tthelpers{
inline reco::TransientTrack buildTT(edm::Handle<reco::TrackCollection> & tracks, edm::ESHandle<TransientTrackBuilder> &trackbuilder, unsigned int k) 
{
        reco::TrackRef ref(tracks, k);
        return trackbuilder->build(ref);
}
inline reco::TransientTrack buildTT(edm::Handle<edm::View<reco::Candidate> > & tracks, edm::ESHandle<TransientTrackBuilder> &trackbuilder, unsigned int k)
{
	if((*tracks)[k].bestTrack() == nullptr) return reco::TransientTrack();
        return trackbuilder->build(tracks->ptrAt(k));
}
}
#endif

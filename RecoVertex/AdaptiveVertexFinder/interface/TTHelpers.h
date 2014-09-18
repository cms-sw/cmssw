#ifndef TTHelper_s
#define TTHelper_s
namespace tthelpers{
inline reco::TransientTrack buildTT(edm::Handle<reco::TrackCollection> & tracks, edm::ESHandle<TransientTrackBuilder> &trackbuilder, unsigned int k) 
{
        reco::TrackRef ref(tracks, k);
        return trackbuilder->build(ref);
}
inline reco::TransientTrack buildTT(edm::Handle<edm::View<reco::Candidate> > & tracks, edm::ESHandle<TransientTrackBuilder> &trackbuilder, unsigned int k)
{
	if((*tracks)[k].bestTrack() == 0) return reco::TransientTrack();
        return trackbuilder->build(tracks->ptrAt(k));
}
}
#endif

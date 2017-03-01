#ifndef SimGeneral_TrackingAnalysis_TrackingParticleNumberOfLayers_h
#define SimGeneral_TrackingAnalysis_TrackingParticleNumberOfLayers_h

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

/**
 * This class calculates the number of tracker layers, pixel layers,
 * and strip mono+stereo layers "crossed" by TrackingParticle.
 *
 * The numbers of pixel and strip mono+stereo layers are not available
 * from TrackingParticle itself, so they are calculated here in a
 * standalone way in order to not to modify the TP dataformat (for
 * now). The number of tracker layers is available in TP, but its
 * calculation in TrackingTruthAccumulator gives wrong results (too
 * many layers) for loopers, so also it is calculated here on the same
 * go.
 *
 * The PSimHits are needed for the calculation, so, in practice, in
 * events with pileup the numbers of layers can be calculated only for
 * TPs from the signal event (i.e. not for pileup TPs). Fortunately
 * this is exactly what is sufficient for MultiTrackValidator.
 *
 * Eventually we should move to use HitPattern as in reco::TrackBase
 * (more information in a compact format), and consider adding it to
 * the TrackingParticle itself.
 *
 * In principle we could utilize the TP->SimHit map produced in
 * SimHitTPAssociationProducer instead of doing the association here
 * again, but
 * - SimTrack SimHits need to looped over in the order defined by
 *   SimTrack (to do the same as in TrackingTruthAccumulator). While
 *   possible, it would be cumbersome to do (more than just doing the
 *   association via SimTrack id)
 * - The main customer of this class, MultiTrackValidator, can in
 *   principle take a TrackingParticle collection different from the
 *   one of SimHitTPAssociationProducer (e.g. after a selection, since
 *   MTV does not support TP Refs because of how
 *   reco::RecoToSimCollection and reco::SimToRecoCollection are defined
 */
class TrackingParticleNumberOfLayers {
public:
  TrackingParticleNumberOfLayers(const edm::Event& iEvent, const std::vector<edm::EDGetTokenT<std::vector<PSimHit> > >& simHitTokens);

  enum {
    nTrackerLayers = 0,
    nPixelLayers = 1,
    nStripMonoAndStereoLayers = 2
  };
  std::tuple<std::unique_ptr<edm::ValueMap<unsigned int>>,
             std::unique_ptr<edm::ValueMap<unsigned int>>,
             std::unique_ptr<edm::ValueMap<unsigned int>>>
  calculate(const edm::Handle<TrackingParticleCollection>& tps, const edm::EventSetup& iSetup) const;

private:
  // used as multimap, but faster
  std::vector<std::pair<unsigned int, const PSimHit *>> trackIdToHitPtr_;
};

#endif

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/NanoAOD/interface/OneToManyWithQualityFlatTableProducer.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"

using TrackingParticleRecoTrackAssociationFlatTableProducer =
    OneToManyWithQualityFlatTableProducer<reco::SimToRecoCollection>;
DEFINE_FWK_MODULE(TrackingParticleRecoTrackAssociationFlatTableProducer);

using RecoTrackTrackingParticleAssociationFlatTableProducer =
    OneToManyWithQualityFlatTableProducer<reco::RecoToSimCollection>;
DEFINE_FWK_MODULE(RecoTrackTrackingParticleAssociationFlatTableProducer);

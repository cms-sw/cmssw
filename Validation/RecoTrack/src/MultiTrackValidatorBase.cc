#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"

MultiTrackValidatorBase::MultiTrackValidatorBase(const edm::ParameterSet& pset, edm::ConsumesCollector && iC, bool isSeed){
  //dbe_ = edm::Service<DQMStore>().operator->();

  associators = pset.getUntrackedParameter< std::vector<edm::InputTag> >("associators");

  const edm::InputTag& label_tp_effic_tag = pset.getParameter< edm::InputTag >("label_tp_effic");
  const edm::InputTag& label_tp_fake_tag = pset.getParameter< edm::InputTag >("label_tp_fake");

  if(pset.getParameter<bool>("label_tp_effic_refvector")) {
    label_tp_effic_refvector = iC.consumes<TrackingParticleRefVector>(label_tp_effic_tag);
  }
  else {
    label_tp_effic = iC.consumes<TrackingParticleCollection>(label_tp_effic_tag);
  }
  if(pset.getParameter<bool>("label_tp_fake_refvector")) {
    label_tp_fake_refvector = iC.consumes<TrackingParticleRefVector>(label_tp_fake_tag);
  }
  else {
    label_tp_fake = iC.consumes<TrackingParticleCollection>(label_tp_fake_tag);
  }
  label_pileupinfo = iC.consumes<std::vector<PileupSummaryInfo> >(pset.getParameter< edm::InputTag >("label_pileupinfo"));
  for(const auto& tag: pset.getParameter<std::vector<edm::InputTag>>("sim")) {
    simHitTokens_.push_back(iC.consumes<std::vector<PSimHit>>(tag));
  }

  parametersDefiner = pset.getParameter<std::string>("parametersDefiner");


  label = pset.getParameter< std::vector<edm::InputTag> >("label");
  if (isSeed) {
    for (auto& itag : label) labelTokenSeed.push_back(iC.consumes<edm::View<TrajectorySeed> >(itag));
  } else {
    for (auto& itag : label) labelToken.push_back(iC.consumes<edm::View<reco::Track> >(itag));
  }
  bsSrc = iC.consumes<reco::BeamSpot>(pset.getParameter<edm::InputTag>( "beamSpot" ));

  ignoremissingtkcollection_ = pset.getUntrackedParameter<bool>("ignoremissingtrackcollection",false);
}

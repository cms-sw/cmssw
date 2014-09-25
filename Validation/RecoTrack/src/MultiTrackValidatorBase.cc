#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"

MultiTrackValidatorBase::MultiTrackValidatorBase(const edm::ParameterSet& pset, edm::ConsumesCollector && iC, bool isSeed){
  //dbe_ = edm::Service<DQMStore>().operator->();

  associators = pset.getParameter< std::vector<std::string> >("associators");
  label_tp_effic = iC.consumes<TrackingParticleCollection>(pset.getParameter< edm::InputTag >("label_tp_effic"));
  label_tp_fake = iC.consumes<TrackingParticleCollection>(pset.getParameter< edm::InputTag >("label_tp_fake"));
  label_tv = iC.mayConsume<TrackingVertexCollection>(pset.getParameter< edm::InputTag >("label_tv"));
  label_pileupinfo = iC.consumes<std::vector<PileupSummaryInfo> >(pset.getParameter< edm::InputTag >("label_pileupinfo"));
  sim = pset.getParameter<std::string>("sim");
  parametersDefiner = pset.getParameter<std::string>("parametersDefiner");


  label = pset.getParameter< std::vector<edm::InputTag> >("label");
  if (isSeed) {
    for (auto itag : label) labelTokenSeed.push_back(iC.consumes<edm::View<TrajectorySeed> >(itag));
  } else {
    for (auto itag : label) labelToken.push_back(iC.consumes<edm::View<reco::Track> >(itag));
  }
  bsSrc = iC.consumes<reco::BeamSpot>(pset.getParameter<edm::InputTag>( "beamSpot" ));

  out = pset.getParameter<std::string>("outputFile");   

  ignoremissingtkcollection_ = pset.getUntrackedParameter<bool>("ignoremissingtrackcollection",false);
  skipHistoFit = pset.getUntrackedParameter<bool>("skipHistoFit",false);    

}

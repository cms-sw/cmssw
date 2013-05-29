#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"

MultiTrackValidatorBase::MultiTrackValidatorBase(const edm::ParameterSet& pset){
  dbe_ = edm::Service<DQMStore>().operator->();

  associators = pset.getParameter< std::vector<std::string> >("associators");
  label_tp_effic = pset.getParameter< edm::InputTag >("label_tp_effic");
  label_tp_fake = pset.getParameter< edm::InputTag >("label_tp_fake");
  sim = pset.getParameter<std::string>("sim");
  parametersDefiner = pset.getParameter<std::string>("parametersDefiner");


  label = pset.getParameter< std::vector<edm::InputTag> >("label");
  bsSrc = pset.getParameter< edm::InputTag >("beamSpot");

  out = pset.getParameter<std::string>("outputFile");   

  ignoremissingtkcollection_ = pset.getUntrackedParameter<bool>("ignoremissingtrackcollection",false);
  skipHistoFit = pset.getUntrackedParameter<bool>("skipHistoFit",false);    

}

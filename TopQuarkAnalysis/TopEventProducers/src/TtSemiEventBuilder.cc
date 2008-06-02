#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEventBuilder.h"


TtSemiEventBuilder::TtSemiEventBuilder(const edm::ParameterSet& cfg):
  hyps_(cfg.getParameter<std::vector<edm::InputTag> >("hyps")),
  keys_(cfg.getParameter<std::vector<edm::InputTag> >("keys")),
  decay_(cfg.getParameter<int>("decay")),
  genEvt_(cfg.getParameter<edm::InputTag>("genEvent")),
  genMatch_(cfg.getParameter<edm::ParameterSet>("genMatch"))
{
  // get parameter subsets for genMatch
  match_=genMatch_.getParameter<edm::InputTag>("match");
  sumPt_=genMatch_.getParameter<edm::InputTag>("sumPt");
  sumDR_=genMatch_.getParameter<edm::InputTag>("sumDR");

  // produces an TtSemiEvent from hypothesis
  // and associated extra information
  produces<TtSemiEvent>();
}

TtSemiEventBuilder::~TtSemiEventBuilder()
{
}

void
TtSemiEventBuilder::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  TtSemiEvent event;
  
  // set decay
  event.setDecay((TtSemiEvent::Decay&)decay_);

  // set genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel(genEvt_, genEvt);
  event.setGenEvent(genEvt);

  // set eventHypos
  typedef std::vector<edm::InputTag>::const_iterator EventHypo;
  for(EventHypo h=hyps_.begin(), k=keys_.begin(); h!=hyps_.end() && k!=keys_.end(); ++h, ++k){
    edm::Handle<int> key; 
    evt.getByLabel(*k, key);

    edm::Handle<reco::NamedCompositeCandidate> hyp; 
    evt.getByLabel(*h, hyp);

    event.addEventHypo((TtSemiEvent::HypoKey&)*key, *hyp);
  }

  // set extras
  edm::Handle<std::vector<int> > match;
  evt.getByLabel(match_, match);
  event.setGenMatch(*match);  

  edm::Handle<double> sumPt;
  evt.getByLabel(sumPt_, sumPt);
  event.setGenMatchSumPt(*sumPt);  

  edm::Handle<double> sumDR;
  evt.getByLabel(sumDR_, sumDR);
  event.setGenMatchSumDR(*sumDR);  

  // feed out 
  std::auto_ptr<TtSemiEvent> pOut(new TtSemiEvent);
  *pOut=event;
  evt.put(pOut);
}

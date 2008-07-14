#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEventBuilder.h"


TtSemiEventBuilder::TtSemiEventBuilder(const edm::ParameterSet& cfg) :
  hyps_    (cfg.getParameter<std::vector<edm::InputTag> >("hyps")),
  keys_    (cfg.getParameter<std::vector<edm::InputTag> >("keys")),
  matches_ (cfg.getParameter<std::vector<edm::InputTag> >("matches")),
  decay_   (cfg.getParameter<int>("decay")),
  genEvt_  (cfg.getParameter<edm::InputTag>("genEvent"))
{
  // get parameter subsets for genMatch
  if( cfg.exists("genMatch") ) {
    genMatch_= cfg.getParameter<edm::ParameterSet>("genMatch");
    sumPt_=genMatch_.getParameter<edm::InputTag>("sumPt");
    sumDR_=genMatch_.getParameter<edm::InputTag>("sumDR");
  }
  // get parameter subsets for mvaDisc
  if( cfg.exists("mvaDisc") ) {
    mvaDisc_= cfg.getParameter<edm::ParameterSet>("mvaDisc");
    meth_=mvaDisc_.getParameter<edm::InputTag>("meth");
    disc_=mvaDisc_.getParameter<edm::InputTag>("disc");
  }
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

    edm::Handle<reco::CompositeCandidate> hyp; 
    evt.getByLabel(*h, hyp);

    event.addEventHypo((TtSemiEvent::HypoKey&)*key, *hyp);
  }

  // set jetMatch extras
  for(EventHypo k=keys_.begin(), m=matches_.begin(); k!=keys_.end() && m!=matches_.end(); ++k, ++m){
    edm::Handle<int> key; 
    evt.getByLabel(*k, key);

    edm::Handle<std::vector<int> > match;
    evt.getByLabel(*m, match);

    event.addJetMatch((TtSemiEvent::HypoKey&)*key, *match);
  }

  // set genMatch extras
  edm::Handle<double> sumPt;
  evt.getByLabel(sumPt_, sumPt);
  event.setGenMatchSumPt(*sumPt);  

  edm::Handle<double> sumDR;
  evt.getByLabel(sumDR_, sumDR);
  event.setGenMatchSumDR(*sumDR);  

  // set mvaDisc extras
  edm::Handle<std::string> meth;
  evt.getByLabel(meth_, meth);

  edm::Handle<double> disc;
  evt.getByLabel(disc_, disc);
  event.setMvaDiscAndMethod((std::string&)*meth, *disc);

  // feed out 
  std::auto_ptr<TtSemiEvent> pOut(new TtSemiEvent);
  *pOut=event;
  evt.put(pOut);
}

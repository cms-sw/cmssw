#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiLepEvtBuilder.h"

#include "TString.h"

TtSemiLepEvtBuilder::TtSemiLepEvtBuilder(const edm::ParameterSet& cfg) :
  verbosity_(cfg.getParameter<int>                      ("verbosity")),
  hyps_     (cfg.getParameter<std::vector<std::string> >("hyps"     )),
  decay_    (cfg.getParameter<int>                      ("decay"    )),
  genEvt_   (cfg.getParameter<edm::InputTag>            ("genEvent" ))
{
  // get parameter subsets for kinFit
  if( cfg.exists("kinFit") ) {
    kinFit_  = cfg.getParameter<edm::ParameterSet>("kinFit");
    fitChi2_ = kinFit_.getParameter<edm::InputTag>("chi2");
    fitProb_ = kinFit_.getParameter<edm::InputTag>("prob");
  }
  // get parameter subsets for genMatch
  if( cfg.exists("genMatch") ) {
    genMatch_ = cfg.getParameter<edm::ParameterSet>("genMatch");
    sumPt_    = genMatch_.getParameter<edm::InputTag>("sumPt");
    sumDR_    = genMatch_.getParameter<edm::InputTag>("sumDR");
  }
  // get parameter subsets for mvaDisc
  if( cfg.exists("mvaDisc") ) {
    mvaDisc_ = cfg.getParameter<edm::ParameterSet>("mvaDisc");
    meth_    = mvaDisc_.getParameter<edm::InputTag>("meth");
    disc_    = mvaDisc_.getParameter<edm::InputTag>("disc");
  }
  // produces a TtSemiLeptonicEvent from hypotheses
  // and associated extra information
  produces<TtSemiLeptonicEvent>();
}

TtSemiLepEvtBuilder::~TtSemiLepEvtBuilder()
{
}

void
TtSemiLepEvtBuilder::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  TtSemiLeptonicEvent event;

  // set decay
  event.setDecay((TtSemiLeptonicEvent::Decay&)decay_);

  // set genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel(genEvt_, genEvt);
  event.setGenEvent(genEvt);

  typedef std::vector<std::string>::const_iterator EventHypo;
  for(EventHypo h=hyps_.begin(); h!=hyps_.end(); ++h){
    edm::Handle<int> key; 
    evt.getByLabel(*h, "Key", key);

    edm::Handle<std::vector<TtSemiLeptonicEvent::HypoCombPair> > hypMatchVec; 
    evt.getByLabel(*h, hypMatchVec);

    typedef std::vector<TtSemiLeptonicEvent::HypoCombPair>::const_iterator HypMatch;
    for(HypMatch hm=hypMatchVec->begin(); hm != hypMatchVec->end(); ++hm){
      event.addEventHypo((TtSemiLeptonicEvent::HypoKey&)*key, *hm);
    }
  }

  // set kinFit extras
  if( event.isHypoAvailable(TtSemiLeptonicEvent::kKinFit) ) {
    edm::Handle<std::vector<double> > fitChi2;
    evt.getByLabel(fitChi2_, fitChi2);
    event.setFitChi2( *fitChi2 );
    
    edm::Handle<std::vector<double> > fitProb;
    evt.getByLabel(fitProb_, fitProb);
    event.setFitProb( *fitProb );
  }

  // set genMatch extras
  if( event.isHypoAvailable(TtSemiLeptonicEvent::kGenMatch) ) {
    edm::Handle<std::vector<double> > sumPt;
    evt.getByLabel(sumPt_, sumPt);
    event.setGenMatchSumPt( *sumPt );

    edm::Handle<std::vector<double> > sumDR;
    evt.getByLabel(sumDR_, sumDR);
    event.setGenMatchSumDR( *sumDR );
  }

  // set mvaDisc extras
  if( event.isHypoAvailable(TtSemiLeptonicEvent::kMVADisc) ) {
    edm::Handle<TString> meth;
    evt.getByLabel(meth_, meth);
    event.setMvaMethod( (std::string) *meth );

    edm::Handle<std::vector<double> > disc;
    evt.getByLabel(disc_, disc);
    event.setMvaDiscriminators( *disc );
  }

  // print summary via MessageLogger for each event
  if(verbosity_ > 0) event.print();

  // feed out 
  std::auto_ptr<TtSemiLeptonicEvent> pOut(new TtSemiLeptonicEvent);
  *pOut=event;
  evt.put(pOut);
}

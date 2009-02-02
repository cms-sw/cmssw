#ifndef TtEvtBuilder_h
#define TtEvtBuilder_h

//#include <memory>
#include <vector>
//#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "FWCore/Framework/interface/MakerMacros.h"
//#include "TopQuarkAnalysis/TopEventProducers/interface/TtEvtBuilder.h"

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

#include "TString.h"

//#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"

template <typename C>
class TtEvtBuilder : public edm::EDProducer {

 public:

  explicit TtEvtBuilder(const edm::ParameterSet&);
  ~TtEvtBuilder(){};
  
 private:

  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  int verbosity_;

  // hypotheses
  std::vector<std::string> hyps_;

  // kinFit extras
  edm::ParameterSet kinFit_;
  edm::InputTag fitChi2_;
  edm::InputTag fitProb_;

  // gen match extras
  int lepDecayTop1_;
  int lepDecayTop2_;
  edm::InputTag genEvt_;

  edm::ParameterSet genMatch_;
  edm::InputTag sumPt_;
  edm::InputTag sumDR_;

  // mvaDisc extras
  edm::ParameterSet mvaDisc_;
  edm::InputTag meth_;
  edm::InputTag disc_;
};

template <typename C>
TtEvtBuilder<C>::TtEvtBuilder(const edm::ParameterSet& cfg) :
  verbosity_   (cfg.getParameter<int>                      ("verbosity"   )),
  hyps_        (cfg.getParameter<std::vector<std::string> >("hyps"        )),
  lepDecayTop1_(cfg.getParameter<int>                      ("lepDecayTop1")),
  lepDecayTop2_(cfg.getParameter<int>                      ("lepDecayTop2")),
  genEvt_      (cfg.getParameter<edm::InputTag>            ("genEvent"    ))
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
  // produces a TtSemiLeptonicEvent, TtFullLeptonicEvent or a
  // TtFullHadronicEvent (still to be implemented)
  // from hypotheses and associated extra information
  produces<C>();
}

template <typename C>
void
TtEvtBuilder<C>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  C event;

  // set leptonic decay channels
  event.setLepDecays( (TtEvent::LepDecay&) lepDecayTop1_ , (TtEvent::LepDecay&) lepDecayTop2_ );

  // set genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel(genEvt_, genEvt);
  event.setGenEvent(genEvt);

  typedef std::vector<std::string>::const_iterator EventHypo;
  for(EventHypo h=hyps_.begin(); h!=hyps_.end(); ++h){
    edm::Handle<int> key; 
    evt.getByLabel(*h, "Key", key);

    edm::Handle<std::vector<TtEvent::HypoCombPair> > hypMatchVec; 
    evt.getByLabel(*h, hypMatchVec);

    typedef std::vector<TtEvent::HypoCombPair>::const_iterator HypMatch;
    for(HypMatch hm=hypMatchVec->begin(); hm != hypMatchVec->end(); ++hm){
      event.addEventHypo((TtEvent::HypoClassKey&)*key, *hm);
    }
  }

  // set kinFit extras
  if( event.isHypoAvailable(TtEvent::kKinFit) ) {
    edm::Handle<std::vector<double> > fitChi2;
    evt.getByLabel(fitChi2_, fitChi2);
    event.setFitChi2( *fitChi2 );
    
    edm::Handle<std::vector<double> > fitProb;
    evt.getByLabel(fitProb_, fitProb);
    event.setFitProb( *fitProb );
  }

  // set genMatch extras
  if( event.isHypoAvailable(TtEvent::kGenMatch) ) {
    edm::Handle<std::vector<double> > sumPt;
    evt.getByLabel(sumPt_, sumPt);
    event.setGenMatchSumPt( *sumPt );

    edm::Handle<std::vector<double> > sumDR;
    evt.getByLabel(sumDR_, sumDR);
    event.setGenMatchSumDR( *sumDR );
  }

  // set mvaDisc extras
  if( event.isHypoAvailable(TtEvent::kMVADisc) ) {
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
  std::auto_ptr<C> pOut(new C);
  *pOut=event;
  evt.put(pOut);
}

#endif

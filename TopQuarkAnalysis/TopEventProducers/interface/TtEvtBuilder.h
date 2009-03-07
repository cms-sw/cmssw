#ifndef TtEvtBuilder_h
#define TtEvtBuilder_h

#include <vector>

#include "TString.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

// ----------------------------------------------------------------------
// template to fill the TtEvent structure for:
//
//  * TtSemiLeptonicEvent
//  * TtFullLeptonicEvent
//  * TtFullHadronicEvent (still to be implemented)
//
//  event hypothesis, genEvent and extra information (if 
//  available) are read from the event and contracted into 
//  the TtEvent
// ----------------------------------------------------------------------

template <typename C>
class TtEvtBuilder : public edm::EDProducer {

 public:

  /// default contructor
  explicit TtEvtBuilder(const edm::ParameterSet&);
  /// default destructor
  ~TtEvtBuilder(){};
  
 private:

  /// produce function (this one is not even accessible for
  /// derived classes)
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  /// internal event counter for verbosity switch
  int event_;
  /// determine vebosity level (0 or >0 are supported)
  int verbosity_;
  /// vector of hypothesis class names
  std::vector<std::string> hyps_;
  /// TtGenEvent
  edm::InputTag genEvt_;
  /// decay channels of the two top decay branches; to be
  /// filled according to WDecay::LeptonTypes in TtGenEvent
  int decayChnTop1_;
  int decayChnTop2_;

  /// input parameters for the kKinFit
  /// hypothesis class extras
  edm::ParameterSet kinFit_;
  edm::InputTag fitChi2_;
  edm::InputTag fitProb_;
  /// input parameters for the kGenMatch
  /// hypothesis class extras 
  edm::ParameterSet genMatch_;
  edm::InputTag sumPt_;
  edm::InputTag sumDR_;
  /// input parameters for the kMVADisc
  /// hypothesis class extras 
  edm::ParameterSet mvaDisc_;
  edm::InputTag meth_;
  edm::InputTag disc_;
};

template <typename C>
TtEvtBuilder<C>::TtEvtBuilder(const edm::ParameterSet& cfg) : event_(0),
  verbosity_   (cfg.getParameter<int>                      ("verbosity"    )),
  hyps_        (cfg.getParameter<std::vector<std::string> >("hypotheses"   )),
  genEvt_      (cfg.getParameter<edm::InputTag>            ("genEvent"     )),
  decayChnTop1_(cfg.getParameter<int>                      ("decayChannel1")),
  decayChnTop2_(cfg.getParameter<int>                      ("decayChannel2"))
{
  // parameter subsets for kKinFit
  if( cfg.exists("kinFit") ) {
    kinFit_  = cfg.getParameter<edm::ParameterSet>("kinFit");
    fitChi2_ = kinFit_.getParameter<edm::InputTag>("chi2");
    fitProb_ = kinFit_.getParameter<edm::InputTag>("prob");
  }
  // parameter subsets for kGenMatch
  if( cfg.exists("genMatch") ) {
    genMatch_ = cfg.getParameter<edm::ParameterSet>("genMatch");
    sumPt_    = genMatch_.getParameter<edm::InputTag>("sumPt");
    sumDR_    = genMatch_.getParameter<edm::InputTag>("sumDR");
  }
  // parameter subsets for kMvaDisc
  if( cfg.exists("mvaDisc") ) {
    mvaDisc_ = cfg.getParameter<edm::ParameterSet>("mvaDisc");
    meth_    = mvaDisc_.getParameter<edm::InputTag>("meth");
    disc_    = mvaDisc_.getParameter<edm::InputTag>("disc");
  }
  // produces a TtEventEvent for:
  //  * TtSemiLeptonicEvent 
  //  * TtFullLeptonicEvent
  //  * TtFullHadronicEvent (still to be implemented)
  // from hypotheses and associated extra information
  produces<C>();
}

template <typename C>
void
TtEvtBuilder<C>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  C event;

  // set leptonic decay channels
  event.setLepDecays( WDecay::LeptonType(decayChnTop1_), WDecay::LeptonType(decayChnTop2_) );

  // set genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel(genEvt_, genEvt);
  event.setGenEvent(genEvt);

  // add event hypotheses for all given 
  // hypothesis classes to the TtEvent
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

  // set kKinFit extras
  if( event.isHypoAvailable(TtEvent::kKinFit) ) {
    edm::Handle<std::vector<double> > fitChi2;
    evt.getByLabel(fitChi2_, fitChi2);
    event.setFitChi2( *fitChi2 );
    
    edm::Handle<std::vector<double> > fitProb;
    evt.getByLabel(fitProb_, fitProb);
    event.setFitProb( *fitProb );
  }

  // set kGenMatch extras
  if( event.isHypoAvailable(TtEvent::kGenMatch) ) {
    edm::Handle<std::vector<double> > sumPt;
    evt.getByLabel(sumPt_, sumPt);
    event.setGenMatchSumPt( *sumPt );

    edm::Handle<std::vector<double> > sumDR;
    evt.getByLabel(sumDR_, sumDR);
    event.setGenMatchSumDR( *sumDR );
  }

  // set kMvaDisc extras
  if( event.isHypoAvailable(TtEvent::kMVADisc) ) {
    edm::Handle<TString> meth;
    evt.getByLabel(meth_, meth);
    event.setMvaMethod( (std::string) *meth );

    edm::Handle<std::vector<double> > disc;
    evt.getByLabel(disc_, disc);
    event.setMvaDiscriminators( *disc );
  }

  // print summary via MessageLogger for up
  // to verbosity events if verbosity_>0
  if(verbosity_ > 0 && event_<verbosity_){ 
    event.print();
  }

  // write object to root file 
  std::auto_ptr<C> pOut(new C);
  *pOut=event;
  evt.put(pOut);
}

#endif

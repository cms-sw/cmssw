#ifndef TtEvtBuilder_h
#define TtEvtBuilder_h

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

/**
   \class   TtEvtBuilder TtEvtBuilder.h "TopQuarkAnalysis/TopEventProducers/interface/TtEvtBuilder.h"

   \brief    Template class to fill the TtEvent structure

   Template class to fill the TtEvent structure for:

   * TtSemiLeptonicEvent
   * TtFullLeptonicEvent
   * TtFullHadronicEvent

   event hypothesis, genEvent and extra information (if
   available) are read from the event and contracted into
   the TtEvent
*/

template <typename C>
class TtEvtBuilder : public edm::EDProducer {

 public:

  /// default constructor
  explicit TtEvtBuilder(const edm::ParameterSet&);
  /// default destructor
  ~TtEvtBuilder(){};

 private:

  /// produce function (this one is not even accessible for
  /// derived classes)
  virtual void produce(edm::Event&, const edm::EventSetup&);
  /// fill data members that are decay-channel specific
  virtual void fillSpecific(C&, const edm::Event&);

 private:

  /// vebosity level
  int verbosity_;
  /// vector of hypothesis class names
  std::vector<edm::InputTag> hyps_;
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
  /// input parameters for the kHitFit
  /// hypothesis class extras
  edm::ParameterSet hitFit_;
  edm::InputTag hitFitChi2_;
  edm::InputTag hitFitProb_;
  edm::InputTag hitFitMT_;
  edm::InputTag hitFitSigMT_;
  /// input parameters for the kKinSolution
  /// hypothesis class extras
  edm::ParameterSet kinSolution_;
  edm::InputTag solWeight_;
  edm::InputTag wrongCharge_;
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
TtEvtBuilder<C>::TtEvtBuilder(const edm::ParameterSet& cfg) :
  verbosity_   (cfg.getParameter<int>                        ("verbosity"    )),
  hyps_        (cfg.getParameter<std::vector<edm::InputTag> >("hypotheses"   )),
  genEvt_      (cfg.getParameter<edm::InputTag>              ("genEvent"     )),
  decayChnTop1_(cfg.getParameter<int>                        ("decayChannel1")),
  decayChnTop2_(cfg.getParameter<int>                        ("decayChannel2"))
{
  // parameter subsets for kKinFit
  if( cfg.exists("kinFit") ) {
    kinFit_  = cfg.getParameter<edm::ParameterSet>("kinFit");
    fitChi2_ = kinFit_.getParameter<edm::InputTag>("chi2");
    fitProb_ = kinFit_.getParameter<edm::InputTag>("prob");
  }
  // parameter subsets for kHitFit
  if( cfg.exists("hitFit") ) {
    hitFit_  = cfg.getParameter<edm::ParameterSet>("hitFit");
    hitFitChi2_ = hitFit_.getParameter<edm::InputTag>("chi2");
    hitFitProb_ = hitFit_.getParameter<edm::InputTag>("prob");
    hitFitMT_ = hitFit_.getParameter<edm::InputTag>("mt");
    hitFitSigMT_ = hitFit_.getParameter<edm::InputTag>("sigmt");
  }
  // parameter subsets for kKinSolution
  if( cfg.exists("kinSolution") ) {
    kinSolution_  = cfg.getParameter<edm::ParameterSet>("kinSolution");
    solWeight_    = kinSolution_.getParameter<edm::InputTag>("solWeight");
    wrongCharge_  = kinSolution_.getParameter<edm::InputTag>("wrongCharge");
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
  //  * TtFullHadronicEvent
  // from hypotheses and associated extra information
  produces<C>();
}

template <typename C>
void
TtEvtBuilder<C>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  C ttEvent;

  // set leptonic decay channels
  ttEvent.setLepDecays( WDecay::LeptonType(decayChnTop1_), WDecay::LeptonType(decayChnTop2_) );

  // set genEvent (if available)
  edm::Handle<TtGenEvent> genEvt;
  if ( genEvt_.label().size() > 0 )
    if( evt.getByLabel(genEvt_, genEvt) )
      ttEvent.setGenEvent(genEvt);

  // add event hypotheses for all given
  // hypothesis classes to the TtEvent
  typedef std::vector<edm::InputTag>::const_iterator EventHypo;
  for(EventHypo h=hyps_.begin(); h!=hyps_.end(); ++h){
    edm::Handle<int> key;
    evt.getByLabel(h->label(), "Key", key);

    edm::Handle<std::vector<TtEvent::HypoCombPair> > hypMatchVec;
    evt.getByLabel(*h, hypMatchVec);

    typedef std::vector<TtEvent::HypoCombPair>::const_iterator HypMatch;
    for(HypMatch hm=hypMatchVec->begin(); hm != hypMatchVec->end(); ++hm){
      ttEvent.addEventHypo((TtEvent::HypoClassKey&)*key, *hm);
    }
  }

  // set kKinFit extras
  if( ttEvent.isHypoAvailable(TtEvent::kKinFit) ) {
    edm::Handle<std::vector<double> > fitChi2;
    evt.getByLabel(fitChi2_, fitChi2);
    ttEvent.setFitChi2( *fitChi2 );

    edm::Handle<std::vector<double> > fitProb;
    evt.getByLabel(fitProb_, fitProb);
    ttEvent.setFitProb( *fitProb );
  }

  // set kHitFit extras
  if( ttEvent.isHypoAvailable(TtEvent::kHitFit) ) {
    edm::Handle<std::vector<double> > hitFitChi2;
    evt.getByLabel(hitFitChi2_, hitFitChi2);
    ttEvent.setHitFitChi2( *hitFitChi2 );

    edm::Handle<std::vector<double> > hitFitProb;
    evt.getByLabel(hitFitProb_, hitFitProb);
    ttEvent.setHitFitProb( *hitFitProb );

    edm::Handle<std::vector<double> > hitFitMT;
    evt.getByLabel(hitFitMT_, hitFitMT);
    ttEvent.setHitFitMT( *hitFitMT );

    edm::Handle<std::vector<double> > hitFitSigMT;
    evt.getByLabel(hitFitSigMT_, hitFitSigMT);
    ttEvent.setHitFitSigMT( *hitFitSigMT );
  }

  // set kGenMatch extras
  if( ttEvent.isHypoAvailable(TtEvent::kGenMatch) ) {
    edm::Handle<std::vector<double> > sumPt;
    evt.getByLabel(sumPt_, sumPt);
    ttEvent.setGenMatchSumPt( *sumPt );

    edm::Handle<std::vector<double> > sumDR;
    evt.getByLabel(sumDR_, sumDR);
    ttEvent.setGenMatchSumDR( *sumDR );
  }

  // set kMvaDisc extras
  if( ttEvent.isHypoAvailable(TtEvent::kMVADisc) ) {
    edm::Handle<std::string> meth;
    evt.getByLabel(meth_, meth);
    ttEvent.setMvaMethod( *meth );

    edm::Handle<std::vector<double> > disc;
    evt.getByLabel(disc_, disc);
    ttEvent.setMvaDiscriminators( *disc );
  }

  // fill data members that are decay-channel specific
  fillSpecific(ttEvent, evt);

  // print summary via MessageLogger if verbosity_>0
  ttEvent.print(verbosity_);

  // write object into the edm::Event
  std::auto_ptr<C> pOut(new C);
  *pOut=ttEvent;
  evt.put(pOut);
}

template <>
void TtEvtBuilder<TtFullHadronicEvent>::fillSpecific(TtFullHadronicEvent& ttEvent, const edm::Event& evt)
{
}

template <>
void TtEvtBuilder<TtFullLeptonicEvent>::fillSpecific(TtFullLeptonicEvent& ttEvent, const edm::Event& evt)
{

  // set kKinSolution extras
  if( ttEvent.isHypoAvailable(TtEvent::kKinSolution) ) {
    edm::Handle<std::vector<double> > solWeight;
    evt.getByLabel(solWeight_, solWeight);
    ttEvent.setSolWeight( *solWeight );

    edm::Handle<bool> wrongCharge;
    evt.getByLabel(wrongCharge_, wrongCharge);
    ttEvent.setWrongCharge( *wrongCharge );
  }

}

template <>
void TtEvtBuilder<TtSemiLeptonicEvent>::fillSpecific(TtSemiLeptonicEvent& ttEvent, const edm::Event& evt)
{

  typedef std::vector<edm::InputTag>::const_iterator EventHypo;
  for(EventHypo h=hyps_.begin(); h!=hyps_.end(); ++h){
    edm::Handle<int> key;
    evt.getByLabel(h->label(), "Key", key);

    // set number of real neutrino solutions for all hypotheses
    edm::Handle<int> numberOfRealNeutrinoSolutions;
    evt.getByLabel(h->label(), "NumberOfRealNeutrinoSolutions", numberOfRealNeutrinoSolutions);
    ttEvent.setNumberOfRealNeutrinoSolutions((TtEvent::HypoClassKey&)*key, *numberOfRealNeutrinoSolutions);

    // set number of considered jets for all hypotheses
    edm::Handle<int> numberOfConsideredJets;
    evt.getByLabel(h->label(), "NumberOfConsideredJets", numberOfConsideredJets);
    ttEvent.setNumberOfConsideredJets((TtEvent::HypoClassKey&)*key, *numberOfConsideredJets);
  }

}

#endif

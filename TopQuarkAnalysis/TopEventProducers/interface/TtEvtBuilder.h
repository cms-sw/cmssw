#ifndef TtEvtBuilder_h
#define TtEvtBuilder_h

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadronicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"

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
  ~TtEvtBuilder() override{};

private:
  /// produce function (this one is not even accessible for
  /// derived classes)
  void produce(edm::Event&, const edm::EventSetup&) override;
  /// fill data members that are decay-channel specific
  virtual void fillSpecific(C&, const edm::Event&);

private:
  /// vebosity level
  int verbosity_;
  /// vector of hypothesis class names
  std::vector<edm::EDGetTokenT<int> > hypKeyTokens_;
  std::vector<edm::EDGetTokenT<std::vector<TtEvent::HypoCombPair> > > hypTokens_;
  std::vector<edm::EDGetTokenT<int> > hypNeutrTokens_;
  std::vector<edm::EDGetTokenT<int> > hypJetTokens_;
  typedef std::vector<edm::EDGetTokenT<int> >::const_iterator EventHypoIntToken;
  typedef std::vector<edm::EDGetTokenT<std::vector<TtEvent::HypoCombPair> > >::const_iterator EventHypoToken;
  /// TtGenEvent
  edm::InputTag genEvt_;
  edm::EDGetTokenT<TtGenEvent> genEvtToken_;
  /// decay channels of the two top decay branches; to be
  /// filled according to WDecay::LeptonTypes in TtGenEvent
  int decayChnTop1_;
  int decayChnTop2_;

  /// input parameters for the kKinFit
  /// hypothesis class extras
  edm::ParameterSet kinFit_;
  edm::EDGetTokenT<std::vector<double> > fitChi2Token_;
  edm::EDGetTokenT<std::vector<double> > fitProbToken_;
  /// input parameters for the kHitFit
  /// hypothesis class extras
  edm::ParameterSet hitFit_;
  edm::EDGetTokenT<std::vector<double> > hitFitChi2Token_;
  edm::EDGetTokenT<std::vector<double> > hitFitProbToken_;
  edm::EDGetTokenT<std::vector<double> > hitFitMTToken_;
  edm::EDGetTokenT<std::vector<double> > hitFitSigMTToken_;
  /// input parameters for the kKinSolution
  /// hypothesis class extras
  edm::ParameterSet kinSolution_;
  edm::EDGetTokenT<std::vector<double> > solWeightToken_;
  edm::EDGetTokenT<bool> wrongChargeToken_;
  /// input parameters for the kGenMatch
  /// hypothesis class extras
  edm::ParameterSet genMatch_;
  edm::EDGetTokenT<std::vector<double> > sumPtToken_;
  edm::EDGetTokenT<std::vector<double> > sumDRToken_;
  /// input parameters for the kMVADisc
  /// hypothesis class extras
  edm::ParameterSet mvaDisc_;
  edm::EDGetTokenT<std::string> methToken_;
  edm::EDGetTokenT<std::vector<double> > discToken_;
};

template <typename C>
TtEvtBuilder<C>::TtEvtBuilder(const edm::ParameterSet& cfg)
    : verbosity_(cfg.getParameter<int>("verbosity")),
      hypKeyTokens_(edm::vector_transform(
          cfg.getParameter<std::vector<edm::InputTag> >("hypotheses"),
          [this](edm::InputTag const& tag) { return consumes<int>(edm::InputTag(tag.label(), "Key")); })),
      hypTokens_(edm::vector_transform(
          cfg.getParameter<std::vector<edm::InputTag> >("hypotheses"),
          [this](edm::InputTag const& tag) { return consumes<std::vector<TtEvent::HypoCombPair> >(tag); })),
      hypNeutrTokens_(edm::vector_transform(cfg.getParameter<std::vector<edm::InputTag> >("hypotheses"),
                                            [this](edm::InputTag const& tag) {
                                              return consumes<int>(
                                                  edm::InputTag(tag.label(), "NumberOfRealNeutrinoSolutions"));
                                            })),
      hypJetTokens_(edm::vector_transform(cfg.getParameter<std::vector<edm::InputTag> >("hypotheses"),
                                          [this](edm::InputTag const& tag) {
                                            return consumes<int>(edm::InputTag(tag.label(), "NumberOfConsideredJets"));
                                          })),
      genEvt_(cfg.getParameter<edm::InputTag>("genEvent")),
      genEvtToken_(mayConsume<TtGenEvent>(genEvt_)),
      decayChnTop1_(cfg.getParameter<int>("decayChannel1")),
      decayChnTop2_(cfg.getParameter<int>("decayChannel2")) {
  // parameter subsets for kKinFit
  if (cfg.exists("kinFit")) {
    kinFit_ = cfg.getParameter<edm::ParameterSet>("kinFit");
    fitChi2Token_ = mayConsume<std::vector<double> >(kinFit_.getParameter<edm::InputTag>("chi2"));
    fitProbToken_ = mayConsume<std::vector<double> >(kinFit_.getParameter<edm::InputTag>("prob"));
  }
  // parameter subsets for kHitFit
  if (cfg.exists("hitFit")) {
    hitFit_ = cfg.getParameter<edm::ParameterSet>("hitFit");
    hitFitChi2Token_ = mayConsume<std::vector<double> >(hitFit_.getParameter<edm::InputTag>("chi2"));
    hitFitProbToken_ = mayConsume<std::vector<double> >(hitFit_.getParameter<edm::InputTag>("prob"));
    hitFitMTToken_ = mayConsume<std::vector<double> >(hitFit_.getParameter<edm::InputTag>("mt"));
    hitFitSigMTToken_ = mayConsume<std::vector<double> >(hitFit_.getParameter<edm::InputTag>("sigmt"));
  }
  // parameter subsets for kKinSolution
  if (cfg.exists("kinSolution")) {
    kinSolution_ = cfg.getParameter<edm::ParameterSet>("kinSolution");
    solWeightToken_ = mayConsume<std::vector<double> >(kinSolution_.getParameter<edm::InputTag>("solWeight"));
    wrongChargeToken_ = mayConsume<bool>(kinSolution_.getParameter<edm::InputTag>("wrongCharge"));
  }
  // parameter subsets for kGenMatch
  if (cfg.exists("genMatch")) {
    genMatch_ = cfg.getParameter<edm::ParameterSet>("genMatch");
    sumPtToken_ = mayConsume<std::vector<double> >(genMatch_.getParameter<edm::InputTag>("sumPt"));
    sumDRToken_ = mayConsume<std::vector<double> >(genMatch_.getParameter<edm::InputTag>("sumDR"));
  }
  // parameter subsets for kMvaDisc
  if (cfg.exists("mvaDisc")) {
    mvaDisc_ = cfg.getParameter<edm::ParameterSet>("mvaDisc");
    methToken_ = mayConsume<std::string>(mvaDisc_.getParameter<edm::InputTag>("meth"));
    discToken_ = mayConsume<std::vector<double> >(mvaDisc_.getParameter<edm::InputTag>("disc"));
  }
  // produces a TtEventEvent for:
  //  * TtSemiLeptonicEvent
  //  * TtFullLeptonicEvent
  //  * TtFullHadronicEvent
  // from hypotheses and associated extra information
  produces<C>();
}

template <typename C>
void TtEvtBuilder<C>::produce(edm::Event& evt, const edm::EventSetup& setup) {
  C ttEvent;

  // set leptonic decay channels
  ttEvent.setLepDecays(WDecay::LeptonType(decayChnTop1_), WDecay::LeptonType(decayChnTop2_));

  // set genEvent (if available)
  edm::Handle<TtGenEvent> genEvt;
  if (!genEvt_.label().empty())
    if (evt.getByToken(genEvtToken_, genEvt))
      ttEvent.setGenEvent(genEvt);

  // add event hypotheses for all given
  // hypothesis classes to the TtEvent
  EventHypoIntToken hKey = hypKeyTokens_.begin();
  EventHypoToken h = hypTokens_.begin();
  for (; hKey != hypKeyTokens_.end(); ++hKey, ++h) {
    edm::Handle<int> key;
    evt.getByToken(*hKey, key);

    edm::Handle<std::vector<TtEvent::HypoCombPair> > hypMatchVec;
    evt.getByToken(*h, hypMatchVec);

    typedef std::vector<TtEvent::HypoCombPair>::const_iterator HypMatch;
    for (HypMatch hm = hypMatchVec->begin(); hm != hypMatchVec->end(); ++hm) {
      ttEvent.addEventHypo((TtEvent::HypoClassKey&)*key, *hm);
    }
  }

  // set kKinFit extras
  if (ttEvent.isHypoAvailable(TtEvent::kKinFit)) {
    edm::Handle<std::vector<double> > fitChi2;
    evt.getByToken(fitChi2Token_, fitChi2);
    ttEvent.setFitChi2(*fitChi2);

    edm::Handle<std::vector<double> > fitProb;
    evt.getByToken(fitProbToken_, fitProb);
    ttEvent.setFitProb(*fitProb);
  }

  // set kHitFit extras
  if (ttEvent.isHypoAvailable(TtEvent::kHitFit)) {
    edm::Handle<std::vector<double> > hitFitChi2;
    evt.getByToken(hitFitChi2Token_, hitFitChi2);
    ttEvent.setHitFitChi2(*hitFitChi2);

    edm::Handle<std::vector<double> > hitFitProb;
    evt.getByToken(hitFitProbToken_, hitFitProb);
    ttEvent.setHitFitProb(*hitFitProb);

    edm::Handle<std::vector<double> > hitFitMT;
    evt.getByToken(hitFitMTToken_, hitFitMT);
    ttEvent.setHitFitMT(*hitFitMT);

    edm::Handle<std::vector<double> > hitFitSigMT;
    evt.getByToken(hitFitSigMTToken_, hitFitSigMT);
    ttEvent.setHitFitSigMT(*hitFitSigMT);
  }

  // set kGenMatch extras
  if (ttEvent.isHypoAvailable(TtEvent::kGenMatch)) {
    edm::Handle<std::vector<double> > sumPt;
    evt.getByToken(sumPtToken_, sumPt);
    ttEvent.setGenMatchSumPt(*sumPt);

    edm::Handle<std::vector<double> > sumDR;
    evt.getByToken(sumDRToken_, sumDR);
    ttEvent.setGenMatchSumDR(*sumDR);
  }

  // set kMvaDisc extras
  if (ttEvent.isHypoAvailable(TtEvent::kMVADisc)) {
    edm::Handle<std::string> meth;
    evt.getByToken(methToken_, meth);
    ttEvent.setMvaMethod(*meth);

    edm::Handle<std::vector<double> > disc;
    evt.getByToken(discToken_, disc);
    ttEvent.setMvaDiscriminators(*disc);
  }

  // fill data members that are decay-channel specific
  fillSpecific(ttEvent, evt);

  // print summary via MessageLogger if verbosity_>0
  ttEvent.print(verbosity_);

  // write object into the edm::Event
  std::unique_ptr<C> pOut(new C);
  *pOut = ttEvent;
  evt.put(std::move(pOut));
}

template <>
inline void TtEvtBuilder<TtFullHadronicEvent>::fillSpecific(TtFullHadronicEvent& ttEvent, const edm::Event& evt) {}

template <>
inline void TtEvtBuilder<TtFullLeptonicEvent>::fillSpecific(TtFullLeptonicEvent& ttEvent, const edm::Event& evt) {
  // set kKinSolution extras
  if (ttEvent.isHypoAvailable(TtEvent::kKinSolution)) {
    edm::Handle<std::vector<double> > solWeight;
    evt.getByToken(solWeightToken_, solWeight);
    ttEvent.setSolWeight(*solWeight);

    edm::Handle<bool> wrongCharge;
    evt.getByToken(wrongChargeToken_, wrongCharge);
    ttEvent.setWrongCharge(*wrongCharge);
  }
}

template <>
inline void TtEvtBuilder<TtSemiLeptonicEvent>::fillSpecific(TtSemiLeptonicEvent& ttEvent, const edm::Event& evt) {
  EventHypoIntToken hKey = hypKeyTokens_.begin();
  EventHypoIntToken hNeutr = hypNeutrTokens_.begin();
  EventHypoIntToken hJet = hypJetTokens_.begin();
  for (; hKey != hypKeyTokens_.end(); ++hKey, ++hNeutr, ++hJet) {
    edm::Handle<int> key;
    evt.getByToken(*hKey, key);

    // set number of real neutrino solutions for all hypotheses
    edm::Handle<int> numberOfRealNeutrinoSolutions;
    evt.getByToken(*hNeutr, numberOfRealNeutrinoSolutions);
    ttEvent.setNumberOfRealNeutrinoSolutions((TtEvent::HypoClassKey&)*key, *numberOfRealNeutrinoSolutions);

    // set number of considered jets for all hypotheses
    edm::Handle<int> numberOfConsideredJets;
    evt.getByToken(*hJet, numberOfConsideredJets);
    ttEvent.setNumberOfConsideredJets((TtEvent::HypoClassKey&)*key, *numberOfConsideredJets);
  }
}

#endif

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
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
class TtEvtBuilder : public edm::global::EDProducer<> {
public:
  /// default constructor
  explicit TtEvtBuilder(const edm::ParameterSet&);

private:
  /// produce function (this one is not even accessible for
  /// derived classes)
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  /// fill data members that are decay-channel specific
  void fillSpecific(C&, const edm::Event&) const;

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

  edm::EDPutTokenT<C> putToken_;
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
  putToken_ = produces<C>();
}

template <typename C>
void TtEvtBuilder<C>::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& setup) const {
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
    const int key = evt.get(*hKey);

    const std::vector<TtEvent::HypoCombPair>& hypMatchVec = evt.get(*h);

    for (const auto& hm : hypMatchVec) {
      ttEvent.addEventHypo(static_cast<TtEvent::HypoClassKey>(key), hm);
    }
  }

  // set kKinFit extras
  if (ttEvent.isHypoAvailable(TtEvent::kKinFit)) {
    ttEvent.setFitChi2(evt.get(fitChi2Token_));
    ttEvent.setFitProb(evt.get(fitProbToken_));
  }

  // set kHitFit extras
  if (ttEvent.isHypoAvailable(TtEvent::kHitFit)) {
    ttEvent.setHitFitChi2(evt.get(hitFitChi2Token_));
    ttEvent.setHitFitProb(evt.get(hitFitProbToken_));
    ttEvent.setHitFitMT(evt.get(hitFitMTToken_));
    ttEvent.setHitFitSigMT(evt.get(hitFitSigMTToken_));
  }

  // set kGenMatch extras
  if (ttEvent.isHypoAvailable(TtEvent::kGenMatch)) {
    ttEvent.setGenMatchSumPt(evt.get(sumPtToken_));
    ttEvent.setGenMatchSumDR(evt.get(sumDRToken_));
  }

  // set kMvaDisc extras
  if (ttEvent.isHypoAvailable(TtEvent::kMVADisc)) {
    ttEvent.setMvaMethod(evt.get(methToken_));
    ttEvent.setMvaDiscriminators(evt.get(discToken_));
  }

  // fill data members that are decay-channel specific
  fillSpecific(ttEvent, evt);

  // print summary via MessageLogger if verbosity_>0
  ttEvent.print(verbosity_);

  // write object into the edm::Event
  evt.emplace(putToken_, std::move(ttEvent));
}

template <>
inline void TtEvtBuilder<TtFullHadronicEvent>::fillSpecific(TtFullHadronicEvent& ttEvent, const edm::Event& evt) const {
}

template <>
inline void TtEvtBuilder<TtFullLeptonicEvent>::fillSpecific(TtFullLeptonicEvent& ttEvent, const edm::Event& evt) const {
  // set kKinSolution extras
  if (ttEvent.isHypoAvailable(TtEvent::kKinSolution)) {
    ttEvent.setSolWeight(evt.get(solWeightToken_));
    ttEvent.setWrongCharge(evt.get(wrongChargeToken_));
  }
}

template <>
inline void TtEvtBuilder<TtSemiLeptonicEvent>::fillSpecific(TtSemiLeptonicEvent& ttEvent, const edm::Event& evt) const {
  EventHypoIntToken hKey = hypKeyTokens_.begin();
  EventHypoIntToken hNeutr = hypNeutrTokens_.begin();
  EventHypoIntToken hJet = hypJetTokens_.begin();
  for (; hKey != hypKeyTokens_.end(); ++hKey, ++hNeutr, ++hJet) {
    const int key = evt.get(*hKey);

    // set number of real neutrino solutions for all hypotheses
    const int numberOfRealNeutrinoSolutions = evt.get(*hNeutr);
    ttEvent.setNumberOfRealNeutrinoSolutions(static_cast<TtEvent::HypoClassKey>(key), numberOfRealNeutrinoSolutions);

    // set number of considered jets for all hypotheses
    const int numberOfConsideredJets = evt.get(*hJet);
    ttEvent.setNumberOfConsideredJets(static_cast<TtEvent::HypoClassKey>(key), numberOfConsideredJets);
  }
}

#include "AnalysisDataFormats/TopObjects/interface/TtFullHadronicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
using TtFullHadEvtBuilder = TtEvtBuilder<TtFullHadronicEvent>;
using TtFullLepEvtBuilder = TtEvtBuilder<TtFullLeptonicEvent>;
using TtSemiLepEvtBuilder = TtEvtBuilder<TtSemiLeptonicEvent>;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtFullHadEvtBuilder);
DEFINE_FWK_MODULE(TtFullLepEvtBuilder);
DEFINE_FWK_MODULE(TtSemiLepEvtBuilder);

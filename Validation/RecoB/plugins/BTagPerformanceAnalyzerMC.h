#ifndef BTagPerformanceAnalyzerMC_H
#define BTagPerformanceAnalyzerMC_H

#include "DQMOffline/RecoB/interface/AcceptJet.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/MatchJet.h"
#include "DQMOffline/RecoB/interface/TagCorrelationPlotter.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/JetMatching/interface/JetFlavour.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "DataFormats/JetMatching/interface/JetFlavourMatching.h"
/** \class BTagPerformanceAnalyzerMC
 *
 *  Top level steering routine for b tag performance analysis.
 *
 */

class BTagPerformanceAnalyzerMC : public DQMEDAnalyzer {
public:
  explicit BTagPerformanceAnalyzerMC(const edm::ParameterSet &pSet);

  ~BTagPerformanceAnalyzerMC() override;

  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  struct JetRefCompare {
    inline bool operator()(const edm::RefToBase<reco::Jet> &j1, const edm::RefToBase<reco::Jet> &j2) const {
      return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key());
    }
  };

  // Get histogram plotting options from configuration.
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  EtaPtBin getEtaPtBin(const int &iEta, const int &iPt);

  typedef std::pair<reco::Jet, reco::JetFlavourInfo> JetWithFlavour;
  typedef std::map<edm::RefToBase<reco::Jet>, unsigned int, JetRefCompare> FlavourMap;
  typedef std::map<edm::RefToBase<reco::Jet>, reco::JetFlavour::Leptons, JetRefCompare> LeptonMap;

  bool getJetWithFlavour(const edm::Event &iEvent,
                         edm::RefToBase<reco::Jet> caloRef,
                         const FlavourMap &_flavours,
                         JetWithFlavour &jetWithFlavour,
                         const reco::JetCorrector *corrector,
                         edm::Handle<edm::Association<reco::GenJetCollection>> genJetsMatched);
  bool getJetWithGenJet(edm::RefToBase<reco::Jet> jetRef,
                        edm::Handle<edm::Association<reco::GenJetCollection>> genJetsMatched);

  std::vector<std::string> tiDataFormatType;
  AcceptJet jetSelector;  // Decides if jet and parton satisfy kinematic cuts.
  std::vector<double> etaRanges, ptRanges;
  bool useOldFlavourTool;
  bool doJEC;

  bool ptHatWeight;

  edm::InputTag jetMCSrc;
  edm::InputTag slInfoTag;
  edm::InputTag genJetsMatchedSrc;

  std::vector<std::vector<std::unique_ptr<JetTagPlotter>>> binJetTagPlotters;
  std::vector<std::vector<std::unique_ptr<TagCorrelationPlotter>>> binTagCorrelationPlotters;
  std::vector<std::vector<std::unique_ptr<BaseTagInfoPlotter>>> binTagInfoPlotters;
  std::vector<edm::InputTag> jetTagInputTags;
  std::vector<std::pair<edm::InputTag, edm::InputTag>> tagCorrelationInputTags;
  std::vector<std::vector<edm::InputTag>> tagInfoInputTags;
  //  JetFlavourIdentifier jfi;
  std::vector<edm::ParameterSet> moduleConfig;

  std::string flavPlots_;
  unsigned int mcPlots_;

  CorrectJet jetCorrector;
  MatchJet jetMatcher;

  bool doPUid;
  bool eventInitialized;
  bool electronPlots, muonPlots, tauPlots;

  // add consumes
  edm::EDGetTokenT<GenEventInfoProduct> genToken;
  edm::EDGetTokenT<edm::Association<reco::GenJetCollection>> genJetsMatchedToken;
  edm::EDGetTokenT<reco::JetCorrector> jecMCToken;
  edm::EDGetTokenT<reco::JetCorrector> jecDataToken;
  edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> jetToken;
  edm::EDGetTokenT<reco::JetFlavourMatchingCollection> caloJetToken;
  edm::EDGetTokenT<reco::SoftLeptonTagInfoCollection> slInfoToken;
  std::vector<edm::EDGetTokenT<reco::JetTagCollection>> jetTagToken;
  std::vector<std::pair<edm::EDGetTokenT<reco::JetTagCollection>, edm::EDGetTokenT<reco::JetTagCollection>>>
      tagCorrelationToken;
  std::vector<std::vector<edm::EDGetTokenT<edm::View<reco::BaseTagInfo>>>> tagInfoToken;
};

#endif

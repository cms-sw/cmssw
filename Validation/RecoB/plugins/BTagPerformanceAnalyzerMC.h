#ifndef BTagPerformanceAnalyzerMC_H
#define BTagPerformanceAnalyzerMC_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DQMOffline/RecoB/interface/AcceptJet.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/TagCorrelationPlotter.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "DQMOffline/RecoB/interface/MatchJet.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
/** \class BTagPerformanceAnalyzerMC
 *
 *  Top level steering routine for b tag performance analysis.
 *
 */

class BTagPerformanceAnalyzerMC : public DQMEDAnalyzer {
   public:
      explicit BTagPerformanceAnalyzerMC(const edm::ParameterSet& pSet);

      ~BTagPerformanceAnalyzerMC();

      virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

   private:

  struct JetRefCompare :
       public std::binary_function<edm::RefToBase<reco::Jet>, edm::RefToBase<reco::Jet>, bool> {
    inline bool operator () (const edm::RefToBase<reco::Jet> &j1,
                             const edm::RefToBase<reco::Jet> &j2) const
    { return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key()); }
  };

  // Get histogram plotting options from configuration.
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;     

  EtaPtBin getEtaPtBin(const int& iEta, const int& iPt);

  typedef std::pair<reco::Jet, reco::JetFlavourInfo> JetWithFlavour;
  typedef std::map<edm::RefToBase<reco::Jet>, unsigned int, JetRefCompare> FlavourMap;
  typedef std::map<edm::RefToBase<reco::Jet>, reco::JetFlavour::Leptons, JetRefCompare> LeptonMap;
  
  bool getJetWithFlavour(const edm::Event& iEvent,
			 edm::RefToBase<reco::Jet> caloRef,
                         const FlavourMap& _flavours, JetWithFlavour &jetWithFlavour,
			 const reco::JetCorrector * corrector, 
			 edm::Handle<edm::Association<reco::GenJetCollection> > genJetsMatched);
  bool getJetWithGenJet(edm::RefToBase<reco::Jet> jetRef, edm::Handle<edm::Association<reco::GenJetCollection> > genJetsMatched); 

  std::vector<std::string> tiDataFormatType;
  AcceptJet jetSelector;   // Decides if jet and parton satisfy kinematic cuts.
  std::vector<double> etaRanges, ptRanges;
  bool useOldFlavourTool;
  bool doJEC;

  bool ptHatWeight;

  edm::InputTag jetMCSrc;
  edm::InputTag slInfoTag;
  edm::InputTag genJetsMatchedSrc;

  std::vector< std::vector<JetTagPlotter*> > binJetTagPlotters;
  std::vector< std::vector<TagCorrelationPlotter*> > binTagCorrelationPlotters;
  std::vector< std::vector<BaseTagInfoPlotter*> > binTagInfoPlotters;
  std::vector<edm::InputTag> jetTagInputTags;
  std::vector< std::pair<edm::InputTag, edm::InputTag> > tagCorrelationInputTags;
  std::vector< std::vector<edm::InputTag> > tagInfoInputTags;
  //  JetFlavourIdentifier jfi;
  std::vector<edm::ParameterSet> moduleConfig;
  std::map<BaseTagInfoPlotter*, size_t> binTagInfoPlottersToModuleConfig;

  std::string flavPlots_;
  unsigned int mcPlots_;

  CorrectJet jetCorrector;
  MatchJet jetMatcher;

  bool doPUid; 
  bool eventInitialized;
  bool electronPlots, muonPlots, tauPlots;

  //add consumes 
  edm::EDGetTokenT<GenEventInfoProduct> genToken;
  edm::EDGetTokenT<edm::Association<reco::GenJetCollection>> genJetsMatchedToken;
  edm::EDGetTokenT<reco::JetCorrector> jecMCToken;
  edm::EDGetTokenT<reco::JetCorrector> jecDataToken;
  edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> jetToken;
  edm::EDGetTokenT<reco::JetFlavourMatchingCollection> caloJetToken;
  edm::EDGetTokenT<reco::SoftLeptonTagInfoCollection> slInfoToken;
  std::vector< edm::EDGetTokenT<reco::JetTagCollection> > jetTagToken;
  std::vector< std::pair<edm::EDGetTokenT<reco::JetTagCollection>, edm::EDGetTokenT<reco::JetTagCollection>> > tagCorrelationToken;
  std::vector<std::vector <edm::EDGetTokenT<edm::View<reco::BaseTagInfo>> >> tagInfoToken;
};


#endif

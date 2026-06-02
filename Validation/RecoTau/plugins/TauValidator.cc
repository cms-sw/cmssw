// Analyzer for validation histograms for tau objects at HLT/RECO
// E. Vernazza Apr. 10, 2026

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>
#include <format>

using namespace edm;
using namespace reco;
using namespace std;

class TauValidator : public DQMEDAnalyzer {

public:
  TauValidator(const edm::ParameterSet &);
  ~TauValidator() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:

  edm::EDGetTokenT<reco::PFTauCollection> recoTauToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genTauToken_;

  const std::unordered_map<std::string, std::tuple<unsigned, float, float>> histoVars = {
    {"pt", std::make_tuple(200, 0., 1000.)},
    {"eta", std::make_tuple(60, -4.0, 4.0)},
    {"phi", std::make_tuple(50, -3.5, 3.5)},
    {"mass", std::make_tuple(200, 0, 10.)},
  };

  using UMap = std::unordered_map<std::string, MonitorElement*>;
  UMap h_recoTau_;
  UMap h_recoTauMatched_;
  UMap h_recoTauMultiMatched_;
  UMap h_genTau_;
  UMap h_genTauMatched_;
  UMap h_genTauMultiMatched_;

  bool isHLT;
  float matchingDeltaR;
  std::string outFolder_;

};

TauValidator::TauValidator(const edm::ParameterSet& iConfig) {
  recoTauToken_ = consumes<reco::PFTauCollection>(iConfig.getParameter<edm::InputTag>("recoTauCollection"));
  genTauToken_ = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("genTauCollection"));
  matchingDeltaR = iConfig.getParameter<double>("minDeltaR");
  outFolder_ = iConfig.getParameter<std::string>("outFolder");
  isHLT = iConfig.getUntrackedParameter<bool>("isHLT");
}

void TauValidator::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {

  // ---------------------------- Book Summary Histograms -------------------------------
  ibooker.setCurrentFolder(outFolder_);
  
  for (auto& hVar : histoVars) {
    auto [nBins, hMin, hMax] = hVar.second;
    h_recoTau_[hVar.first] = ibooker.book1D("recoTau_" + hVar.first, ";#tau^{reco};" + hVar.first, nBins, hMin, hMax);
    h_recoTauMatched_[hVar.first] = ibooker.book1D("recoTauMatched_" + hVar.first, ";#tau^{reco} (Matched);" + hVar.first, nBins, hMin, hMax);
    h_recoTauMultiMatched_[hVar.first] = ibooker.book1D("recoTauMultiMatched_" + hVar.first, ";#tau^{reco} (Multi-Matched);" + hVar.first, nBins, hMin, hMax);
    h_genTau_[hVar.first] = ibooker.book1D("genTau_" + hVar.first, "#tau^{gen};" + hVar.first, nBins, hMin, hMax);
    h_genTauMatched_[hVar.first] = ibooker.book1D("genTauMatched_" + hVar.first, ";#tau^{gen} (Matched);" + hVar.first, nBins, hMin, hMax);
    h_genTauMultiMatched_[hVar.first] = ibooker.book1D("genTauMultiMatched_" + hVar.first, ";#tau^{gen} (Multi-Matched);" + hVar.first, nBins, hMin, hMax);
  }

}

//------------------------------------------------------------------------------
// ~TauValidator
//------------------------------------------------------------------------------
TauValidator::~TauValidator() = default;

//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void TauValidator::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup) {

  edm::Handle<reco::PFTauCollection> recoTaus;
  mEvent.getByToken(recoTauToken_, recoTaus);
  if (!recoTaus.isValid()) {
    edm::LogPrint("TauValidator") << " Reco Tau collection not found while running TauValidator.cc ";
    return;
  }
  // std::cout << "Number of reco taus: " << recoTaus->size() << std::endl; // [DEBUG]

  edm::Handle<reco::GenJetCollection> genTaus;
  mEvent.getByToken(genTauToken_, genTaus);
  if (!genTaus.isValid()) {
    edm::LogPrint("TauValidator") << " Gen Tau collection not found while running TauValidator.cc ";
    return;
  }
  // std::cout << "Number of gen taus: " << genTaus->size() << std::endl; // [DEBUG]

  // Loop for efficiency 
  for (uint itau = 0; itau < genTaus->size(); ++itau) {
    
    h_genTau_["pt"]->Fill(genTaus->at(itau).pt());
    h_genTau_["eta"]->Fill(genTaus->at(itau).eta());
    h_genTau_["phi"]->Fill(genTaus->at(itau).phi());
    h_genTau_["mass"]->Fill(genTaus->at(itau).mass());
    
    // Count how many reco taus are matched to the gen tau
    int nRecoMatchedToOneGen = 0;
    for (uint jtau = 0; jtau < recoTaus->size(); ++jtau) {
      if (deltaR(genTaus->at(itau), recoTaus->at(jtau)) < matchingDeltaR) {
        nRecoMatchedToOneGen++;
      }
    }

    // Fill histograms for gen taus matched to at least one reco tau
    if (nRecoMatchedToOneGen > 0) {
      h_genTauMatched_["pt"]->Fill(genTaus->at(itau).pt());
      h_genTauMatched_["eta"]->Fill(genTaus->at(itau).eta());
      h_genTauMatched_["phi"]->Fill(genTaus->at(itau).phi());
      h_genTauMatched_["mass"]->Fill(genTaus->at(itau).mass());
      if (nRecoMatchedToOneGen > 1) {
        h_genTauMultiMatched_["pt"]->Fill(genTaus->at(itau).pt());
        h_genTauMultiMatched_["eta"]->Fill(genTaus->at(itau).eta());
        h_genTauMultiMatched_["phi"]->Fill(genTaus->at(itau).phi());
        h_genTauMultiMatched_["mass"]->Fill(genTaus->at(itau).mass());
      }
    }
  }

  // Loop for fake rate
  for (uint itau = 0; itau < recoTaus->size(); ++itau) {
    h_recoTau_["pt"]->Fill(recoTaus->at(itau).pt());
    h_recoTau_["eta"]->Fill(recoTaus->at(itau).eta());
    h_recoTau_["phi"]->Fill(recoTaus->at(itau).phi());
    h_recoTau_["mass"]->Fill(recoTaus->at(itau).mass());

    // Count how many gen taus are matched to the reco tau
    int nGenMatchedToOneReco = 0;
    for (uint jtau = 0; jtau < genTaus->size(); ++jtau) {
      if (deltaR(genTaus->at(jtau), recoTaus->at(itau)) < matchingDeltaR) {
        nGenMatchedToOneReco++;
      }

    // Fill histograms for reco taus matched to at least one gen tau
      if (nGenMatchedToOneReco > 0) {
        h_recoTauMatched_["pt"]->Fill(recoTaus->at(itau).pt());
        h_recoTauMatched_["eta"]->Fill(recoTaus->at(itau).eta());
        h_recoTauMatched_["phi"]->Fill(recoTaus->at(itau).phi());
        h_recoTauMatched_["mass"]->Fill(recoTaus->at(itau).mass());
        if (nGenMatchedToOneReco > 1) {
          h_recoTauMultiMatched_["pt"]->Fill(recoTaus->at(itau).pt());
          h_recoTauMultiMatched_["eta"]->Fill(recoTaus->at(itau).eta());
          h_recoTauMultiMatched_["phi"]->Fill(recoTaus->at(itau).phi());
          h_recoTauMultiMatched_["mass"]->Fill(recoTaus->at(itau).mass());
        }
      }
    }
  }

}

//------------------------------------------------------------------------------
// fill description
//------------------------------------------------------------------------------
void TauValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // Default tau validation HLT
  desc.add<edm::InputTag>("recoTauCollection", edm::InputTag("hltHpsPFTauProducer"));
  desc.add<edm::InputTag>("genTauCollection", edm::InputTag("tauGenJets"));
  desc.add<double>("minDeltaR", 0.3);
  desc.add<std::string>("outFolder", "HLT/Tau/TauValidator");
  desc.addUntracked<bool>("isHLT", true);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TauValidator);
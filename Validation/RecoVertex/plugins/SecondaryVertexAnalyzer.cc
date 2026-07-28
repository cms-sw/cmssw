// Package:    Validation/RecoVertex
// Class:      SecondaryVertexAnalyzer
//
/**\class SecondaryVertexAnalyzer SecondaryVertexAnalyzer.cc
   Validation/RecoVertex/plugins/SecondaryVertexAnalyzer.cc

 Description: DQMEDAnalyzer plugin for secondary vertex validation.
              Responsible only for EDM framework interaction:
                - token declaration and handle fetching
                - bookHistograms callback
                - delegating analyze() to SecondaryVertexAnalyzerAlgo

              All algorithmic logic (getSimSVs, getRecoSVs, matching,
              histogram filling) lives in SecondaryVertexAnalyzerAlgo.

 Supported vertex collection types (registered as separate plugins):
   SecondaryVertexAnalyzer
       for std::vector<reco::Vertex>  (track-based SVs, e.g. IVF)
   VertexCompositePtrCandidateSVAnalyzer
       for std::vector<reco::VertexCompositePtrCandidate>
       (PF-candidate-based SVs, V0s)

 Original Author: Jan Schulz
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "HepMC/GenEvent.h"

#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "Validation/RecoVertex/interface/SecondaryVertexAnalyzerAlgo.h"

template <typename VertexCollection>
class SecondaryVertexAnalyzerBase : public DQMEDAnalyzer {
public:
  using VertexType = typename VertexCollection::value_type;
  using AssociatorWrapper = reco::VertexToTrackingVertexAssociator<VertexCollection>;

  explicit SecondaryVertexAnalyzerBase(const edm::ParameterSet &pset);
  ~SecondaryVertexAnalyzerBase() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void bookHistograms(DQMStore::IBooker &ibook, edm::Run const &, edm::EventSetup const &) override;

  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  // -----------------------------------------------------------------------
  // Tokens — one per input collection
  // -----------------------------------------------------------------------

  // Reco vertex collections (multiple supported for comparison)
  std::vector<edm::EDGetTokenT<edm::View<VertexType>>> recoVertexTokens_;
  std::vector<edm::InputTag> recoVertexTags_;

  // One associator wrapper per reco vertex collection —
  // produced upstream by VertexAssociatorByPositionAndTracksProducer
  std::vector<edm::EDGetTokenT<AssociatorWrapper>> associatorTokens_;

  // Shared across all collections
  const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
  const edm::EDGetTokenT<TrackingVertexCollection> simVertexToken_;
  // const edm::EDGetTokenT<reco::RecoToSimCollection> trackRecoToSimToken_;
  const edm::EDGetTokenT<reco::SimToRecoCollection> trackSimToRecoToken_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> primaryVertexToken_;

  // -----------------------------------------------------------------------
  // Algorithm
  // -----------------------------------------------------------------------
  SecondaryVertexAnalyzerAlgo algo_;
};

// =============================================================================
// Constructor
// =============================================================================

template <typename VertexCollection>
SecondaryVertexAnalyzerBase<VertexCollection>::SecondaryVertexAnalyzerBase(const edm::ParameterSet &pset)
    : hepMCToken_(consumes<edm::HepMCProduct>(pset.getParameter<edm::InputTag>("hepMCProduct"))),
      simVertexToken_(consumes<TrackingVertexCollection>(pset.getParameter<edm::InputTag>("simVertices"))),
      // trackRecoToSimToken_(consumes<reco::RecoToSimCollection>(pset.getParameter<edm::InputTag>("trackAssociation"))),
      trackSimToRecoToken_(consumes<reco::SimToRecoCollection>(pset.getParameter<edm::InputTag>("trackAssociation"))),
      primaryVertexToken_(consumes<std::vector<reco::Vertex>>(pset.getParameter<edm::InputTag>("primaryVertices"))),
      algo_(SecondaryVertexAnalyzerAlgo::Config{
          pset.getUntrackedParameter<std::string>("rootFolder", "Validation/Vertices/Secondary"),
          pset.getUntrackedParameter<bool>("verbose", false),
          pset.getUntrackedParameter<bool>("doGenericSimPlots", true),
          pset.getUntrackedParameter<bool>("doPerPdgPlots", true),
          pset.getParameter<double>("minDecayLength"),
          pset.getParameter<double>("maxDecayLength"),
          pset.getParameter<double>("minPt"),
          pset.getParameter<double>("minPtReconstructableDaughters"),
          pset.getParameter<int>("minReconstructableDaughters"),
          pset.getParameter<bool>("bHadrons"),
          pset.getParameter<bool>("cHadrons"),
          pset.getParameter<bool>("sHadrons"),
          pset.getParameter<bool>("taus"),
          pset.getParameter<bool>("otherParticles"),
          pset.getParameter<std::vector<int>>("signalPdgIds"),
      }) {
  // Reco vertex collections and their paired associators
  const auto recoTags = pset.getParameter<std::vector<edm::InputTag>>("recoVertexCollections");
  const auto assocTags = pset.getParameter<std::vector<edm::InputTag>>("vertexAssociators");

  if (recoTags.size() != assocTags.size())
    throw cms::Exception("Configuration") << "SecondaryVertexAnalyzer: 'recoVertexCollections' and "
                                             "'vertexAssociators' must have the same number of entries.";

  recoVertexTags_ = recoTags;
  recoVertexTokens_.reserve(recoTags.size());
  associatorTokens_.reserve(assocTags.size());
  for (size_t i = 0; i < recoTags.size(); ++i) {
    recoVertexTokens_.push_back(consumes<edm::View<VertexType>>(recoTags[i]));
    associatorTokens_.push_back(consumes<AssociatorWrapper>(assocTags[i]));
  }
}

// =============================================================================
// bookHistograms
// =============================================================================

template <typename VertexCollection>
void SecondaryVertexAnalyzerBase<VertexCollection>::bookHistograms(DQMStore::IBooker &ibook,
                                                                   edm::Run const &,
                                                                   edm::EventSetup const &) {
  std::vector<std::string> labels;
  labels.reserve(recoVertexTags_.size());
  for (const auto &tag : recoVertexTags_)
    labels.push_back(tag.label());

  algo_.bookHistograms(ibook, labels);
}

// =============================================================================
// analyze
// =============================================================================

template <typename VertexCollection>
void SecondaryVertexAnalyzerBase<VertexCollection>::analyze(const edm::Event &iEvent, const edm::EventSetup &) {
  // Fetch shared inputs
  edm::Handle<edm::HepMCProduct> hepMCProdHandle;
  iEvent.getByToken(hepMCToken_, hepMCProdHandle);
  const HepMC::GenEvent *genEvent = hepMCProdHandle.isValid() ? hepMCProdHandle->GetEvent() : nullptr;

  edm::Handle<TrackingVertexCollection> simVertices;
  iEvent.getByToken(simVertexToken_, simVertices);
  if (!simVertices.isValid()) {
    edm::LogWarning("SecondaryVertexAnalyzer") << "TrackingVertexCollection not available — skipping event.";
    return;
  }

  edm::Handle<reco::SimToRecoCollection> trackSimToRecoHandle;
  iEvent.getByToken(trackSimToRecoToken_, trackSimToRecoHandle);
  reco::SimToRecoCollection trackSimToReco;
  if (!trackSimToRecoHandle.isValid()) {
    edm::LogWarning("SecondaryVertexAnalyzer") << "Track SimToRecoCollection not available — cannot evaluate actual "
                                                  "reconstructability of SVs from given track collection.";
  } else {
    trackSimToReco = *trackSimToRecoHandle;
  }

  edm::Handle<std::vector<reco::Vertex>> primaryVertices;
  iEvent.getByToken(primaryVertexToken_, primaryVertices);

  algo_.setPrimaryVertex(primaryVertices);

  algo_.prepareEventTruth(simVertices, genEvent);
  algo_.fillEventTruthHistograms();

  // Loop over configured reco vertex collections
  for (size_t i = 0; i < recoVertexTokens_.size(); ++i) {
    edm::Handle<edm::View<VertexType>> recoVertices;
    iEvent.getByToken(recoVertexTokens_[i], recoVertices);

    edm::Handle<AssociatorWrapper> associator;
    iEvent.getByToken(associatorTokens_[i], associator);

    if (!recoVertices.isValid()) {
      edm::LogWarning("SecondaryVertexAnalyzer")
          << "Reco vertex collection '" << recoVertexTags_[i].label() << "' not available — skipping.";
      continue;
    }
    if (!associator.isValid()) {
      edm::LogWarning("SecondaryVertexAnalyzer")
          << "VertexToTrackingVertexAssociator for '" << recoVertexTags_[i].label() << "' not available — skipping.";
      continue;
    }

    // Associate SimVertices <-> RecoVertices
    const auto &vertexAssociator = *(associator.product());
    auto recoToSim = vertexAssociator.associateRecoToSim(recoVertices, simVertices);
    auto simToReco = vertexAssociator.associateSimToReco(recoVertices, simVertices);

    // Analyze the given collection of RecoVertices
    algo_.analyze(*recoVertices, recoToSim, simToReco, trackSimToReco, recoVertexTags_[i].label());
  }

  algo_.clearEventTruth();
}

// =============================================================================
// fillDescriptions
// =============================================================================

template <typename VertexCollection>
void SecondaryVertexAnalyzerBase<VertexCollection>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<std::string>("rootFolder", "Validation/Vertices/Secondary")
      ->setComment("DQM root folder for all histograms.");
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<bool>("doGenericSimPlots", true)
      ->setComment(
          "Book and fill collection-independent sim vertex plots (decay length spectrum, mother PDG distribution, "
          "etc.).");
  desc.addUntracked<bool>("doPerPdgPlots", true)->setComment("Book per-b/c/other efficiency breakdown histograms.");

  desc.add<std::vector<edm::InputTag>>("recoVertexCollections")
      ->setComment("Reco vertex collections to validate. One entry per collection.");
  desc.add<std::vector<edm::InputTag>>("vertexAssociators")
      ->setComment(
          "VertexToTrackingVertexAssociator wrappers, one per entry in recoVertexCollections. Produced by "
          "e.g. VertexAssociatorByPositionAndTracksProducer.");
  desc.add<edm::InputTag>("primaryVertices")->setComment("Reco primary vertices used for decay length calculation.");
  desc.add<edm::InputTag>("hepMCProduct", edm::InputTag("generatorSmeared"))
      ->setComment("Input generated HepMC event after vtx smearing");
  desc.add<edm::InputTag>("simVertices", edm::InputTag("mix", "MergedTrackTruth"))
      ->setComment("TrackingVertex collection (sim truth).");
  desc.add<edm::InputTag>("trackAssociation", edm::InputTag("trackingParticleRecoTrackAsssociation"))
      ->setComment(
          "Track-TrackingParticle association map (both directions expected under the same InputTag, as produced by "
          "TrackAssociatorEDProducer).");

  desc.add<double>("minDecayLength", 0.01)
      ->setComment(
          "Minimum 3D decay length [cm] for a TrackingVertex to be considered as an SV truth candidate for "
          "efficiency.");
  desc.add<double>("maxDecayLength", 20)
      ->setComment(
          "Maximum 3D decay length [cm] for a TrackingVertex to be considered as an SV truth candidate for "
          "efficiency.");
  desc.add<double>("minPt", 10.)
      ->setComment("Minimum pT for the vector sum of all charged daughters to be included for the efficiency.");
  desc.add<double>("minPtReconstructableDaughters", 0.5)
      ->setComment("Minimum pT for charged daughters to be considered in the count for minReconstructableDaughters.");
  desc.add<int>("minReconstructableDaughters", 2)
      ->setComment("Minimum number of charged daughters for a sim SV to be classified as reconstructable.");
  desc.add<std::vector<int>>("signalPdgIds", {})
      ->setComment("List of PdgIds for the mother daughters of SVs to be included for the efficiency.");
  desc.add<bool>("bHadrons", true)->setComment("Include b-hadrons for the efficiency.");
  desc.add<bool>("cHadrons", true)->setComment("Include c-hadrons for the efficiency.");
  desc.add<bool>("sHadrons", true)->setComment("Include s-hadrons for the efficiency.");
  desc.add<bool>("taus", true)->setComment("Include taus for the efficiency.");
  desc.add<bool>("otherParticles", false)->setComment("Include other particels (not b/c/s/tau) for the efficiency.");

  descriptions.addWithDefaultLabel(desc);
}

// =============================================================================
// Plugin registration
// =============================================================================

using SecondaryVertexAnalyzer = SecondaryVertexAnalyzerBase<std::vector<reco::Vertex>>;
DEFINE_FWK_MODULE(SecondaryVertexAnalyzer);

using SecondaryVertexAnalyzerCPC = SecondaryVertexAnalyzerBase<std::vector<reco::VertexCompositePtrCandidate>>;
DEFINE_FWK_MODULE(SecondaryVertexAnalyzerCPC);

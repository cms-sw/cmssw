#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/MCTruth/plugins/MuonAssociatorEDProducer.h"
#include <memory>

MuonAssociatorEDProducer::MuonAssociatorEDProducer(const edm::ParameterSet &parset)
    : tracksTag(parset.getParameter<edm::InputTag>("tracksTag")),
      tpTag(parset.getParameter<edm::InputTag>("tpTag")),
      tpRefVector(parset.getParameter<bool>("tpRefVector")),
      ignoreMissingTrackCollection(parset.getUntrackedParameter<bool>("ignoreMissingTrackCollection", false)),
      parset_(parset) {
  edm::LogVerbatim("MuonAssociatorEDProducer") << "constructing  MuonAssociatorEDProducer";
  produces<reco::RecoToSimCollection>();
  produces<reco::SimToRecoCollection>();
  if (tpRefVector)
    tpRefVectorToken_ = consumes<TrackingParticleRefVector>(tpTag);
  else
    tpToken_ = consumes<TrackingParticleCollection>(tpTag);
  tracksToken_ = consumes<edm::View<reco::Track>>(tracksTag);

  /// Perform some sanity checks of the configuration
  LogTrace("MuonAssociatorEDProducer") << "constructing  MuonAssociatorByHits" << parset_.dump();
  edm::LogVerbatim("MuonAssociatorEDProducer") << "\n MuonAssociatorByHits will associate reco::Tracks with "
                                               << tracksTag << "\n\t\t and TrackingParticles with " << tpTag;
  const std::string recoTracksLabel = tracksTag.label();

  // check and fix inconsistent input settings
  // tracks with hits only on muon detectors
  if (recoTracksLabel == "seedsOfSTAmuons" || recoTracksLabel == "standAloneMuons" ||
      recoTracksLabel == "refittedStandAloneMuons" || recoTracksLabel == "seedsOfDisplacedSTAmuons" ||
      recoTracksLabel == "displacedStandAloneMuons" || recoTracksLabel == "cosmicMuons" ||
      recoTracksLabel == "cosmicMuons1Leg" || recoTracksLabel == "hltL2Muons") {
    if (parset_.getParameter<bool>("UseTracker")) {
      edm::LogWarning("MuonAssociatorEDProducer")
          << "\n*** WARNING : inconsistent input tracksTag = " << tracksTag << "\n with UseTracker = true"
          << "\n ---> setting UseTracker = false ";
      parset_.addParameter<bool>("UseTracker", false);
    }
    if (!parset_.getParameter<bool>("UseMuon")) {
      edm::LogWarning("MuonAssociatorEDProducer")
          << "\n*** WARNING : inconsistent input tracksTag = " << tracksTag << "\n with UseMuon = false"
          << "\n ---> setting UseMuon = true ";
      parset_.addParameter<bool>("UseMuon", true);
    }
  }
  // tracks with hits only on tracker
  if (recoTracksLabel == "generalTracks" || recoTracksLabel == "probeTracks" || recoTracksLabel == "displacedTracks" ||
      recoTracksLabel == "extractGemMuons" || recoTracksLabel == "extractMe0Muons" ||
      recoTracksLabel == "ctfWithMaterialTracksP5LHCNavigation" || recoTracksLabel == "ctfWithMaterialTracksP5" ||
      recoTracksLabel == "hltIterL3OIMuonTrackSelectionHighPurity" || recoTracksLabel == "hltIterL3MuonMerged" ||
      recoTracksLabel == "hltIterL3MuonAndMuonFromL1Merged") {
    if (parset_.getParameter<bool>("UseMuon")) {
      edm::LogWarning("MuonAssociatorEDProducer")
          << "\n*** WARNING : inconsistent input tracksTag = " << tracksTag << "\n with UseMuon = true"
          << "\n ---> setting UseMuon = false ";
      parset_.addParameter<bool>("UseMuon", false);
    }
    if (!parset_.getParameter<bool>("UseTracker")) {
      edm::LogWarning("MuonAssociatorEDProducer")
          << "\n*** WARNING : inconsistent input tracksTag = " << tracksTag << "\n with UseTracker = false"
          << "\n ---> setting UseTracker = true ";
      parset_.addParameter<bool>("UseTracker", true);
    }
  }

  LogTrace("MuonAssociatorEDProducer") << "MuonAssociatorEDProducer::beginJob "
                                          ": constructing MuonAssociatorByHits";
  associatorByHits = new MuonAssociatorByHits(parset_, consumesCollector());
}

MuonAssociatorEDProducer::~MuonAssociatorEDProducer() {}

void MuonAssociatorEDProducer::beginJob() {}

void MuonAssociatorEDProducer::endJob() {}

void MuonAssociatorEDProducer::produce(edm::Event &event, const edm::EventSetup &setup) {
  using namespace edm;

  TrackingParticleRefVector tmpTP;
  const TrackingParticleRefVector *tmpTPptr = nullptr;
  Handle<TrackingParticleCollection> TPCollection;
  Handle<TrackingParticleRefVector> TPCollectionRefVector;

  if (tpRefVector) {
    event.getByToken(tpRefVectorToken_, TPCollectionRefVector);
    tmpTPptr = TPCollectionRefVector.product();
    //
    tmpTP = *tmpTPptr;
  } else {
    event.getByToken(tpToken_, TPCollection);
    size_t nTP = TPCollection->size();
    for (size_t i = 0; i < nTP; ++i) {
      tmpTP.push_back(TrackingParticleRef(TPCollection, i));
    }
    tmpTPptr = &tmpTP;
  }

  LogTrace("MuonAssociatorEDProducer") << "getting TrackingParticle collection - " << tpTag;
  LogTrace("MuonAssociatorEDProducer") << "\t... size = " << tmpTPptr->size();

  Handle<edm::View<reco::Track>> trackCollection;
  LogTrace("MuonAssociatorEDProducer") << "getting reco::Track collection - " << tracksTag;
  bool trackAvailable = event.getByToken(tracksToken_, trackCollection);
  if (trackAvailable)
    LogTrace("MuonAssociatorEDProducer") << "\t... size = " << trackCollection->size();
  else
    LogTrace("MuonAssociatorEDProducer") << "\t... NOT FOUND.";

  std::unique_ptr<reco::RecoToSimCollection> rts;
  std::unique_ptr<reco::SimToRecoCollection> str;

  if (ignoreMissingTrackCollection && !trackAvailable) {
    // the track collection is not in the event and we're being told to ignore
    // this. do not output anything to the event, other wise this would be
    // considered as inefficiency.
    LogTrace("MuonAssociatorEDProducer") << "\n ignoring missing track collection."
                                         << "\n";
  } else {
    edm::RefToBaseVector<reco::Track> tmpT;
    for (size_t i = 0; i < trackCollection->size(); ++i)
      tmpT.push_back(trackCollection->refAt(i));

    edm::LogVerbatim("MuonAssociatorEDProducer")
        << "\n >>> RecoToSim association <<< \n"
        << "     Track collection : " << tracksTag.label() << ":" << tracksTag.instance()
        << " (size = " << trackCollection->size() << ") \n"
        << "     TrackingParticle collection : " << tpTag.label() << ":" << tpTag.instance()
        << " (size = " << tmpTPptr->size() << ")";

    reco::RecoToSimCollection recSimColl = associatorByHits->associateRecoToSim(tmpT, tmpTP, &event, &setup);

    edm::LogVerbatim("MuonAssociatorEDProducer")
        << "\n >>> SimToReco association <<< \n"
        << "     TrackingParticle collection : " << tpTag.label() << ":" << tpTag.instance()
        << " (size = " << tmpTPptr->size() << ") \n"
        << "     Track collection : " << tracksTag.label() << ":" << tracksTag.instance()
        << " (size = " << trackCollection->size() << ")";

    reco::SimToRecoCollection simRecColl = associatorByHits->associateSimToReco(tmpT, tmpTP, &event, &setup);

    rts = std::make_unique<reco::RecoToSimCollection>(recSimColl);
    str = std::make_unique<reco::SimToRecoCollection>(simRecColl);

    event.put(std::move(rts));
    event.put(std::move(str));
  }
}

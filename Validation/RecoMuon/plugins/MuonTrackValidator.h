#ifndef MuonTrackValidator_h
#define MuonTrackValidator_h

/** \class MuonTrackValidator
* Class that produces histograms to validate Muon Track Reconstruction performances
*
*/
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Validation/RecoMuon/plugins/MuonTrackValidatorBase.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class MuonTrackValidator : public DQMEDAnalyzer, protected MuonTrackValidatorBase {
public:
  /// Constructor
  MuonTrackValidator(const edm::ParameterSet& pset) : MuonTrackValidatorBase(pset) {
    dirName_ = pset.getParameter<std::string>("dirName");
    associatormap = pset.getParameter<edm::InputTag>("associatormap");
    UseAssociators = pset.getParameter<bool>("UseAssociators");
    useGEMs_ = pset.getParameter<bool>("useGEMs");
    useME0_ = pset.getParameter<bool>("useME0");
    edm::ParameterSet tpset = pset.getParameter<edm::ParameterSet>("muonTPSelector");
    tpSelector = TrackingParticleSelector(tpset.getParameter<double>("ptMin"),
                                          tpset.getParameter<double>("ptMax"),
                                          tpset.getParameter<double>("minRapidity"),
                                          tpset.getParameter<double>("maxRapidity"),
                                          tpset.getParameter<double>("tip"),
                                          tpset.getParameter<double>("lip"),
                                          tpset.getParameter<int>("minHit"),
                                          tpset.getParameter<bool>("signalOnly"),
                                          tpset.getParameter<bool>("intimeOnly"),
                                          tpset.getParameter<bool>("chargedOnly"),
                                          tpset.getParameter<bool>("stableOnly"),
                                          tpset.getParameter<std::vector<int> >("pdgId"));

    cosmictpSelector = CosmicTrackingParticleSelector(tpset.getParameter<double>("ptMin"),
                                                      tpset.getParameter<double>("minRapidity"),
                                                      tpset.getParameter<double>("maxRapidity"),
                                                      tpset.getParameter<double>("tip"),
                                                      tpset.getParameter<double>("lip"),
                                                      tpset.getParameter<int>("minHit"),
                                                      tpset.getParameter<bool>("chargedOnly"),
                                                      tpset.getParameter<std::vector<int> >("pdgId"));

    BiDirectional_RecoToSim_association = pset.getParameter<bool>("BiDirectional_RecoToSim_association");

    // dump cfg parameters
    edm::LogVerbatim("MuonTrackValidator") << "constructing MuonTrackValidator: " << pset.dump();

    // Declare consumes (also for the base class)
    bsSrc_Token = consumes<reco::BeamSpot>(bsSrc);
    tp_effic_Token = consumes<TrackingParticleCollection>(label_tp_effic);
    tp_fake_Token = consumes<TrackingParticleCollection>(label_tp_fake);
    pileupinfo_Token = consumes<std::vector<PileupSummaryInfo> >(label_pileupinfo);
    for (unsigned int www = 0; www < label.size(); www++) {
      track_Collection_Token.push_back(consumes<edm::View<reco::Track> >(label[www]));
    }
    simToRecoCollection_Token = consumes<reco::SimToRecoCollection>(associatormap);
    recoToSimCollection_Token = consumes<reco::RecoToSimCollection>(associatormap);

    _simHitTpMapTag = mayConsume<SimHitTPAssociationProducer::SimHitTPAssociationList>(
        pset.getParameter<edm::InputTag>("simHitTpMapTag"));

    MABH = false;
    if (!UseAssociators) {
      // flag MuonAssociatorByHits
      if (associators[0] == "MuonAssociationByHits")
        MABH = true;
      // reset string associators to the map label
      associators.clear();
      associators.push_back(associatormap.label());
      edm::LogVerbatim("MuonTrackValidator") << "--> associators reset to: " << associators[0];
    } else {
      for (auto const& associator : associators) {
        consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag(associator));
      }
    }

    // inform on which SimHits will be counted
    if (usetracker)
      edm::LogVerbatim("MuonTrackValidator") << "\n usetracker = TRUE : Tracker SimHits WILL be counted";
    else
      edm::LogVerbatim("MuonTrackValidator") << "\n usetracker = FALSE : Tracker SimHits WILL NOT be counted";
    if (usemuon)
      edm::LogVerbatim("MuonTrackValidator") << " usemuon = TRUE : Muon SimHits WILL be counted";
    else
      edm::LogVerbatim("MuonTrackValidator") << " usemuon = FALSE : Muon SimHits WILL NOT be counted" << std::endl;

    // loop over the reco::Track collections to validate: check for inconsistent input settings
    for (unsigned int www = 0; www < label.size(); www++) {
      std::string recoTracksLabel = label[www].label();
      std::string recoTracksInstance = label[www].instance();

      // tracks with hits only on tracker
      if (recoTracksLabel == "generalTracks" || (recoTracksLabel.find("cutsRecoTracks") != std::string::npos) ||
          recoTracksLabel == "ctfWithMaterialTracksP5LHCNavigation" || recoTracksLabel == "hltL3TkTracksFromL2" ||
          (recoTracksLabel == "hltL3Muons" && recoTracksInstance == "L2Seeded")) {
        if (usemuon) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usemuon == true"
              << "\n ---> please change to usemuon == false ";
        }
        if (!usetracker) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usetracker == false"
              << "\n ---> please change to usetracker == true ";
        }
      }

      // tracks with hits only on muon detectors
      else if (recoTracksLabel == "standAloneMuons" || recoTracksLabel == "standAloneSETMuons" ||
               recoTracksLabel == "cosmicMuons" || recoTracksLabel == "hltL2Muons") {
        if (usetracker) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usetracker == true"
              << "\n ---> please change to usetracker == false ";
        }
        if (!usemuon) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usemuon == false"
              << "\n ---> please change to usemuon == true ";
        }
      }

    }  // for (unsigned int www=0;www<label.size();www++)
  }

  /// Destructor
  ~MuonTrackValidator() override {}

  /// Method called before the event loop
  //  void beginRun(edm::Run const&, edm::EventSetup const&);
  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  /// Method called at the end of the event loop
  //   void dqmEndRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMEDAnalyzer::DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  std::string dirName_;
  edm::InputTag associatormap;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoCollection_Token;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimCollection_Token;
  edm::EDGetTokenT<SimHitTPAssociationProducer::SimHitTPAssociationList> _simHitTpMapTag;

  bool UseAssociators;
  bool useGEMs_;
  bool useME0_;

  // select tracking particles
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;
  CosmicTrackingParticleSelector cosmictpSelector;

  // flag new validation logic (bidirectional RecoToSim association)
  bool BiDirectional_RecoToSim_association;
  // flag MuonAssociatorByHits
  bool MABH;
};

#endif

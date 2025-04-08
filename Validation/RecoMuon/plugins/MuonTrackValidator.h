#ifndef MuonTrackValidator_h
#define MuonTrackValidator_h

/** \class MuonTrackValidator
* Class that produces histograms to validate Muon Track Reconstruction performances
*
*/
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Validation/RecoMuon/plugins/MuonTrackValidatorBase.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"
#include "SimTracker/TrackAssociation/interface/CosmicParametersDefinerForTP.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class MuonTrackValidator : public DQMEDAnalyzer, protected MuonTrackValidatorBase {
public:
  /// Constructor
  MuonTrackValidator(const edm::ParameterSet& pset) : MuonTrackValidatorBase(pset) {
    dirName_ = pset.getParameter<std::string>("dirName");
    associatormap = pset.getParameter<std::vector<edm::InputTag>>("associatormap");
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
                                          tpset.getParameter<std::vector<int>>("pdgId"));

    cosmictpSelector = CosmicTrackingParticleSelector(tpset.getParameter<double>("ptMin"),
                                                      tpset.getParameter<double>("minRapidity"),
                                                      tpset.getParameter<double>("maxRapidity"),
                                                      tpset.getParameter<double>("tip"),
                                                      tpset.getParameter<double>("lip"),
                                                      tpset.getParameter<int>("minHit"),
                                                      tpset.getParameter<bool>("chargedOnly"),
                                                      tpset.getParameter<std::vector<int>>("pdgId"));

    BiDirectional_RecoToSim_association = pset.getParameter<bool>("BiDirectional_RecoToSim_association");
    doSummaryPlots_ = pset.getParameter<bool>("doSummaryPlots");

    // dump cfg parameters
    edm::LogVerbatim("MuonTrackValidator") << "constructing MuonTrackValidator: " << pset.dump();

    // Declare consumes (also for the base class)
    bsSrc_Token = consumes<reco::BeamSpot>(bsSrc);
    if (label_tp_refvector)
      tp_refvector_Token = consumes<TrackingParticleRefVector>(label_tp);
    else
      tp_Token = consumes<TrackingParticleCollection>(label_tp);
    pileupinfo_Token = consumes<std::vector<PileupSummaryInfo>>(label_pileupinfo);

    if (!UseAssociators && label.size() != associatormap.size()) {
      throw cms::Exception("Configuration")
          << "Different number of labels and associators provided!" << '\n'
          << "Please, make sure to configure the muonValidator with the same number and ordering "
             "of labels and associators!";
    }

    if (label.size() == 1 && doSummaryPlots_) {
      edm::LogWarning("MuonTrackValidator")
          << "Cannot produce summary plots for a single collection label, disabling summary plots creation!" << '\n';
      doSummaryPlots_ = false;
    }

    if (label.size() != histoParameters.size()) {
      throw cms::Exception("Configuration")
          << "Different number of labels and histoParameters provided!" << '\n'
          << "Please, make sure to configure the muonValidator with the same number and ordering "
             "of labels and histoParameters!";
    }

    track_Collection_Token.reserve(label.size());
    simToRecoCollection_Token.reserve(label.size());
    recoToSimCollection_Token.reserve(label.size());
    for (unsigned int www = 0; www < label.size(); www++) {
      track_Collection_Token.push_back(consumes<edm::View<reco::Track>>(label[www]));
      simToRecoCollection_Token.push_back(consumes<reco::SimToRecoCollection>(associatormap[www]));
      recoToSimCollection_Token.push_back(consumes<reco::RecoToSimCollection>(associatormap[www]));
    }

    if (parametersDefiner == "LhcParametersDefinerForTP") {
      lhcParametersDefinerTP_ = std::make_unique<ParametersDefinerForTP>(bsSrc, consumesCollector());
    } else if (parametersDefiner == "CosmicParametersDefinerForTP") {
      cosmicParametersDefinerTP_ = std::make_unique<CosmicParametersDefinerForTP>(consumesCollector());
    } else {
      throw cms::Exception("Configuration") << "Unexpected label: parametersDefiner = " << parametersDefiner;
    }

    _simHitTpMapTag = mayConsume<SimHitTPAssociationProducer::SimHitTPAssociationList>(
        pset.getParameter<edm::InputTag>("simHitTpMapTag"));

    MABH = false;
    if (!UseAssociators) {
      // flag MuonAssociatorByHits
      if (associators[0] == "MuonAssociationByHits")
        MABH = true;
      // reset string associators to the map label
      associators.clear();
      associators.reserve(associatormap.size());
      std::transform(
          associatormap.begin(), associatormap.end(), std::back_inserter(associators), [](const edm::InputTag& tag) {
            return tag.label();
          });
    } else {
      for (auto const& associator : associators) {
        consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag(associator));
      }
    }

    // loop over the reco::Track collections to validate: check for inconsistent input settings
    for (unsigned int www = 0; www < label.size(); www++) {
      // inform on which SimHits will be counted
      if (histoParameters[www].usetracker)
        edm::LogVerbatim("MuonTrackValidator") << "\n usetracker = TRUE : Tracker SimHits WILL be counted";
      else
        edm::LogVerbatim("MuonTrackValidator") << "\n usetracker = FALSE : Tracker SimHits WILL NOT be counted";
      if (histoParameters[www].usemuon)
        edm::LogVerbatim("MuonTrackValidator") << " usemuon = TRUE : Muon SimHits WILL be counted";
      else
        edm::LogVerbatim("MuonTrackValidator") << " usemuon = FALSE : Muon SimHits WILL NOT be counted" << std::endl;

      std::string recoTracksLabel = label[www].label();
      std::string recoTracksInstance = label[www].instance();

      // tracks with hits only on tracker
      if (recoTracksLabel == "generalTracks" || recoTracksLabel == "probeTracks" ||
          recoTracksLabel == "displacedTracks" || recoTracksLabel == "extractGemMuons" ||
          recoTracksLabel == "extractMe0Muons" || recoTracksLabel == "ctfWithMaterialTracksP5LHCNavigation" ||
          recoTracksLabel == "ctfWithMaterialTracksP5" ||
          recoTracksLabel == "hltIterL3OIMuonTrackSelectionHighPurity" || recoTracksLabel == "hltIterL3MuonMerged" ||
          recoTracksLabel == "hltIterL3MuonAndMuonFromL1Merged") {
        if (histoParameters[www].usemuon) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usemuon == true"
              << "\n ---> resetting to usemuon == false ";
          histoParameters[www].usemuon = false;
        }
        if (!histoParameters[www].usetracker) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usetracker == false"
              << "\n ---> resetting to usetracker == true ";
          histoParameters[www].usetracker = true;
        }
      }

      // tracks with hits only on muon detectors
      else if (recoTracksLabel == "seedsOfSTAmuons" || recoTracksLabel == "standAloneMuons" ||
               recoTracksLabel == "seedsOfDisplacedSTAmuons" || recoTracksLabel == "displacedStandAloneMuons" ||
               recoTracksLabel == "refittedStandAloneMuons" || recoTracksLabel == "cosmicMuons" ||
               recoTracksLabel == "cosmicMuons1Leg" || recoTracksLabel == "hltL2Muons") {
        if (histoParameters[www].usetracker) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usetracker == true"
              << "\n ---> resetting to usetracker == false ";
          histoParameters[www].usetracker = false;
        }
        if (!histoParameters[www].usemuon) {
          edm::LogWarning("MuonTrackValidator")
              << "\n*** WARNING : inconsistent input tracksTag = " << label[www] << "\n with usemuon == false"
              << "\n ---> resetting to usemuon == true ";
          histoParameters[www].usemuon = true;
        }
      }

    }  // for (unsigned int www=0;www<label.size();www++)
  }

  /// Destructor
  ~MuonTrackValidator() override {}

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMEDAnalyzer::DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  std::string dirName_;
  std::vector<edm::InputTag> associatormap;
  std::vector<edm::EDGetTokenT<reco::SimToRecoCollection>> simToRecoCollection_Token;
  std::vector<edm::EDGetTokenT<reco::RecoToSimCollection>> recoToSimCollection_Token;
  edm::EDGetTokenT<SimHitTPAssociationProducer::SimHitTPAssociationList> _simHitTpMapTag;

  std::unique_ptr<ParametersDefinerForTP> lhcParametersDefinerTP_;
  std::unique_ptr<CosmicParametersDefinerForTP> cosmicParametersDefinerTP_;

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
  bool doSummaryPlots_;
};

#endif

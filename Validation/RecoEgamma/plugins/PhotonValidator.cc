#include <iostream>
#include <memory>

//
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "Validation/RecoEgamma/plugins/PhotonValidator.h"

//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
//
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//
#include "DataFormats/Math/interface/deltaR.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
//
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

//
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
//
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//
/** \class PhotonValidator
 **
 **
 **  $Id: PhotonValidator
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

using namespace std;

PhotonValidator::PhotonValidator(const edm::ParameterSet& pset)
    : magneticFieldToken_{esConsumes<edm::Transition::BeginRun>()},
      caloGeometryToken_{esConsumes()},
      transientTrackBuilderToken_{esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))} {
  fName_ = pset.getParameter<std::string>("analyzerName");
  verbosity_ = pset.getUntrackedParameter<int>("Verbosity");
  parameters_ = pset;
  fastSim_ = pset.getParameter<bool>("fastSim");
  isRunCentrally_ = pset.getParameter<bool>("isRunCentrally");

  photonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
  photonCollection_ = pset.getParameter<std::string>("photonCollection");
  photonCollectionToken_ =
      consumes<reco::PhotonCollection>(edm::InputTag(photonCollectionProducer_, photonCollection_));

  token_tp_ = consumes<TrackingParticleCollection>(pset.getParameter<edm::InputTag>("label_tp"));

  barrelEcalHits_ = consumes<EcalRecHitCollection>(pset.getParameter<edm::InputTag>("barrelEcalHits"));
  endcapEcalHits_ = consumes<EcalRecHitCollection>(pset.getParameter<edm::InputTag>("endcapEcalHits"));

  conversionOITrackProducer_ = pset.getParameter<std::string>("conversionOITrackProducer");
  conversionIOTrackProducer_ = pset.getParameter<std::string>("conversionIOTrackProducer");
  conversionOITrackPr_Token_ = consumes<edm::View<reco::Track> >(edm::InputTag(conversionOITrackProducer_));
  conversionIOTrackPr_Token_ = consumes<edm::View<reco::Track> >(edm::InputTag(conversionIOTrackProducer_));

  pfCandidates_ = consumes<reco::PFCandidateCollection>(pset.getParameter<edm::InputTag>("pfCandidates"));
  valueMapPhoPFCandIso_ = pset.getParameter<std::string>("valueMapPhoToParticleBasedIso");
  particleBasedIso_token =
      consumes<edm::ValueMap<std::vector<reco::PFCandidateRef> > >(pset.getUntrackedParameter<edm::InputTag>(
          "particleBasedIso", edm::InputTag("particleBasedIsolation", valueMapPhoPFCandIso_)));

  minPhoEtCut_ = pset.getParameter<double>("minPhoEtCut");
  convTrackMinPtCut_ = pset.getParameter<double>("convTrackMinPtCut");
  likelihoodCut_ = pset.getParameter<double>("likelihoodCut");

  trkIsolExtRadius_ = pset.getParameter<double>("trkIsolExtR");
  trkIsolInnRadius_ = pset.getParameter<double>("trkIsolInnR");
  trkPtLow_ = pset.getParameter<double>("minTrackPtCut");
  lip_ = pset.getParameter<double>("lipCut");
  ecalIsolRadius_ = pset.getParameter<double>("ecalIsolR");
  bcEtLow_ = pset.getParameter<double>("minBcEtCut");
  hcalIsolExtRadius_ = pset.getParameter<double>("hcalIsolExtR");
  hcalIsolInnRadius_ = pset.getParameter<double>("hcalIsolInnR");
  hcalHitEtLow_ = pset.getParameter<double>("minHcalHitEtCut");

  numOfTracksInCone_ = pset.getParameter<int>("maxNumOfTracksInCone");
  trkPtSumCut_ = pset.getParameter<double>("trkPtSumCut");
  ecalEtSumCut_ = pset.getParameter<double>("ecalEtSumCut");
  hcalEtSumCut_ = pset.getParameter<double>("hcalEtSumCut");
  dCotCutOn_ = pset.getParameter<bool>("dCotCutOn");
  dCotCutValue_ = pset.getParameter<double>("dCotCutValue");
  dCotHardCutValue_ = pset.getParameter<double>("dCotHardCutValue");

  offline_pvToken_ = consumes<reco::VertexCollection>(
      pset.getUntrackedParameter<edm::InputTag>("offlinePV", edm::InputTag("offlinePrimaryVertices")));
  g4_simTk_Token_ = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  g4_simVtx_Token_ = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  famos_simTk_Token_ = consumes<edm::SimTrackContainer>(edm::InputTag("fastSimProducer"));
  famos_simVtx_Token_ = consumes<edm::SimVertexContainer>(edm::InputTag("fastSimProducer"));
  hepMC_Token_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));
  genjets_Token_ = consumes<reco::GenJetCollection>(edm::InputTag("ak4GenJets"));

  genpartToken_ = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));

  consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByHitsForPhotonValidation"));

  nEvt_ = 0;
  nEntry_ = 0;
  nRecConv_ = 0;
  nRecConvAss_ = 0;
  nRecConvAssWithEcal_ = 0;
  nInvalidPCA_ = 0;
}

PhotonValidator::~PhotonValidator() {}

void PhotonValidator::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const&) {
  double resMin = parameters_.getParameter<double>("resMin");
  double resMax = parameters_.getParameter<double>("resMax");
  int resBin = parameters_.getParameter<int>("resBin");

  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");

  double etScale = parameters_.getParameter<double>("etScale");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");
  int etaBin2 = parameters_.getParameter<int>("etaBin2");

  double dEtaMin = parameters_.getParameter<double>("dEtaMin");
  double dEtaMax = parameters_.getParameter<double>("dEtaMax");
  int dEtaBin = parameters_.getParameter<int>("dEtaBin");

  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int phiBin = parameters_.getParameter<int>("phiBin");

  double dPhiMin = parameters_.getParameter<double>("dPhiMin");
  double dPhiMax = parameters_.getParameter<double>("dPhiMax");
  int dPhiBin = parameters_.getParameter<int>("dPhiBin");

  double rMin = parameters_.getParameter<double>("rMin");
  double rMax = parameters_.getParameter<double>("rMax");
  int rBin = parameters_.getParameter<int>("rBin");

  double zMin = parameters_.getParameter<double>("zMin");
  double zMax = parameters_.getParameter<double>("zMax");
  int zBin = parameters_.getParameter<int>("zBin");

  double r9Min = parameters_.getParameter<double>("r9Min");
  double r9Max = parameters_.getParameter<double>("r9Max");
  int r9Bin = parameters_.getParameter<int>("r9Bin");

  double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin");
  double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax");
  int dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin");

  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin");
  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax");
  int dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");

  double dCotTracksMin = parameters_.getParameter<double>("dCotTracksMin");
  double dCotTracksMax = parameters_.getParameter<double>("dCotTracksMax");
  int dCotTracksBin = parameters_.getParameter<int>("dCotTracksBin");

  double povereMin = parameters_.getParameter<double>("povereMin");
  double povereMax = parameters_.getParameter<double>("povereMax");
  int povereBin = parameters_.getParameter<int>("povereBin");

  double eoverpMin = parameters_.getParameter<double>("eoverpMin");
  double eoverpMax = parameters_.getParameter<double>("eoverpMax");
  int eoverpBin = parameters_.getParameter<int>("eoverpBin");

  double chi2Min = parameters_.getParameter<double>("chi2Min");
  double chi2Max = parameters_.getParameter<double>("chi2Max");

  int ggMassBin = parameters_.getParameter<int>("ggMassBin");
  double ggMassMin = parameters_.getParameter<double>("ggMassMin");
  double ggMassMax = parameters_.getParameter<double>("ggMassMax");

  double rMinForXray = parameters_.getParameter<double>("rMinForXray");
  double rMaxForXray = parameters_.getParameter<double>("rMaxForXray");
  int rBinForXray = parameters_.getParameter<int>("rBinForXray");
  double zMinForXray = parameters_.getParameter<double>("zMinForXray");
  double zMaxForXray = parameters_.getParameter<double>("zMaxForXray");
  int zBinForXray = parameters_.getParameter<int>("zBinForXray");
  int zBin2ForXray = parameters_.getParameter<int>("zBin2ForXray");

  //// All MC photons
  // SC from reco photons

  iBooker.setCurrentFolder("EgammaV/" + fName_ + "/SimulationInfo");

  // simulation information about all MC photons found
  std::string histname = "nOfSimPhotons";
  if (!isRunCentrally_) {
    h_nSimPho_[0] = iBooker.book1D(histname, "# of Sim photons per event ", 20, -0.5, 19.5);
    histname = "SimPhoMotherEt";
    h_SimPhoMotherEt_[0] = iBooker.book1D(histname, "Sim photon Mother tranverse energy spectrum", etBin, etMin, etMax);
    h_SimPhoMotherEta_[0] = iBooker.book1D("SimPhoMotherEta", " Sim Photon Mother Eta ", etaBin, etaMin, etaMax);
    histname = "SimPhoMotherEtMatched";
    h_SimPhoMotherEt_[1] = iBooker.book1D(
        histname, "Sim photon  matched by a reco Photon: Mother tranverse energy spectrum", etBin, etMin, etMax);
    h_SimPhoMotherEta_[1] = iBooker.book1D(
        "SimPhoMotherEtaMatched", " Sim Photon matched by a reco Photon:  Mother Eta ", etaBin, etaMin, etaMax);
  }

  histname = "h_SimPhoEta";
  h_SimPho_[0] = iBooker.book1D(histname, " All photons simulated #eta", etaBin, etaMin, etaMax);
  histname = "h_SimPhoPhi";
  h_SimPho_[1] = iBooker.book1D(histname, " All photons simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimPhoEt";
  h_SimPho_[2] = iBooker.book1D(histname, " All photons simulated Et", etBin, etMin, etMax);
  // Numerators
  histname = "nOfSimPhotonsMatched";
  h_nSimPho_[1] = iBooker.book1D(histname, "# of Sim photons matched by a reco Photon per event ", 20, -0.5, 19.5);
  histname = "h_MatchedSimPhoEta";
  h_MatchedSimPho_[0] = iBooker.book1D(histname, " Matching photons simulated #eta", etaBin, etaMin, etaMax);
  histname = "h_MatchedSimPhoPhi";
  h_MatchedSimPho_[1] = iBooker.book1D(histname, " Matching photons simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_MatchedSimPhoEt";
  h_MatchedSimPho_[2] = iBooker.book1D(histname, " Matching photons simulated Et", etBin, etMin, etMax);
  //
  histname = "h_MatchedSimPhoBadChEta";
  h_MatchedSimPhoBadCh_[0] = iBooker.book1D(histname, " Matching photons simulated #eta", etaBin, etaMin, etaMax);
  histname = "h_MatchedSimPhoBadChPhi";
  h_MatchedSimPhoBadCh_[1] = iBooker.book1D(histname, " Matching photons simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_MatchedSimPhoBadChEt";
  h_MatchedSimPhoBadCh_[2] = iBooker.book1D(histname, " Matching photons simulated Et", etBin, etMin, etMax);

  /// Histograms for efficiencies
  histname = "nOfSimConversions";
  if (!isRunCentrally_) {
    h_nSimConv_[0] = iBooker.book1D(histname, "# of Sim conversions per event ", 20, -0.5, 19.5);
    histname = "nOfVisSimConversions";
    h_nSimConv_[1] = iBooker.book1D(histname, "# of Sim conversions per event ", 20, -0.5, 19.5);
  }
  /// Denominators
  histname = "h_AllSimConvEta";
  h_AllSimConv_[0] = iBooker.book1D(histname, " All conversions: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_AllSimConvPhi";
  h_AllSimConv_[1] = iBooker.book1D(histname, " All conversions: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_AllSimConvR";
  h_AllSimConv_[2] = iBooker.book1D(histname, " All conversions: simulated R", rBin, rMin, rMax);
  histname = "h_AllSimConvZ";
  h_AllSimConv_[3] = iBooker.book1D(histname, " All conversions: simulated Z", zBin, zMin, zMax);
  histname = "h_AllSimConvEt";
  h_AllSimConv_[4] = iBooker.book1D(histname, " All conversions: simulated Et", etBin, etMin, etMax);
  //
  histname = "h_VisSimConvEta";
  h_VisSimConv_[0] = iBooker.book1D(histname, " All vis conversions: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_VisSimConvPhi";
  h_VisSimConv_[1] = iBooker.book1D(histname, " All vis conversions: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_VisSimConvR";
  h_VisSimConv_[2] = iBooker.book1D(histname, " All vis conversions: simulated R", rBin, rMin, rMax);
  histname = "h_VisSimConvZ";
  h_VisSimConv_[3] = iBooker.book1D(histname, " All vis conversions: simulated Z", zBin, zMin, zMax);
  histname = "h_VisSimConvEt";
  h_VisSimConv_[4] = iBooker.book1D(histname, " All vis conversions: simulated Et", etBin, etMin, etMax);
  /// Numerators
  histname = "h_SimConvOneTracksEta";
  h_SimConvOneTracks_[0] =
      iBooker.book1D(histname, " All vis conversions with 1 reco  tracks: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_SimConvOneTracksPhi";
  h_SimConvOneTracks_[1] =
      iBooker.book1D(histname, " All vis conversions with 1 reco  tracks: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimConvOneTracksR";
  h_SimConvOneTracks_[2] =
      iBooker.book1D(histname, " All vis conversions with 1 reco  tracks: simulated R", rBin, rMin, rMax);
  histname = "h_SimConvOneTracksZ";
  h_SimConvOneTracks_[3] =
      iBooker.book1D(histname, " All vis conversions with 1 reco  tracks: simulated Z", zBin, zMin, zMax);
  histname = "h_SimConvOneTracksEt";
  h_SimConvOneTracks_[4] =
      iBooker.book1D(histname, " All vis conversions with 1 reco  tracks: simulated Et", etBin, etMin, etMax);
  //
  histname = "h_SimConvTwoMTracksEta";
  h_SimConvTwoMTracks_[0] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_SimConvTwoMTracksPhi";
  h_SimConvTwoMTracks_[1] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimConvTwoMTracksR";
  h_SimConvTwoMTracks_[2] =
      iBooker.book1D(histname, " All vis conversions with 2 reco-matching tracks: simulated R", rBin, rMin, rMax);
  histname = "h_SimConvTwoMTracksZ";
  h_SimConvTwoMTracks_[3] =
      iBooker.book1D(histname, " All vis conversions with 2 reco-matching tracks: simulated Z", zBin, zMin, zMax);
  histname = "h_SimConvTwoMTracksEt";
  h_SimConvTwoMTracks_[4] =
      iBooker.book1D(histname, " All vis conversions with 2 reco-matching tracks: simulated Et", etBin, etMin, etMax);
  //
  histname = "h_SimConvTwoTracksEta";
  h_SimConvTwoTracks_[0] =
      iBooker.book1D(histname, " All vis conversions with 2 reco  tracks: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_SimConvTwoTracksPhi";
  h_SimConvTwoTracks_[1] =
      iBooker.book1D(histname, " All vis conversions with 2 reco tracks: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimConvTwoTracksR";
  h_SimConvTwoTracks_[2] =
      iBooker.book1D(histname, " All vis conversions with 2 reco tracks: simulated R", rBin, rMin, rMax);
  histname = "h_SimConvTwoTracksZ";
  h_SimConvTwoTracks_[3] =
      iBooker.book1D(histname, " All vis conversions with 2 reco tracks: simulated Z", zBin, zMin, zMax);
  histname = "h_SimConvTwoTracksEt";
  h_SimConvTwoTracks_[4] =
      iBooker.book1D(histname, " All vis conversions with 2 reco tracks: simulated Et", etBin, etMin, etMax);
  //
  histname = "h_SimConvOneMTracksEta";
  h_SimConvOneMTracks_[0] = iBooker.book1D(
      histname, " All vis conversions with 1 reco-matching tracks: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_SimConvOneMTracksPhi";
  h_SimConvOneMTracks_[1] = iBooker.book1D(
      histname, " All vis conversions with 1 reco-matching tracks: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimConvOneMTracksR";
  h_SimConvOneMTracks_[2] =
      iBooker.book1D(histname, " All vis conversions with 1 reco-matching tracks: simulated R", rBin, rMin, rMax);
  histname = "h_SimConvOneMTracksZ";
  h_SimConvOneMTracks_[3] =
      iBooker.book1D(histname, " All vis conversions with 1 reco-matching tracks: simulated Z", zBin, zMin, zMax);
  histname = "h_SimConvOneMTracksEt";
  h_SimConvOneMTracks_[4] =
      iBooker.book1D(histname, " All vis conversions with 1 reco-matching tracks: simulated Et", etBin, etMin, etMax);
  //
  histname = "h_SimConvTwoMTracksEtaAndVtxPGT0";
  h_SimConvTwoMTracksAndVtxPGT0_[0] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_SimConvTwoMTracksPhiAndVtxPGT0";
  h_SimConvTwoMTracksAndVtxPGT0_[1] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimConvTwoMTracksRAndVtxPGT0";
  h_SimConvTwoMTracksAndVtxPGT0_[2] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated R", rBin, rMin, rMax);
  histname = "h_SimConvTwoMTracksZAndVtxPGT0";
  h_SimConvTwoMTracksAndVtxPGT0_[3] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated Z", zBin, zMin, zMax);
  histname = "h_SimConvTwoMTracksEtAndVtxPGT0";
  h_SimConvTwoMTracksAndVtxPGT0_[4] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated Et", etBin, etMin, etMax);
  //
  histname = "h_SimConvTwoMTracksEtaAndVtxPGT0005";
  h_SimConvTwoMTracksAndVtxPGT0005_[0] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_SimConvTwoMTracksPhiAndVtxPGT0005";
  h_SimConvTwoMTracksAndVtxPGT0005_[1] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimConvTwoMTracksRAndVtxPGT0005";
  h_SimConvTwoMTracksAndVtxPGT0005_[2] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated R", rBin, rMin, rMax);
  histname = "h_SimConvTwoMTracksZAndVtxPGT0005";
  h_SimConvTwoMTracksAndVtxPGT0005_[3] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated Z", zBin, zMin, zMax);
  histname = "h_SimConvTwoMTracksEtAndVtxPGT0005";
  h_SimConvTwoMTracksAndVtxPGT0005_[4] = iBooker.book1D(
      histname, " All vis conversions with 2 reco-matching tracks + vertex: simulated Et", etBin, etMin, etMax);

  if (!isRunCentrally_) {
    h_SimConvEtaPix_[0] = iBooker.book1D("simConvEtaPix", " sim converted Photon Eta: Pix ", etaBin, etaMin, etaMax);
    h_simTkPt_ = iBooker.book1D("simTkPt", "Sim conversion tracks pt ", etBin * 3, 0., etMax);
    h_simTkEta_ = iBooker.book1D("simTkEta", "Sim conversion tracks eta ", etaBin, etaMin, etaMax);
    h_simConvVtxRvsZ_[0] = iBooker.book2D("simConvVtxRvsZAll",
                                          " Photon Sim conversion vtx position",
                                          zBinForXray,
                                          zMinForXray,
                                          zMaxForXray,
                                          rBinForXray,
                                          rMinForXray,
                                          rMaxForXray);
    h_simConvVtxRvsZ_[1] = iBooker.book2D("simConvVtxRvsZBarrel",
                                          " Photon Sim conversion vtx position",
                                          zBinForXray,
                                          zMinForXray,
                                          zMaxForXray,
                                          rBinForXray,
                                          rMinForXray,
                                          rMaxForXray);
    h_simConvVtxRvsZ_[2] = iBooker.book2D("simConvVtxRvsZEndcap",
                                          " Photon Sim conversion vtx position",
                                          zBin2ForXray,
                                          zMinForXray,
                                          zMaxForXray,
                                          rBinForXray,
                                          rMinForXray,
                                          rMaxForXray);
    h_simConvVtxYvsX_ = iBooker.book2D(
        "simConvVtxYvsXTrkBarrel", " Photon Sim conversion vtx position, (x,y) eta<1 ", 100, -80., 80., 100, -80., 80.);
  }

  //// histograms for bkg
  histname = "h_SimJetEta";
  h_SimJet_[0] = iBooker.book1D(histname, " Jet bkg simulated #eta", etaBin, etaMin, etaMax);
  histname = "h_SimJetPhi";
  h_SimJet_[1] = iBooker.book1D(histname, " Jet bkg simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_SimJetEt";
  h_SimJet_[2] = iBooker.book1D(histname, " Jet bkg simulated Et", etBin, etMin, etMax);
  //
  histname = "h_MatchedSimJetEta";
  h_MatchedSimJet_[0] = iBooker.book1D(histname, " Matching jet simulated #eta", etaBin, etaMin, etaMax);
  histname = "h_MatchedSimJetPhi";
  h_MatchedSimJet_[1] = iBooker.book1D(histname, " Matching jet simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_MatchedSimJetEt";
  h_MatchedSimJet_[2] = iBooker.book1D(histname, " Matching jet simulated Et", etBin, etMin, etMax);
  //
  histname = "h_MatchedSimJetBadChEta";
  h_MatchedSimJetBadCh_[0] = iBooker.book1D(histname, " Matching jet simulated #eta", etaBin, etaMin, etaMax);
  histname = "h_MatchedSimJetBadChPhi";
  h_MatchedSimJetBadCh_[1] = iBooker.book1D(histname, " Matching jet simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_MatchedSimJetBadChEt";
  h_MatchedSimJetBadCh_[2] = iBooker.book1D(histname, " Matching jet simulated Et", etBin, etMin, etMax);

  iBooker.setCurrentFolder("EgammaV/" + fName_ + "/Background");

  histname = "nOfPhotons";
  h_nPho_ = iBooker.book1D(histname, "# of Reco photons per event ", 20, -0.5, 19.5);

  h_scBkgEta_ = iBooker.book1D("scBkgEta", " SC Bkg Eta ", etaBin, etaMin, etaMax);
  h_scBkgPhi_ = iBooker.book1D("scBkgPhi", " SC Bkg  Phi ", phiBin, phiMin, phiMax);
  //
  h_phoBkgEta_ = iBooker.book1D("phoBkgEta", " Photon Bkg Eta ", etaBin, etaMin, etaMax);
  h_phoBkgPhi_ = iBooker.book1D("phoBkgPhi", " Photon Bkg Phi ", phiBin, phiMin, phiMax);
  //
  h_phoBkgDEta_ = iBooker.book1D("phoBkgDEta", " Photon Eta(rec)-Eta(true) ", dEtaBin, dEtaMin, dEtaMax);
  h_phoBkgDPhi_ = iBooker.book1D("phoBkgDPhi", " Photon  Phi(rec)-Phi(true) ", dPhiBin, dPhiMin, dPhiMax);
  //
  histname = "phoBkgE";
  h_phoBkgE_[0] = iBooker.book1D(histname + "All", " Photon Bkg Energy: All ecal ", eBin, eMin, eMax);
  h_phoBkgE_[1] = iBooker.book1D(histname + "Barrel", " Photon Bkg Energy: barrel ", eBin, eMin, eMax);
  h_phoBkgE_[2] = iBooker.book1D(histname + "Endcap", " Photon Bkg Energy: Endcap ", eBin, eMin, eMax);
  //
  histname = "phoBkgEt";
  h_phoBkgEt_[0] = iBooker.book1D(histname + "All", " Photon Bkg Transverse Energy: All ecal ", etBin, etMin, etMax);
  h_phoBkgEt_[1] = iBooker.book1D(histname + "Barrel", " Photon Bkg Transverse Energy: Barrel ", etBin, etMin, etMax);
  h_phoBkgEt_[2] = iBooker.book1D(histname + "Endcap", " Photon BkgTransverse Energy: Endcap ", etBin, etMin, etMax);

  //
  histname = "scBkgE";
  h_scBkgE_[0] = iBooker.book1D(histname + "All", "    SC bkg Energy: All Ecal  ", eBin, eMin, eMax);
  h_scBkgE_[1] = iBooker.book1D(histname + "Barrel", " SC bkg Energy: Barrel ", eBin, eMin, eMax);
  h_scBkgE_[2] = iBooker.book1D(histname + "Endcap", " SC bkg Energy: Endcap ", eBin, eMin, eMax);
  histname = "scBkgEt";
  h_scBkgEt_[0] = iBooker.book1D(histname + "All", "    SC bkg Et: All Ecal  ", eBin, eMin, eMax);
  h_scBkgEt_[1] = iBooker.book1D(histname + "Barrel", " SC bkg Et: Barrel ", eBin, eMin, eMax);
  h_scBkgEt_[2] = iBooker.book1D(histname + "Endcap", " SC bkg Et: Endcap ", eBin, eMin, eMax);
  //
  histname = "r9Bkg";
  h_r9Bkg_[0] = iBooker.book1D(histname + "All", " r9 bkg: All Ecal", r9Bin, r9Min, r9Max);
  h_r9Bkg_[1] = iBooker.book1D(histname + "Barrel", " r9 bkg: Barrel ", r9Bin, r9Min, r9Max);
  h_r9Bkg_[2] = iBooker.book1D(histname + "Endcap", " r9 bkg: Endcap ", r9Bin, r9Min, r9Max);
  //
  histname = "R9VsEtaBkg";
  if (!isRunCentrally_)
    h2_r9VsEtaBkg_ =
        iBooker.book2D(histname + "All", " Bkg r9 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  //
  histname = "R9VsEtBkg";
  if (!isRunCentrally_)
    h2_r9VsEtBkg_ =
        iBooker.book2D(histname + "All", " Bkg photons r9 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  //
  histname = "r1Bkg";
  h_r1Bkg_[0] = iBooker.book1D(histname + "All", " Bkg photon e1x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r1Bkg_[1] = iBooker.book1D(histname + "Barrel", " Bkg photon e1x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r1Bkg_[2] = iBooker.book1D(histname + "Endcap", " Bkg photon e1x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  //
  histname = "R1VsEtaBkg";
  if (!isRunCentrally_)
    h2_r1VsEtaBkg_ = iBooker.book2D(
        histname + "All", " Bkg photons e1x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  histname = "pR1VsEtaBkg";
  if (!isRunCentrally_)
    p_r1VsEtaBkg_ = iBooker.bookProfile(
        histname + "All", " Bkg photons e1x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  //
  histname = "R1VsEtBkg";
  if (!isRunCentrally_)
    h2_r1VsEtBkg_ =
        iBooker.book2D(histname + "All", " Bkg photons e1x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  histname = "pR1VsEtBkg";
  if (!isRunCentrally_)
    p_r1VsEtBkg_ = iBooker.bookProfile(
        histname + "All", " Bkg photons e2x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  //
  histname = "r2Bkg";
  h_r2Bkg_[0] = iBooker.book1D(histname + "All", " Bkg photon e2x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r2Bkg_[1] = iBooker.book1D(histname + "Barrel", " Bkg photon e2x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r2Bkg_[2] = iBooker.book1D(histname + "Endcap", " Bkg photon e2x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  //
  histname = "R2VsEtaBkg";
  if (!isRunCentrally_)
    h2_r2VsEtaBkg_ = iBooker.book2D(
        histname + "All", " Bkg photons e2x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  histname = "pR2VsEtaBkg";
  if (!isRunCentrally_)
    p_r2VsEtaBkg_ = iBooker.bookProfile(
        histname + "All", " Bkg photons e2x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  //
  histname = "R2VsEtBkg";
  if (!isRunCentrally_)
    h2_r2VsEtBkg_ =
        iBooker.book2D(histname + "All", " Bkg photons e2x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  histname = "pR2VsEtBkg";
  if (!isRunCentrally_)
    p_r2VsEtBkg_ = iBooker.bookProfile(
        histname + "All", " Bkg photons e2x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);

  histname = "sigmaIetaIetaBkg";
  h_sigmaIetaIetaBkg_[0] = iBooker.book1D(histname + "All", "Bkg sigmaIetaIeta: All Ecal", 100, 0., 0.1);
  h_sigmaIetaIetaBkg_[1] = iBooker.book1D(histname + "Barrel", "Bkg sigmaIetaIeta: Barrel ", 100, 0., 0.05);
  h_sigmaIetaIetaBkg_[2] = iBooker.book1D(histname + "Endcap", "Bkg sigmaIetaIeta: Endcap ", 100, 0., 0.1);
  //
  histname = "sigmaIetaIetaVsEtaBkg";
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEtaBkg_ = iBooker.book2D(
        histname + "All", " Bkg photons sigmaIetaIeta vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  histname = "pSigmaIetaIetaVsEtaBkg";
  if (!isRunCentrally_)
    p_sigmaIetaIetaVsEtaBkg_ = iBooker.bookProfile(
        histname + "All", " Bkg photons sigmaIetaIeta vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  //
  histname = "sigmaIetaIetaVsEtBkg";
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEtBkg_[0] = iBooker.book2D(
        histname + "All", " Bkg photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEtBkg_[1] = iBooker.book2D(
        histname + "Barrel", " Bkg photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEtBkg_[2] = iBooker.book2D(
        histname + "Endcap", " Bkg photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  //
  histname = "pSigmaIetaIetaVsEtBkg";
  if (!isRunCentrally_)
    p_sigmaIetaIetaVsEtBkg_[0] = iBooker.bookProfile(
        histname + "All", " Bkg photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  if (!isRunCentrally_)
    p_sigmaIetaIetaVsEtBkg_[1] = iBooker.bookProfile(
        histname + "Barrel", " Bkg photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  if (!isRunCentrally_)
    p_sigmaIetaIetaVsEtBkg_[2] = iBooker.bookProfile(
        histname + "Endcap", " Bkg photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  //
  histname = "hOverEBkg";
  h_hOverEBkg_[0] = iBooker.book1D(histname + "All", "H/E bkg: All Ecal", 100, 0., 1.);
  h_hOverEBkg_[1] = iBooker.book1D(histname + "Barrel", "H/E bkg: Barrel ", 100, 0., 1.);
  h_hOverEBkg_[2] = iBooker.book1D(histname + "Endcap", "H/E bkg: Endcap ", 100, 0., 1.);
  //
  histname = "pHOverEVsEtaBkg";
  if (!isRunCentrally_)
    p_hOverEVsEtaBkg_ =
        iBooker.bookProfile(histname + "All", " Bkg H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  histname = "pHOverEVsEtBkg";
  if (!isRunCentrally_)
    p_hOverEVsEtBkg_ =
        iBooker.bookProfile(histname + "All", " Bkg photons H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  if (!isRunCentrally_) {
    histname = "hOverEVsEtaBkg";
    h2_hOverEVsEtaBkg_ =
        iBooker.book2D(histname + "All", " Bkg H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
    //
    histname = "hOverEVsEtBkg";
    h2_hOverEVsEtBkg_ =
        iBooker.book2D(histname + "All", " Bkg photons H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  }
  //
  histname = "ecalRecHitSumEtConeDR04Bkg";
  h_ecalRecHitSumEtConeDR04Bkg_[0] =
      iBooker.book1D(histname + "All", "bkg ecalRecHitSumEtDR04: All Ecal", etBin, etMin, 50.);
  h_ecalRecHitSumEtConeDR04Bkg_[1] =
      iBooker.book1D(histname + "Barrel", "bkg ecalRecHitSumEtDR04: Barrel ", etBin, etMin, 50.);
  h_ecalRecHitSumEtConeDR04Bkg_[2] =
      iBooker.book1D(histname + "Endcap", "bkg ecalRecHitSumEtDR04: Endcap ", etBin, etMin, 50.);
  //
  if (!isRunCentrally_) {
    histname = "ecalRecHitSumEtConeDR04VsEtaBkg";
    h2_ecalRecHitSumEtConeDR04VsEtaBkg_ = iBooker.book2D(histname + "All",
                                                         " bkg ecalRecHitSumEtDR04 vs #eta: all Ecal ",
                                                         etaBin2,
                                                         etaMin,
                                                         etaMax,
                                                         etBin,
                                                         etMin,
                                                         etMax * etScale);
    histname = "ecalRecHitSumEtConeDR04VsEtBkg";
    h2_ecalRecHitSumEtConeDR04VsEtBkg_[0] = iBooker.book2D(histname + "All",
                                                           " Bkg ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale);
    h2_ecalRecHitSumEtConeDR04VsEtBkg_[1] = iBooker.book2D(histname + "Barrel",
                                                           " Bkg ecalRecHitSumEtDR04 vs Et: Barrel ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale);
    h2_ecalRecHitSumEtConeDR04VsEtBkg_[2] = iBooker.book2D(histname + "Endcap",
                                                           " Bkg ecalRecHitSumEtDR04 vs Et: Endcap ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale);
    histname = "hcalTowerSumEtConeDR04VsEtaBkg";
    h2_hcalTowerSumEtConeDR04VsEtaBkg_ = iBooker.book2D(histname + "All",
                                                        " bkg hcalTowerSumEtDR04 vs #eta: all Ecal ",
                                                        etaBin2,
                                                        etaMin,
                                                        etaMax,
                                                        etBin,
                                                        etMin,
                                                        etMax * etScale);
    histname = "hcalTowerSumEtConeDR04VsEtBkg";
    h2_hcalTowerSumEtConeDR04VsEtBkg_[0] = iBooker.book2D(histname + "All",
                                                          " Bkg hcalTowerSumEtDR04 vs Et: all Ecal ",
                                                          etBin,
                                                          etMin,
                                                          etMax,
                                                          etBin,
                                                          etMin,
                                                          etMax * etScale);
    h2_hcalTowerSumEtConeDR04VsEtBkg_[1] = iBooker.book2D(histname + "Barrel",
                                                          " Bkg hcalTowerSumEtDR04 vs Et: Barrel ",
                                                          etBin,
                                                          etMin,
                                                          etMax,
                                                          etBin,
                                                          etMin,
                                                          etMax * etScale);
    h2_hcalTowerSumEtConeDR04VsEtBkg_[2] = iBooker.book2D(histname + "Endcap",
                                                          " Bkg hcalTowerSumEtDR04 vs Et: Endcap ",
                                                          etBin,
                                                          etMin,
                                                          etMax,
                                                          etBin,
                                                          etMin,
                                                          etMax * etScale);
  }

  histname = "pEcalRecHitSumEtConeDR04VsEtaBkg";
  if (!isRunCentrally_)
    p_ecalRecHitSumEtConeDR04VsEtaBkg_ = iBooker.bookProfile(histname + "All",
                                                             "bkg photons ecalRecHitSumEtDR04 vs #eta: all Ecal ",
                                                             etaBin2,
                                                             etaMin,
                                                             etaMax,
                                                             etBin,
                                                             etMin,
                                                             etMax * etScale,
                                                             "");
  //
  histname = "pEcalRecHitSumEtConeDR04VsEtBkg";
  if (!isRunCentrally_)
    p_ecalRecHitSumEtConeDR04VsEtBkg_[0] = iBooker.bookProfile(histname + "All",
                                                               "Bkg ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                               etBin,
                                                               etMin,
                                                               etMax,
                                                               etBin,
                                                               etMin,
                                                               etMax * etScale,
                                                               "");
  if (!isRunCentrally_)
    p_ecalRecHitSumEtConeDR04VsEtBkg_[1] = iBooker.bookProfile(histname + "Barrel",
                                                               "Bkg ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                               etBin,
                                                               etMin,
                                                               etMax,
                                                               etBin,
                                                               etMin,
                                                               etMax * etScale,
                                                               "");
  if (!isRunCentrally_)
    p_ecalRecHitSumEtConeDR04VsEtBkg_[2] = iBooker.bookProfile(histname + "Endcap",
                                                               "Bkg ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                               etBin,
                                                               etMin,
                                                               etMax,
                                                               etBin,
                                                               etMin,
                                                               etMax * etScale,
                                                               "");
  //
  histname = "hcalTowerSumEtConeDR04Bkg";
  h_hcalTowerSumEtConeDR04Bkg_[0] =
      iBooker.book1D(histname + "All", "bkg hcalTowerSumEtDR04: All Ecal", etBin, etMin, 20.);
  h_hcalTowerSumEtConeDR04Bkg_[1] =
      iBooker.book1D(histname + "Barrel", "bkg hcalTowerSumEtDR04: Barrel ", etBin, etMin, 20.);
  h_hcalTowerSumEtConeDR04Bkg_[2] =
      iBooker.book1D(histname + "Endcap", "bkg hcalTowerSumEtDR04: Endcap ", etBin, etMin, 20.);
  //
  histname = "pHcalTowerSumEtConeDR04VsEtaBkg";
  if (!isRunCentrally_)
    p_hcalTowerSumEtConeDR04VsEtaBkg_ = iBooker.bookProfile(histname + "All",
                                                            "bkg photons hcalTowerSumEtDR04 vs #eta: all Ecal ",
                                                            etaBin2,
                                                            etaMin,
                                                            etaMax,
                                                            etBin,
                                                            etMin,
                                                            etMax * etScale,
                                                            "");
  //
  histname = "pHcalTowerSumEtConeDR04VsEtBkg";
  if (!isRunCentrally_)
    p_hcalTowerSumEtConeDR04VsEtBkg_[0] = iBooker.bookProfile(histname + "All",
                                                              "Bkg hcalTowerSumEtDR04 vs Et: all Ecal ",
                                                              etBin,
                                                              etMin,
                                                              etMax,
                                                              etBin,
                                                              etMin,
                                                              etMax * etScale,
                                                              "");
  if (!isRunCentrally_)
    p_hcalTowerSumEtConeDR04VsEtBkg_[1] = iBooker.bookProfile(histname + "Barrel",
                                                              "Bkg hcalTowerSumEtDR04 vs Et: all Ecal ",
                                                              etBin,
                                                              etMin,
                                                              etMax,
                                                              etBin,
                                                              etMin,
                                                              etMax * etScale,
                                                              "");
  if (!isRunCentrally_)
    p_hcalTowerSumEtConeDR04VsEtBkg_[2] = iBooker.bookProfile(histname + "Endcap",
                                                              "Bkg hcalTowerSumEtDR04 vs Et: all Ecal ",
                                                              etBin,
                                                              etMin,
                                                              etMax,
                                                              etBin,
                                                              etMin,
                                                              etMax * etScale,
                                                              "");
  //
  histname = "isoTrkSolidConeDR04Bkg";
  h_isoTrkSolidConeDR04Bkg_[0] =
      iBooker.book1D(histname + "All", "isoTrkSolidConeDR04 Bkg: All Ecal", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04Bkg_[1] =
      iBooker.book1D(histname + "Barrel", "isoTrkSolidConeDR04 Bkg: Barrel ", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04Bkg_[2] =
      iBooker.book1D(histname + "Endcap", "isoTrkSolidConeDR04 Bkg: Endcap ", etBin, etMin, etMax * 0.1);
  //
  histname = "isoTrkSolidConeDR04VsEtaBkg";
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEtaBkg_ = iBooker.book2D(histname + "All",
                                                     " Bkg photons isoTrkSolidConeDR04 vs #eta: all Ecal ",
                                                     etaBin2,
                                                     etaMin,
                                                     etaMax,
                                                     etBin,
                                                     etMin,
                                                     etMax * 0.1);
  histname = "pIsoTrkSolidConeDR04VsEtaBkg";
  if (!isRunCentrally_)
    p_isoTrkSolidConeDR04VsEtaBkg_ = iBooker.bookProfile(histname + "All",
                                                         " Bkg photons isoTrkSolidConeDR04 vs #eta: all Ecal ",
                                                         etaBin2,
                                                         etaMin,
                                                         etaMax,
                                                         etBin,
                                                         etMin,
                                                         etMax * 0.1);
  //
  histname = "isoTrkSolidConeDR04VsEtBkg";
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEtBkg_[0] = iBooker.book2D(histname + "All",
                                                       " Bkg photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                       etBin,
                                                       etMin,
                                                       etMax,
                                                       etBin,
                                                       etMin,
                                                       etMax * 0.1);
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEtBkg_[1] = iBooker.book2D(histname + "Barrel",
                                                       " Bkg photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                       etBin,
                                                       etMin,
                                                       etMax,
                                                       etBin,
                                                       etMin,
                                                       etMax * 0.1);
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEtBkg_[2] = iBooker.book2D(histname + "Endcap",
                                                       " Bkg photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                       etBin,
                                                       etMin,
                                                       etMax,
                                                       etBin,
                                                       etMin,
                                                       etMax * 0.1);
  histname = "pIsoTrkSolidConeDR04VsEtBkg";
  if (!isRunCentrally_)
    p_isoTrkSolidConeDR04VsEtBkg_[0] = iBooker.bookProfile(histname + "All",
                                                           " Bkg photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * 0.1);
  if (!isRunCentrally_)
    p_isoTrkSolidConeDR04VsEtBkg_[1] = iBooker.bookProfile(histname + "Barrel",
                                                           " Bkg photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * 0.1);
  if (!isRunCentrally_)
    p_isoTrkSolidConeDR04VsEtBkg_[2] = iBooker.bookProfile(histname + "Endcap",
                                                           " Bkg photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * 0.1);
  //
  histname = "nTrkSolidConeDR04Bkg";
  h_nTrkSolidConeDR04Bkg_[0] = iBooker.book1D(histname + "All", "Bkg nTrkSolidConeDR04: All Ecal", 20, 0., 20);
  h_nTrkSolidConeDR04Bkg_[1] = iBooker.book1D(histname + "Barrel", "Bkg nTrkSolidConeDR04: Barrel ", 20, 0., 20);
  h_nTrkSolidConeDR04Bkg_[2] = iBooker.book1D(histname + "Endcap", "Bkg nTrkSolidConeDR04: Endcap ", 20, 0., 20);
  //
  histname = "nTrkSolidConeDR04VsEtaBkg";
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEtaBkg_ = iBooker.book2D(
        histname + "All", " Bkg photons nTrkSolidConeDR04 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 20, 0., 20);
  histname = "p_nTrkSolidConeDR04VsEtaBkg";
  if (!isRunCentrally_)
    p_nTrkSolidConeDR04VsEtaBkg_ = iBooker.bookProfile(
        histname + "All", " Bkg photons nTrkSolidConeDR04 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 20, 0., 20);
  //
  histname = "nTrkSolidConeDR04VsEtBkg";
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEtBkg_[0] = iBooker.book2D(
        histname + "All", "Bkg photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEtBkg_[1] = iBooker.book2D(
        histname + "Barrel", "Bkg photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEtBkg_[2] = iBooker.book2D(
        histname + "Endcap", "Bkg photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  //
  histname = "pnTrkSolidConeDR04VsEtBkg";
  if (!isRunCentrally_)
    p_nTrkSolidConeDR04VsEtBkg_[0] = iBooker.bookProfile(
        histname + "All", "Bkg photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  if (!isRunCentrally_)
    p_nTrkSolidConeDR04VsEtBkg_[1] = iBooker.bookProfile(
        histname + "Barrel", "Bkg photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  if (!isRunCentrally_)
    p_nTrkSolidConeDR04VsEtBkg_[2] = iBooker.bookProfile(
        histname + "Endcap", "Bkg photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  //
  h_convEtaBkg_ = iBooker.book1D("convEtaBkg", " converted Photon Bkg Eta 2 tracks", etaBin, etaMin, etaMax);
  h_convPhiBkg_ = iBooker.book1D("convPhiBkg", " converted Photon Bkg Phi ", phiBin, phiMin, phiMax);
  //
  histname = "mvaOutBkg";
  h_mvaOutBkg_[0] = iBooker.book1D(histname + "All", " mvaOut  conversions bkg : All Ecal", 100, 0., 1.);
  h_mvaOutBkg_[1] = iBooker.book1D(histname + "Barrel", " mvaOut conversions bkg: Barrel Ecal", 100, 0., 1.);
  h_mvaOutBkg_[2] = iBooker.book1D(histname + "Endcap", " mvaOut  conversions bkg: Endcap Ecal", 100, 0., 1.);

  histname = "PoverEtracksBkg";
  h_PoverETracksBkg_[0] =
      iBooker.book1D(histname + "All", " bkg photons conversion p/E: all Ecal ", povereBin, povereMin, povereMax);
  h_PoverETracksBkg_[1] =
      iBooker.book1D(histname + "Barrel", "bkg photons conversion p/E: Barrel Ecal", povereBin, povereMin, povereMax);
  h_PoverETracksBkg_[2] =
      iBooker.book1D(histname + "Endcap", " bkg photons conversion p/E: Endcap Ecal ", povereBin, povereMin, povereMax);

  histname = "EoverPtracksBkg";
  h_EoverPTracksBkg_[0] =
      iBooker.book1D(histname + "All", " bkg photons conversion E/p: all Ecal ", eoverpBin, eoverpMin, eoverpMax);
  h_EoverPTracksBkg_[1] =
      iBooker.book1D(histname + "Barrel", "bkg photons conversion E/p: Barrel Ecal", eoverpBin, eoverpMin, eoverpMax);
  h_EoverPTracksBkg_[2] =
      iBooker.book1D(histname + "Endcap", " bkg photons conversion E/p: Endcap Ecal ", eoverpBin, eoverpMin, eoverpMax);

  histname = "hDCotTracksBkg";
  h_DCotTracksBkg_[0] = iBooker.book1D(histname + "All",
                                       " bkg Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",
                                       dCotTracksBin,
                                       dCotTracksMin,
                                       dCotTracksMax);
  h_DCotTracksBkg_[1] = iBooker.book1D(histname + "Barrel",
                                       " bkg Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",
                                       dCotTracksBin,
                                       dCotTracksMin,
                                       dCotTracksMax);
  h_DCotTracksBkg_[2] = iBooker.book1D(histname + "Endcap",
                                       " bkg Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",
                                       dCotTracksBin,
                                       dCotTracksMin,
                                       dCotTracksMax);

  histname = "hDPhiTracksAtVtxBkg";
  h_DPhiTracksAtVtxBkg_[0] =
      iBooker.book1D(histname + "All",
                     " Bkg Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",
                     dPhiTracksBin,
                     dPhiTracksMin,
                     dPhiTracksMax);
  h_DPhiTracksAtVtxBkg_[1] =
      iBooker.book1D(histname + "Barrel",
                     " Bkg Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",
                     dPhiTracksBin,
                     dPhiTracksMin,
                     dPhiTracksMax);
  h_DPhiTracksAtVtxBkg_[2] =
      iBooker.book1D(histname + "Endcap",
                     " Bkg Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",
                     dPhiTracksBin,
                     dPhiTracksMin,
                     dPhiTracksMax);

  if (!isRunCentrally_) {
    h_convVtxRvsZBkg_[0] = iBooker.book2D("convVtxRvsZAllBkg",
                                          " Bkg Photon Reco conversion vtx position",
                                          zBinForXray,
                                          zMinForXray,
                                          zMaxForXray,
                                          rBinForXray,
                                          rMinForXray,
                                          rMaxForXray);
    h_convVtxRvsZBkg_[1] = iBooker.book2D("convVtxRvsZBarrelBkg",
                                          " Bkg Photon Reco conversion vtx position",
                                          zBinForXray,
                                          zMinForXray,
                                          zMaxForXray,
                                          rBinForXray,
                                          rMinForXray,
                                          rMaxForXray);
    h_convVtxYvsXBkg_ = iBooker.book2D("convVtxYvsXTrkBarrelBkg",
                                       " Bkg Photon Reco conversion vtx position, (x,y) eta<1 ",
                                       100,
                                       -80.,
                                       80.,
                                       100,
                                       -80.,
                                       80.);
  }

  //
  iBooker.setCurrentFolder("EgammaV/" + fName_ + "/Photons");

  histname = "nOfflineVtx";
  h_nRecoVtx_ = iBooker.book1D(histname, "# of Offline Vertices", 200, -0.5, 199.5);

  h_phoEta_[0] = iBooker.book1D("phoEta", " Photon Eta ", etaBin, etaMin, etaMax);
  h_phoPhi_[0] = iBooker.book1D("phoPhi", " Photon  Phi ", phiBin, phiMin, phiMax);

  h_phoDEta_[0] = iBooker.book1D("phoDEta", " Photon Eta(rec)-Eta(true) ", dEtaBin, dEtaMin, dEtaMax);
  h_phoDPhi_[0] = iBooker.book1D("phoDPhi", " Photon  Phi(rec)-Phi(true) ", dPhiBin, dPhiMin, dPhiMax);

  h_scEta_[0] = iBooker.book1D("scEta", " SC Eta ", etaBin, etaMin, etaMax);
  h_scPhi_[0] = iBooker.book1D("scPhi", " SC Phi ", phiBin, phiMin, phiMax);

  if (!isRunCentrally_) {
    h_scEtaWidth_[0] = iBooker.book1D("scEtaWidth", " SC Eta Width ", 100, 0., 0.1);
    h_scPhiWidth_[0] = iBooker.book1D("scPhiWidth", " SC Phi Width ", 100, 0., 1.);
  }

  histname = "scE";
  h_scE_[0][0] = iBooker.book1D(histname + "All", " SC Energy: All Ecal  ", eBin, eMin, eMax);
  h_scE_[0][1] = iBooker.book1D(histname + "Barrel", " SC Energy: Barrel ", eBin, eMin, eMax);
  h_scE_[0][2] = iBooker.book1D(histname + "Endcap", " SC Energy: Endcap ", eBin, eMin, eMax);

  histname = "psE";
  h_psE_ = iBooker.book1D(histname + "Endcap", " ES Energy  ", eBin, eMin, 50.);

  histname = "scEt";
  h_scEt_[0][0] = iBooker.book1D(histname + "All", " SC Et: All Ecal ", etBin, etMin, etMax);
  h_scEt_[0][1] = iBooker.book1D(histname + "Barrel", " SC Et: Barrel", etBin, etMin, etMax);
  h_scEt_[0][2] = iBooker.book1D(histname + "Endcap", " SC Et: Endcap", etBin, etMin, etMax);

  histname = "r9";
  h_r9_[0][0] = iBooker.book1D(histname + "All", " r9: All Ecal", r9Bin, r9Min, r9Max);
  h_r9_[0][1] = iBooker.book1D(histname + "Barrel", " r9: Barrel ", r9Bin, r9Min, r9Max);
  h_r9_[0][2] = iBooker.book1D(histname + "Endcap", " r9: Endcap ", r9Bin, r9Min, r9Max);
  //

  if (!isRunCentrally_) {
    histname = "r9ConvFromMC";
    h_r9_[1][0] = iBooker.book1D(histname + "All", " r9: All Ecal", r9Bin, r9Min, r9Max);
    h_r9_[1][1] = iBooker.book1D(histname + "Barrel", " r9: Barrel ", r9Bin, r9Min, r9Max);
    h_r9_[1][2] = iBooker.book1D(histname + "Endcap", " r9: Endcap ", r9Bin, r9Min, r9Max);
    //
    histname = "r9ConvFromReco";
    h_r9_[2][0] = iBooker.book1D(histname + "All", " r9: All Ecal", r9Bin, r9Min, r9Max);
    h_r9_[2][1] = iBooker.book1D(histname + "Barrel", " r9: Barrel ", r9Bin, r9Min, r9Max);
    h_r9_[2][2] = iBooker.book1D(histname + "Endcap", " r9: Endcap ", r9Bin, r9Min, r9Max);
    //////
    histname = "EtR9Less093";
    h_EtR9Less093_[0][0] = iBooker.book1D(histname + "All", " r9 < 0.94 or 0.95 : All Ecal", etBin, etMin, etMax);
    h_EtR9Less093_[0][1] = iBooker.book1D(histname + "Barrel", " r9 < 0.94 : Barrel ", etBin, etMin, etMax);
    h_EtR9Less093_[0][2] = iBooker.book1D(histname + "Endcap", " r9 < 0.95 : Endcap ", etBin, etMin, etMax);
    histname = "EtR9Less093Conv";
    h_EtR9Less093_[1][0] =
        iBooker.book1D(histname + "All", " r9 < 0.94, 0.95 and good conv : All Ecal", etBin, etMin, etMax);
    h_EtR9Less093_[1][1] =
        iBooker.book1D(histname + "Barrel", " r9 < 0.94 and good conv : Barrel ", etBin, etMin, etMax);
    h_EtR9Less093_[1][2] =
        iBooker.book1D(histname + "Endcap", " r9 < 0.95 and good conv : Endcap ", etBin, etMin, etMax);
  }

  /////    //
  histname = "pR9VsEta";
  p_r9VsEta_[0] = iBooker.bookProfile(
      histname + "All", " All photons r9 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  p_r9VsEta_[1] = iBooker.bookProfile(
      histname + "Unconv", " Unconv photons r9 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  p_r9VsEta_[2] = iBooker.bookProfile(
      histname + "Conv", " Conv photons r9 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  //
  histname = "R9VsEt";
  if (!isRunCentrally_)
    h2_r9VsEt_[0] =
        iBooker.book2D(histname + "All", " All photons r9 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  if (!isRunCentrally_)
    h2_r9VsEt_[1] =
        iBooker.book2D(histname + "Unconv", " All photons r9 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  //
  histname = "r1";
  h_r1_[0][0] = iBooker.book1D(histname + "All", " e1x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r1_[0][1] = iBooker.book1D(histname + "Barrel", " e1x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r1_[0][2] = iBooker.book1D(histname + "Endcap", " e1x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  //
  histname = "R1VsEta";
  if (!isRunCentrally_)
    h2_r1VsEta_[0] = iBooker.book2D(
        histname + "All", " All photons e1x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  if (!isRunCentrally_)
    h2_r1VsEta_[1] = iBooker.book2D(
        histname + "Unconv", " All photons e1x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  //
  histname = "R1VsEt";
  if (!isRunCentrally_)
    h2_r1VsEt_[0] =
        iBooker.book2D(histname + "All", " All photons e1x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  if (!isRunCentrally_)
    h2_r1VsEt_[1] = iBooker.book2D(
        histname + "Unconv", " All photons e1x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  //
  histname = "r2";
  h_r2_[0][0] = iBooker.book1D(histname + "All", " e2x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r2_[0][1] = iBooker.book1D(histname + "Barrel", " e2x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r2_[0][2] = iBooker.book1D(histname + "Endcap", " e2x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  //
  histname = "R2VsEta";
  if (!isRunCentrally_)
    h2_r2VsEta_[0] = iBooker.book2D(
        histname + "All", " All photons e2x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  if (!isRunCentrally_)
    h2_r2VsEta_[1] = iBooker.book2D(
        histname + "Unconv", " All photons e2x5/e5x5 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 1.1);
  //
  histname = "R2VsEt";
  if (!isRunCentrally_)
    h2_r2VsEt_[0] =
        iBooker.book2D(histname + "All", " All photons e2x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  if (!isRunCentrally_)
    h2_r2VsEt_[1] = iBooker.book2D(
        histname + "Unconv", " All photons e2x5/e5x5 vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 1.1);
  //
  histname = "sigmaIetaIeta";
  h_sigmaIetaIeta_[0][0] = iBooker.book1D(histname + "All", "sigmaIetaIeta: All Ecal", 100, 0., 0.1);
  h_sigmaIetaIeta_[0][1] = iBooker.book1D(histname + "Barrel", "sigmaIetaIeta: Barrel ", 100, 0., 0.05);
  h_sigmaIetaIeta_[0][2] = iBooker.book1D(histname + "Endcap", "sigmaIetaIeta: Endcap ", 100, 0., 0.1);
  //
  histname = "sigmaIetaIetaVsEta";
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEta_[0] = iBooker.book2D(
        histname + "All", " All photons sigmaIetaIeta vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEta_[1] = iBooker.book2D(
        histname + "Unconv", " All photons sigmaIetaIeta vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  //
  histname = "sigmaIetaIetaVsEt";
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEt_[0] = iBooker.book2D(
        histname + "All", " All photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  if (!isRunCentrally_)
    h2_sigmaIetaIetaVsEt_[1] = iBooker.book2D(
        histname + "Unconv", " All photons sigmaIetaIeta vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  //
  histname = "hOverE";
  h_hOverE_[0][0] = iBooker.book1D(histname + "All", "H/E: All Ecal", 100, 0., 0.1);
  h_hOverE_[0][1] = iBooker.book1D(histname + "Barrel", "H/E: Barrel ", 100, 0., 0.1);
  h_hOverE_[0][2] = iBooker.book1D(histname + "Endcap", "H/E: Endcap ", 100, 0., 0.1);
  //
  histname = "newhOverE";
  h_newhOverE_[0][0] = iBooker.book1D(histname + "All", "new H/E: All Ecal", 100, 0., 0.1);
  h_newhOverE_[0][1] = iBooker.book1D(histname + "Barrel", "new H/E: Barrel ", 100, 0., 0.1);
  h_newhOverE_[0][2] = iBooker.book1D(histname + "Endcap", "new H/E: Endcap ", 100, 0., 0.1);

  //
  if (!isRunCentrally_) {
    histname = "hOverEVsEta";
    h2_hOverEVsEta_[0] =
        iBooker.book2D(histname + "All", " All photons H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
    h2_hOverEVsEta_[1] = iBooker.book2D(
        histname + "Unconv", " All photons H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
    //
    histname = "hOverEVsEt";
    h2_hOverEVsEt_[0] =
        iBooker.book2D(histname + "All", " All photons H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
    h2_hOverEVsEt_[1] =
        iBooker.book2D(histname + "Unconv", " All photons H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
    //
  }
  histname = "pHoverEVsEta";
  p_hOverEVsEta_[0] = iBooker.bookProfile(
      histname + "All", " All photons H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  p_hOverEVsEta_[1] = iBooker.bookProfile(
      histname + "Unconv", " All photons H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  //
  histname = "pHoverEVsEt";
  p_hOverEVsEt_[0] =
      iBooker.bookProfile(histname + "All", " All photons H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  p_hOverEVsEt_[1] =
      iBooker.bookProfile(histname + "Unconv", " All photons H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  //
  histname = "pnewHoverEVsEta";
  p_newhOverEVsEta_[0] = iBooker.bookProfile(
      histname + "All", " All photons new H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  p_newhOverEVsEta_[1] = iBooker.bookProfile(
      histname + "Unconv", " All photons new H/E vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 0.1);
  //
  histname = "pnewHoverEVsEt";
  p_newhOverEVsEt_[0] =
      iBooker.bookProfile(histname + "All", " All photons new H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  p_newhOverEVsEt_[1] = iBooker.bookProfile(
      histname + "Unconv", " All photons new H/E vs Et: all Ecal ", etBin, etMin, etMax, 100, 0., 0.1);
  //
  histname = "ecalRecHitSumEtConeDR04";
  h_ecalRecHitSumEtConeDR04_[0][0] =
      iBooker.book1D(histname + "All", "ecalRecHitSumEtDR04: All Ecal", etBin, etMin, 50.);
  h_ecalRecHitSumEtConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel", "ecalRecHitSumEtDR04: Barrel ", etBin, etMin, 50.);
  h_ecalRecHitSumEtConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap", "ecalRecHitSumEtDR04: Endcap ", etBin, etMin, 50.);
  //

  if (!isRunCentrally_) {
    histname = "ecalRecHitSumEtConeDR04VsEta";
    h2_ecalRecHitSumEtConeDR04VsEta_[0] = iBooker.book2D(histname + "All",
                                                         " All photons ecalRecHitSumEtDR04 vs #eta: all Ecal ",
                                                         etaBin2,
                                                         etaMin,
                                                         etaMax,
                                                         etBin,
                                                         etMin,
                                                         etMax * etScale);
    h2_ecalRecHitSumEtConeDR04VsEta_[1] = iBooker.book2D(histname + "Unconv",
                                                         " All photons ecalRecHitSumEtDR04 vs #eta: all Ecal ",
                                                         etaBin2,
                                                         etaMin,
                                                         etaMax,
                                                         etBin,
                                                         etMin,
                                                         etMax * etScale);
  }
  histname = "pEcalRecHitSumEtConeDR04VsEta";
  p_ecalRecHitSumEtConeDR04VsEta_[0] = iBooker.bookProfile(histname + "All",
                                                           "All photons ecalRecHitSumEtDR04 vs #eta: all Ecal ",
                                                           etaBin2,
                                                           etaMin,
                                                           etaMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale,
                                                           "");
  p_ecalRecHitSumEtConeDR04VsEta_[1] = iBooker.bookProfile(histname + "Unconv",
                                                           "All photons ecalRecHitSumEtDR04 vs #eta: all Ecal ",
                                                           etaBin2,
                                                           etaMin,
                                                           etaMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale,
                                                           "");
  //
  if (!isRunCentrally_) {
    histname = "ecalRecHitSumEtConeDR04VsEt";
    h2_ecalRecHitSumEtConeDR04VsEt_[0] = iBooker.book2D(histname + "All",
                                                        " All photons ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                        etBin,
                                                        etMin,
                                                        etMax,
                                                        etBin,
                                                        etMin,
                                                        etMax * etScale);
    h2_ecalRecHitSumEtConeDR04VsEt_[1] = iBooker.book2D(histname + "Barrel",
                                                        " All photons ecalRecHitSumEtDR04 vs Et: Barrel ",
                                                        etBin,
                                                        etMin,
                                                        etMax,
                                                        etBin,
                                                        etMin,
                                                        etMax * etScale);
    h2_ecalRecHitSumEtConeDR04VsEt_[2] = iBooker.book2D(histname + "Endcap",
                                                        " All photons ecalRecHitSumEtDR04 vs Et: Endcap ",
                                                        etBin,
                                                        etMin,
                                                        etMax,
                                                        etBin,
                                                        etMin,
                                                        etMax * etScale);
  }
  histname = "pEcalRecHitSumEtConeDR04VsEt";
  if (!isRunCentrally_)
    p_ecalRecHitSumEtConeDR04VsEt_[0] = iBooker.bookProfile(histname + "All",
                                                            "All photons ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                            etBin,
                                                            etMin,
                                                            etMax,
                                                            etBin,
                                                            etMin,
                                                            etMax * etScale,
                                                            "");
  p_ecalRecHitSumEtConeDR04VsEt_[1] = iBooker.bookProfile(histname + "Barrel",
                                                          "All photons ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                          etBin,
                                                          etMin,
                                                          etMax,
                                                          etBin,
                                                          etMin,
                                                          etMax * etScale,
                                                          "");
  p_ecalRecHitSumEtConeDR04VsEt_[2] = iBooker.bookProfile(histname + "Endcap",
                                                          "All photons ecalRecHitSumEtDR04 vs Et: all Ecal ",
                                                          etBin,
                                                          etMin,
                                                          etMax,
                                                          etBin,
                                                          etMin,
                                                          etMax * etScale,
                                                          "");
  //
  histname = "hcalTowerSumEtConeDR04";
  h_hcalTowerSumEtConeDR04_[0][0] =
      iBooker.book1D(histname + "All", "hcalTowerSumEtConeDR04: All Ecal", etBin, etMin, 50.);
  h_hcalTowerSumEtConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel", "hcalTowerSumEtConeDR04: Barrel ", etBin, etMin, 50.);
  h_hcalTowerSumEtConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap", "hcalTowerSumEtConeDR04: Endcap ", etBin, etMin, 50.);
  //
  histname = "hcalTowerBcSumEtConeDR04";
  if (!isRunCentrally_)
    h_hcalTowerBcSumEtConeDR04_[0][0] =
        iBooker.book1D(histname + "All", "hcalTowerBcSumEtConeDR04: All Ecal", etBin, etMin, 50.);
  h_hcalTowerBcSumEtConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel", "hcalTowerBcSumEtConeDR04: Barrel ", etBin, etMin, 50.);
  h_hcalTowerBcSumEtConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap", "hcalTowerBcSumEtConeDR04: Endcap ", etBin, etMin, 50.);

  //
  if (!isRunCentrally_) {
    histname = "hcalTowerSumEtConeDR04VsEta";
    h2_hcalTowerSumEtConeDR04VsEta_[0] = iBooker.book2D(histname + "All",
                                                        " All photons hcalTowerSumEtConeDR04 vs #eta: all Ecal ",
                                                        etaBin2,
                                                        etaMin,
                                                        etaMax,
                                                        etBin,
                                                        etMin,
                                                        etMax * 0.1);
    h2_hcalTowerSumEtConeDR04VsEta_[1] = iBooker.book2D(histname + "Unconv",
                                                        " All photons hcalTowerSumEtConeDR04 vs #eta: all Ecal ",
                                                        etaBin2,
                                                        etaMin,
                                                        etaMax,
                                                        etBin,
                                                        etMin,
                                                        etMax * 0.1);
  }
  histname = "pHcalTowerSumEtConeDR04VsEta";
  p_hcalTowerSumEtConeDR04VsEta_[0] = iBooker.bookProfile(histname + "All",
                                                          "All photons hcalTowerSumEtDR04 vs #eta: all Ecal ",
                                                          etaBin2,
                                                          etaMin,
                                                          etaMax,
                                                          etBin,
                                                          etMin,
                                                          etMax * 0.1,
                                                          "");
  p_hcalTowerSumEtConeDR04VsEta_[1] = iBooker.bookProfile(histname + "Unconv",
                                                          "All photons hcalTowerSumEtDR04 vs #eta: all Ecal ",
                                                          etaBin2,
                                                          etaMin,
                                                          etaMax,
                                                          etBin,
                                                          etMin,
                                                          etMax * 0.1,
                                                          "");
  histname = "pHcalTowerBcSumEtConeDR04VsEta";
  p_hcalTowerBcSumEtConeDR04VsEta_[0] = iBooker.bookProfile(histname + "All",
                                                            "All photons hcalTowerBcSumEtDR04 vs #eta: all Ecal ",
                                                            etaBin2,
                                                            etaMin,
                                                            etaMax,
                                                            etBin,
                                                            etMin,
                                                            etMax * 0.1,
                                                            "");
  p_hcalTowerBcSumEtConeDR04VsEta_[1] = iBooker.bookProfile(histname + "Unconv",
                                                            "All photons hcalTowerBcSumEtDR04 vs #eta: all Ecal ",
                                                            etaBin2,
                                                            etaMin,
                                                            etaMax,
                                                            etBin,
                                                            etMin,
                                                            etMax * 0.1,
                                                            "");
  //
  if (!isRunCentrally_) {
    histname = "hcalTowerSumEtConeDR04VsEt";
    h2_hcalTowerSumEtConeDR04VsEt_[0] = iBooker.book2D(histname + "All",
                                                       " All photons hcalTowerSumEtConeDR04 vs Et: all Ecal ",
                                                       etBin,
                                                       etMin,
                                                       etMax,
                                                       etBin,
                                                       etMin,
                                                       etMax * 0.1);
    h2_hcalTowerSumEtConeDR04VsEt_[1] = iBooker.book2D(histname + "Barrel",
                                                       " All photons hcalTowerSumEtConeDR04 vs Et: Barrel ",
                                                       etBin,
                                                       etMin,
                                                       etMax,
                                                       etBin,
                                                       etMin,
                                                       etMax * 0.1);
    h2_hcalTowerSumEtConeDR04VsEt_[2] = iBooker.book2D(histname + "Endcap",
                                                       " All photons hcalTowerSumEtConeDR04 vs Et: Endcap ",
                                                       etBin,
                                                       etMin,
                                                       etMax,
                                                       etBin,
                                                       etMin,
                                                       etMax * 0.1);
  }
  histname = "pHcalTowerSumEtConeDR04VsEt";
  if (!isRunCentrally_)
    p_hcalTowerSumEtConeDR04VsEt_[0] = iBooker.bookProfile(histname + "All",
                                                           "All photons hcalTowerSumEtDR04 vs Et: all Ecal ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale,
                                                           "");
  p_hcalTowerSumEtConeDR04VsEt_[1] = iBooker.bookProfile(histname + "Barrel",
                                                         "All photons hcalTowerSumEtDR04 vs Et: all Ecal ",
                                                         etBin,
                                                         etMin,
                                                         etMax,
                                                         etBin,
                                                         etMin,
                                                         etMax * etScale,
                                                         "");
  p_hcalTowerSumEtConeDR04VsEt_[2] = iBooker.bookProfile(histname + "Endcap",
                                                         "All photons hcalTowerSumEtDR04 vs Et: all Ecal ",
                                                         etBin,
                                                         etMin,
                                                         etMax,
                                                         etBin,
                                                         etMin,
                                                         etMax * etScale,
                                                         "");
  //
  histname = "pHcalTowerBcSumEtConeDR04VsEt";
  if (!isRunCentrally_)
    p_hcalTowerBcSumEtConeDR04VsEt_[0] = iBooker.bookProfile(histname + "All",
                                                             "All photons hcalTowerBcSumEtDR04 vs Et: all Ecal ",
                                                             etBin,
                                                             etMin,
                                                             etMax,
                                                             etBin,
                                                             etMin,
                                                             etMax * etScale,
                                                             "");
  p_hcalTowerBcSumEtConeDR04VsEt_[1] = iBooker.bookProfile(histname + "Barrel",
                                                           "All photons hcalTowerBcSumEtDR04 vs Et: all Ecal ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale,
                                                           "");
  p_hcalTowerBcSumEtConeDR04VsEt_[2] = iBooker.bookProfile(histname + "Endcap",
                                                           "All photons hcalTowerBcSumEtDR04 vs Et: all Ecal ",
                                                           etBin,
                                                           etMin,
                                                           etMax,
                                                           etBin,
                                                           etMin,
                                                           etMax * etScale,
                                                           "");

  //
  histname = "isoTrkSolidConeDR04";
  h_isoTrkSolidConeDR04_[0][0] =
      iBooker.book1D(histname + "All", "isoTrkSolidConeDR04: All Ecal", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel", "isoTrkSolidConeDR04: Barrel ", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap", "isoTrkSolidConeDR04: Endcap ", etBin, etMin, etMax * 0.1);
  //

  histname = "isoTrkSolidConeDR04VsEta";
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEta_[0] = iBooker.book2D(histname + "All",
                                                     " All photons isoTrkSolidConeDR04 vs #eta: all Ecal ",
                                                     etaBin2,
                                                     etaMin,
                                                     etaMax,
                                                     etBin,
                                                     etMin,
                                                     etMax * 0.1);
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEta_[1] = iBooker.book2D(histname + "Unconv",
                                                     " All photons isoTrkSolidConeDR04 vs #eta: all Ecal ",
                                                     etaBin2,
                                                     etaMin,
                                                     etaMax,
                                                     etBin,
                                                     etMin,
                                                     etMax * 0.1);

  //
  histname = "isoTrkSolidConeDR04VsEt";
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEt_[0] = iBooker.book2D(histname + "All",
                                                    " All photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                    etBin,
                                                    etMin,
                                                    etMax,
                                                    etBin,
                                                    etMin,
                                                    etMax * 0.1);
  if (!isRunCentrally_)
    h2_isoTrkSolidConeDR04VsEt_[1] = iBooker.book2D(histname + "Unconv",
                                                    " All photons isoTrkSolidConeDR04 vs Et: all Ecal ",
                                                    etBin,
                                                    etMin,
                                                    etMax,
                                                    etBin,
                                                    etMin,
                                                    etMax * 0.1);
  //
  histname = "nTrkSolidConeDR04";
  h_nTrkSolidConeDR04_[0][0] = iBooker.book1D(histname + "All", "nTrkSolidConeDR04: All Ecal", 20, 0., 20);
  h_nTrkSolidConeDR04_[0][1] = iBooker.book1D(histname + "Barrel", "nTrkSolidConeDR04: Barrel ", 20, 0., 20);
  h_nTrkSolidConeDR04_[0][2] = iBooker.book1D(histname + "Endcap", "nTrkSolidConeDR04: Endcap ", 20, 0., 20);
  //
  histname = "nTrkSolidConeDR04VsEta";
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEta_[0] = iBooker.book2D(
        histname + "All", " All photons nTrkSolidConeDR04 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 20, 0., 20);
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEta_[1] = iBooker.book2D(
        histname + "Unconv", " All photons nTrkSolidConeDR04 vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 20, 0., 20);
  //
  histname = "nTrkSolidConeDR04VsEt";
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEt_[0] = iBooker.book2D(
        histname + "All", " All photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  if (!isRunCentrally_)
    h2_nTrkSolidConeDR04VsEt_[1] = iBooker.book2D(
        histname + "Unconv", " All photons nTrkSolidConeDR04 vs Et: all Ecal ", etBin, etMin, etMax, 20, 0., 20);
  //
  histname = "phoE";
  h_phoE_[0][0] = iBooker.book1D(histname + "All", " Photon Energy: All ecal ", eBin, eMin, eMax);
  h_phoE_[0][1] = iBooker.book1D(histname + "Barrel", " Photon Energy: barrel ", eBin, eMin, eMax);
  h_phoE_[0][2] = iBooker.book1D(histname + "Endcap", " Photon Energy: Endcap ", eBin, eMin, eMax);

  histname = "phoEt";
  h_phoEt_[0][0] = iBooker.book1D(histname + "All", " Photon Transverse Energy: All ecal ", etBin, etMin, etMax);
  h_phoEt_[0][1] = iBooker.book1D(histname + "Barrel", " Photon Transverse Energy: Barrel ", etBin, etMin, etMax);
  h_phoEt_[0][2] = iBooker.book1D(histname + "Endcap", " Photon Transverse Energy: Endcap ", etBin, etMin, etMax);

  histname = "eRes";
  h_phoERes_[0][0] =
      iBooker.book1D(histname + "All", " Photon E/E_{true}: All ecal;  E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[0][1] =
      iBooker.book1D(histname + "Barrel", "Photon E/E_{true}: Barrel; E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[0][2] =
      iBooker.book1D(histname + "Endcap", " Photon E/E_{true}: Endcap; E/E_{true} (GeV)", resBin, resMin, resMax);

  h_phoERes_[1][0] = iBooker.book1D(
      histname + "unconvAll", " Photon E/E_{true} if r9>0.94, 0.95: All ecal; E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[1][1] = iBooker.book1D(
      histname + "unconvBarrel", " Photon E/E_{true} if r9>0.94: Barrel; E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[1][2] = iBooker.book1D(
      histname + "unconvEndcap", " Photon E/E_{true} if r9>0.95: Endcap; E/E_{true} (GeV)", resBin, resMin, resMax);

  h_phoERes_[2][0] = iBooker.book1D(
      histname + "convAll", " Photon E/E_{true} if r9<0.0.94, 0.95: All ecal; E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[2][1] = iBooker.book1D(
      histname + "convBarrel", " Photon E/E_{true} if r9<0.94: Barrel; E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[2][2] = iBooker.book1D(
      histname + "convEndcap", " Photon E/E_{true} if r9<0.95: Endcap; E/E_{true} (GeV)", resBin, resMin, resMax);

  histname = "sigmaEoE";
  h_phoSigmaEoE_[0][0] = iBooker.book1D(histname + "All", "#sigma_{E}/E: All ecal; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[0][1] = iBooker.book1D(histname + "Barrel", "#sigma_{E}/E: Barrel; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[0][2] = iBooker.book1D(histname + "Endcap", "#sigma_{E}/E: Endcap, #sigma_{E}/E", 100, 0., 0.08);

  h_phoSigmaEoE_[1][0] =
      iBooker.book1D(histname + "unconvAll", "#sigma_{E}/E if r9>0.94, 0.95: All ecal; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[1][1] =
      iBooker.book1D(histname + "unconvBarrel", "#sigma_{E}/E if r9>0.94: Barrel; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[1][2] =
      iBooker.book1D(histname + "unconvEndcap", "#sigma_{E}/E r9>0.95: Endcap; #sigma_{E}/E", 100, 0., 0.08);

  h_phoSigmaEoE_[2][0] =
      iBooker.book1D(histname + "convAll", "#sigma_{E}/E if r9<0.0.94, 0.95: All ecal, #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[2][1] =
      iBooker.book1D(histname + "convBarrel", "#sigma_{E}/E if r9<0.94: Barrel, #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[2][2] =
      iBooker.book1D(histname + "convEndcap", "#sigma_{E}/E if r9<0.95: Endcap, #sigma_{E}/E", 100, 0., 0.08);

  histname = "eResVsEta";
  if (!isRunCentrally_)
    h2_eResVsEta_[0] = iBooker.book2D(
        histname + "All", " All photons E/Etrue vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 2.5);
  if (!isRunCentrally_)
    h2_eResVsEta_[1] = iBooker.book2D(
        histname + "Unconv", " Unconv photons E/Etrue vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 2.5);

  histname = "pEResVsEta";
  p_eResVsEta_[0] = iBooker.bookProfile(
      histname + "All", "All photons  E/Etrue vs #eta: all Ecal ", etaBin2, etaMin, etaMax, resBin, resMin, resMax, "");
  p_eResVsEta_[1] = iBooker.bookProfile(histname + "Unconv",
                                        "Unconv photons  E/Etrue vs #eta: all Ecal",
                                        etaBin2,
                                        etaMin,
                                        etaMax,
                                        resBin,
                                        resMin,
                                        resMax,
                                        "");
  p_eResVsEta_[2] = iBooker.bookProfile(
      histname + "Conv", "Conv photons  E/Etrue vs #eta: all Ecal", etaBin2, etaMin, etaMax, resBin, resMin, resMax, "");

  histname = "pSigmaEoEVsEta";
  p_sigmaEoEVsEta_[0] = iBooker.bookProfile(histname + "All",
                                            "All photons: #sigma_{E}/E vs #eta: all Ecal; #eta; #sigma_{E}/E",
                                            etaBin2,
                                            etaMin,
                                            etaMax,
                                            100,
                                            0.,
                                            0.08,
                                            "");
  p_sigmaEoEVsEta_[1] = iBooker.bookProfile(histname + "Unconv",
                                            "Unconv photons #sigma_{E}/E vs #eta: all Ecal; #eta; #sigma_{E}/E ",
                                            etaBin2,
                                            etaMin,
                                            etaMax,
                                            100,
                                            0.,
                                            0.08,
                                            "");
  p_sigmaEoEVsEta_[2] = iBooker.bookProfile(histname + "Conv",
                                            "Conv photons  #sigma_{E}/E vs #eta: all Ecal;  #eta; #sigma_{E}/E",
                                            etaBin2,
                                            etaMin,
                                            etaMax,
                                            100,
                                            0.,
                                            0.08,
                                            "");

  histname = "pSigmaEoEVsEt";
  p_sigmaEoEVsEt_[1][0] = iBooker.bookProfile(histname + "Barrel",
                                              "All photons #sigma_{E}/E vs E_{T}: Barrel;  E_{T} (GeV); #sigma_{E}/E ",
                                              etBin,
                                              etMin,
                                              etMax,
                                              100,
                                              0.,
                                              0.08,
                                              "");
  p_sigmaEoEVsEt_[1][1] =
      iBooker.bookProfile(histname + "unconvBarrel",
                          "Unconv photons #sigma_{E}/E vs E_{T}: Barrel;  E_{T} (GeV); #sigma_{E}/E ",
                          etBin,
                          etMin,
                          etMax,
                          100,
                          0.,
                          0.08,
                          "");
  p_sigmaEoEVsEt_[1][2] = iBooker.bookProfile(histname + "convBarrel",
                                              "Conv photons  #sigma_{E}/E vs E_{T}: Barrel;  E_{T} (GeV); #sigma_{E}/E",
                                              etBin,
                                              etMin,
                                              etMax,
                                              100,
                                              0.,
                                              0.08,
                                              "");
  p_sigmaEoEVsEt_[2][0] = iBooker.bookProfile(histname + "Endcap",
                                              "All photons #sigma_{E}/E vs E_{T}: Endcap;  E_{T} (GeV); #sigma_{E}/E ",
                                              etBin,
                                              etMin,
                                              etMax,
                                              100,
                                              0.,
                                              0.08,
                                              "");
  p_sigmaEoEVsEt_[2][1] =
      iBooker.bookProfile(histname + "unconvEndcap",
                          "Unconv photons #sigma_{E}/E vs E_{T}: Endcap;  E_{T} (GeV); #sigma_{E}/E ",
                          etBin,
                          etMin,
                          etMax,
                          100,
                          0.,
                          0.08,
                          "");
  p_sigmaEoEVsEt_[2][2] = iBooker.bookProfile(histname + "convEndcap",
                                              "Conv photons  #sigma_{E}/E vs E_{T}: Endcap;  E_{T} (GeV); #sigma_{E}/E",
                                              etBin,
                                              etMin,
                                              etMax,
                                              100,
                                              0.,
                                              0.08,
                                              "");

  histname = "pSigmaEoEVsNVtx";
  p_sigmaEoEVsNVtx_[1][0] = iBooker.bookProfile(histname + "Barrel",
                                                "All photons: #sigma_{E}/E vs N_{vtx}: Barrel; N_{vtx}; #sigma_{E}/E",
                                                200,
                                                -0.5,
                                                199.5,
                                                100,
                                                0.,
                                                0.08,
                                                "");
  p_sigmaEoEVsNVtx_[1][1] =
      iBooker.bookProfile(histname + "unconvBarrel",
                          "Unconv photons #sigma_{E}/E vs N_{vtx}: Barrel; N_{vtx}; #sigma_{E}/E ",
                          200,
                          -0.5,
                          199.5,
                          100,
                          0.,
                          0.08,
                          "");
  p_sigmaEoEVsNVtx_[1][2] = iBooker.bookProfile(histname + "convBarrel",
                                                "Conv photons  #sigma_{E}/E vs N_{vtx}: Barrel;  N_{vtx}; #sigma_{E}/E",
                                                200,
                                                -0.5,
                                                199.5,
                                                100,
                                                0.,
                                                0.08,
                                                "");
  p_sigmaEoEVsNVtx_[2][0] = iBooker.bookProfile(histname + "Endcap",
                                                "All photons: #sigma_{E}/E vs N_{vtx}: Endcap; N_{vtx}; #sigma_{E}/E",
                                                200,
                                                -0.5,
                                                199.5,
                                                100,
                                                0.,
                                                0.08,
                                                "");
  p_sigmaEoEVsNVtx_[2][1] =
      iBooker.bookProfile(histname + "unconvEndcap",
                          "Unconv photons #sigma_{E}/E vs N_{vtx}: Endcap; N_{vtx}; #sigma_{E}/E ",
                          200,
                          -0.5,
                          199.5,
                          100,
                          0.,
                          0.08,
                          "");
  p_sigmaEoEVsNVtx_[2][2] = iBooker.bookProfile(histname + "convEndcap",
                                                "Conv photons  #sigma_{E}/E vs N_{vtx}: Endcap;  N_{vtx}; #sigma_{E}/E",
                                                200,
                                                -0.5,
                                                199.5,
                                                100,
                                                0.,
                                                0.08,
                                                "");

  if (!isRunCentrally_) {
    histname = "eResVsEt";
    h2_eResVsEt_[0][0] = iBooker.book2D(
        histname + "All", " All photons E/Etrue vs true Et: all Ecal ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[0][1] = iBooker.book2D(
        histname + "unconv", " All photons E/Etrue vs true Et: all Ecal ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[0][2] = iBooker.book2D(
        histname + "conv", " All photons E/Etrue vs true Et: all Ecal ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[1][0] = iBooker.book2D(
        histname + "Barrel", " All photons E/Etrue vs true Et: Barrel ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[1][1] = iBooker.book2D(
        histname + "unconvBarrel", " All photons E/Etrue vs true Et: Barrel ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[1][2] = iBooker.book2D(
        histname + "convBarrel", " All photons E/Etrue vs true Et: Barrel ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[2][0] = iBooker.book2D(
        histname + "Endcap", " All photons E/Etrue vs true Et: Endcap ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[2][1] = iBooker.book2D(
        histname + "unconvEndcap", " All photons E/Etrue vs true Et: Endcap ", etBin, etMin, etMax, 100, 0.9, 1.1);
    h2_eResVsEt_[2][2] = iBooker.book2D(
        histname + "convEndcap", " All photons E/Etrue vs true Et: Endcap ", etBin, etMin, etMax, 100, 0.9, 1.1);
  }

  histname = "pEResVsEt";
  p_eResVsEt_[0][0] = iBooker.bookProfile(
      histname + "All", "All photons  E/Etrue vs Et: all Ecal ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[0][1] = iBooker.bookProfile(
      histname + "unconv", "All photons  E/Etrue vs Et: all Ecal ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[0][2] = iBooker.bookProfile(
      histname + "conv", "All photons  E/Etrue vs Et: all Ecal ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[1][0] = iBooker.bookProfile(
      histname + "Barrel", "All photons  E/Etrue vs Et: Barrel ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[1][1] = iBooker.bookProfile(
      histname + "unconvBarrel", "All photons  E/Etrue vs Et: Barrel ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[1][2] = iBooker.bookProfile(
      histname + "convBarrel", "All photons  E/Etrue vs Et: Barrel ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[2][0] = iBooker.bookProfile(
      histname + "Endcap", "All photons  E/Etrue vs Et: Endcap ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[2][1] = iBooker.bookProfile(
      histname + "unconvEndcap", "All photons  E/Etrue vs Et: Endcap ", etBin, etMin, etMax, resBin, resMin, resMax, "");
  p_eResVsEt_[2][2] = iBooker.bookProfile(
      histname + "convEndcap", "All photons  E/Etrue vs Et: Endcap ", etBin, etMin, etMax, resBin, resMin, resMax, "");

  histname = "pEResVsNVtx";
  p_eResVsNVtx_[1][0] = iBooker.bookProfile(histname + "Barrel",
                                            "All photons  E/E_{true}  vs N_{vtx}: Barrel;  N_{vtx}; E}/E_{true}",
                                            200,
                                            -0.5,
                                            199.5,
                                            resBin,
                                            resMin,
                                            resMax,
                                            "");
  p_eResVsNVtx_[1][1] =
      iBooker.bookProfile(histname + "unconvBarrel",
                          "Unconverted photons E/E_{true}  vs N_{vtx}: Barrel;  N_{vtx}; E}/E_{true} ",
                          200,
                          -0.5,
                          199.5,
                          resBin,
                          resMin,
                          resMax,
                          "");
  p_eResVsNVtx_[1][2] =
      iBooker.bookProfile(histname + "convBarrel",
                          " Converted photons  E/E_{true}  vs N_{vtx}: Barrel;  N_{vtx}; E}/E_{true} ",
                          200,
                          -0.5,
                          199.5,
                          resBin,
                          resMin,
                          resMax,
                          "");
  p_eResVsNVtx_[2][0] = iBooker.bookProfile(histname + "Endcap",
                                            "All photons  E/E_{true}  vs N_{vtx}: Endcap;  N_{vtx}; E}/E_{true} ",
                                            200,
                                            -0.5,
                                            199.5,
                                            resBin,
                                            resMin,
                                            resMax,
                                            "");
  p_eResVsNVtx_[2][1] =
      iBooker.bookProfile(histname + "unconvEndcap",
                          "Uncoverted photons  E/E_{true}  vs N_{vtx}: Endcap;  N_{vtx}; E}/E_{true} ",
                          2080,
                          -0.5,
                          199.5,
                          resBin,
                          resMin,
                          resMax,
                          "");
  p_eResVsNVtx_[2][2] = iBooker.bookProfile(histname + "convEndcap",
                                            "Converted photons E/E_{true}  vs N_{vtx}: Endcap;  N_{vtx}; E}/E_{true} ",
                                            200,
                                            -0.5,
                                            199.5,
                                            resBin,
                                            resMin,
                                            resMax,
                                            "");

  histname = "eResVsR9";
  if (!isRunCentrally_)
    h2_eResVsR9_[0] = iBooker.book2D(
        histname + "All", " All photons E/Etrue vs R9: all Ecal ", r9Bin * 2, r9Min, r9Max, 100, 0., 2.5);
  if (!isRunCentrally_)
    h2_eResVsR9_[1] = iBooker.book2D(
        histname + "Barrel", " All photons E/Etrue vs R9: Barrel ", r9Bin * 2, r9Min, r9Max, 100, 0., 2.5);
  if (!isRunCentrally_)
    h2_eResVsR9_[2] = iBooker.book2D(
        histname + "Endcap", " All photons E/Etrue vs R9: Endcap ", r9Bin * 2, r9Min, r9Max, 100, 0., 2.5);
  histname = "pEResVsR9";
  if (!isRunCentrally_)
    p_eResVsR9_[0] = iBooker.bookProfile(
        histname + "All", " All photons  E/Etrue vs R9: all Ecal ", r9Bin * 2, r9Min, r9Max, resBin, resMin, resMax, "");
  p_eResVsR9_[1] = iBooker.bookProfile(
      histname + "Barrel", " All photons  E/Etrue vs R9: Barrel ", r9Bin * 2, r9Min, r9Max, resBin, resMin, resMax, "");
  p_eResVsR9_[2] = iBooker.bookProfile(
      histname + "Endcap", " All photons  E/Etrue vs R9: Endcap ", r9Bin * 2, r9Min, r9Max, resBin, resMin, resMax, "");
  histname = "sceResVsR9";
  if (!isRunCentrally_)
    h2_sceResVsR9_[0] = iBooker.book2D(
        histname + "All", " All photons scE/Etrue vs R9: all Ecal ", r9Bin * 2, r9Min, r9Max, 100, 0., 2.5);
  if (!isRunCentrally_)
    h2_sceResVsR9_[1] = iBooker.book2D(
        histname + "Barrel", " All photons scE/Etrue vs R9: Barrel ", r9Bin * 2, r9Min, r9Max, 100, 0., 2.5);
  if (!isRunCentrally_)
    h2_sceResVsR9_[2] = iBooker.book2D(
        histname + "Endcap", " All photons scE/Etrue vs R9: Endcap ", r9Bin * 2, r9Min, r9Max, 100, 0., 2.5);
  histname = "scpEResVsR9";
  if (!isRunCentrally_)
    p_sceResVsR9_[0] = iBooker.bookProfile(histname + "All",
                                           " All photons  scE/Etrue vs R9: all Ecal ",
                                           r9Bin * 2,
                                           r9Min,
                                           r9Max,
                                           resBin,
                                           resMin,
                                           resMax,
                                           "");
  p_sceResVsR9_[1] = iBooker.bookProfile(histname + "Barrel",
                                         " All photons  scE/Etrue vs R9: Barrel ",
                                         r9Bin * 2,
                                         r9Min,
                                         r9Max,
                                         resBin,
                                         resMin,
                                         resMax,
                                         "");
  p_sceResVsR9_[2] = iBooker.bookProfile(histname + "Endcap",
                                         " All photons  scE/Etrue vs R9: Endcap ",
                                         r9Bin * 2,
                                         r9Min,
                                         r9Max,
                                         resBin,
                                         resMin,
                                         resMax,
                                         "");

  // Photon E resolution when using energy values from regressions
  histname = "eResRegr1";
  h_phoEResRegr1_[0][0] =
      iBooker.book1D(histname + "All", " Photon rec/true Energy from Regression1 : All ecal ", resBin, resMin, resMax);
  h_phoEResRegr1_[0][1] =
      iBooker.book1D(histname + "Barrel", " Photon rec/true Energy from Regression1: Barrel ", resBin, resMin, resMax);
  h_phoEResRegr1_[0][2] =
      iBooker.book1D(histname + "Endcap", " Photon rec/true Energy from Regression1: Endcap ", resBin, resMin, resMax);

  h_phoEResRegr1_[1][0] = iBooker.book1D(histname + "unconvAll",
                                         " Photon rec/true Energy from Regression1 if r9>0.94, 0.95: All ecal ",
                                         resBin,
                                         resMin,
                                         resMax);
  h_phoEResRegr1_[1][1] = iBooker.book1D(
      histname + "unconvBarrel", " Photon rec/true Energy from Regression1 if r9>0.94: Barrel ", resBin, resMin, resMax);
  h_phoEResRegr1_[1][2] = iBooker.book1D(
      histname + "unconvEndcap", " Photon rec/true Energy from Regression1 if r9>0.95: Endcap ", resBin, resMin, resMax);

  h_phoEResRegr1_[2][0] = iBooker.book1D(histname + "convAll",
                                         " Photon rec/true Energy  from Regression1if r9<0.0.94, 0.95: All ecal ",
                                         resBin,
                                         resMin,
                                         resMax);
  h_phoEResRegr1_[2][1] = iBooker.book1D(
      histname + "convBarrel", " Photon rec/true Energy from Regression1 if r9<0.94: Barrel ", resBin, resMin, resMax);
  h_phoEResRegr1_[2][2] = iBooker.book1D(
      histname + "convEndcap", " Photon rec/true Energy from Regression1 if r9<0.95: Endcap ", resBin, resMin, resMax);

  histname = "eResRegr2";
  h_phoEResRegr2_[0][0] =
      iBooker.book1D(histname + "All", " Photon rec/true Energy from Regression2 : All ecal ", resBin, resMin, resMax);
  h_phoEResRegr2_[0][1] =
      iBooker.book1D(histname + "Barrel", " Photon rec/true Energy from Regression2: Barrel ", resBin, resMin, resMax);
  h_phoEResRegr2_[0][2] =
      iBooker.book1D(histname + "Endcap", " Photon rec/true Energy from Regression2: Endcap ", resBin, resMin, resMax);

  h_phoEResRegr2_[1][0] = iBooker.book1D(histname + "unconvAll",
                                         " Photon rec/true Energy from Regression2 if r9>0.94, 0.95: All ecal ",
                                         resBin,
                                         resMin,
                                         resMax);
  h_phoEResRegr2_[1][1] = iBooker.book1D(
      histname + "unconvBarrel", " Photon rec/true Energy from Regression2 if r9>0.94: Barrel ", resBin, resMin, resMax);
  h_phoEResRegr2_[1][2] = iBooker.book1D(
      histname + "unconvEndcap", " Photon rec/true Energy from Regression2 if r9>0.95: Endcap ", resBin, resMin, resMax);

  h_phoEResRegr2_[2][0] = iBooker.book1D(histname + "convAll",
                                         " Photon rec/true Energy  from Regression2 if r9<0.0.94, 0.95: All ecal ",
                                         resBin,
                                         resMin,
                                         resMax);
  h_phoEResRegr2_[2][1] = iBooker.book1D(
      histname + "convBarrel", " Photon rec/true Energy from Regression2 if r9<0.94: Barrel ", resBin, resMin, resMax);
  h_phoEResRegr2_[2][2] = iBooker.book1D(
      histname + "convEndcap", " Photon rec/true Energy from Regression2 if r9<0.95: Endcap ", resBin, resMin, resMax);
  //
  histname = "phoPixSeedSize";
  h_phoPixSeedSize_[0] = iBooker.book1D(histname + "Barrel", "Pixel seeds size ", 100, 0., 100.);
  h_phoPixSeedSize_[1] = iBooker.book1D(histname + "Endcap", "Pixel seeds size ", 100, 0., 100.);

  //  Infos from Particle Flow - isolation and ID
  histname = "chargedHadIso";
  h_chHadIso_[0] = iBooker.book1D(histname + "All", "PF chargedHadIso:  All Ecal", etBin, etMin, 20.);
  h_chHadIso_[1] = iBooker.book1D(histname + "Barrel", "PF chargedHadIso:  Barrel", etBin, etMin, 20.);
  h_chHadIso_[2] = iBooker.book1D(histname + "Endcap", "PF chargedHadIso:  Endcap", etBin, etMin, 20.);
  histname = "neutralHadIso";
  h_nHadIso_[0] = iBooker.book1D(histname + "All", "PF neutralHadIso:  All Ecal", etBin, etMin, 20.);
  h_nHadIso_[1] = iBooker.book1D(histname + "Barrel", "PF neutralHadIso:  Barrel", etBin, etMin, 20.);
  h_nHadIso_[2] = iBooker.book1D(histname + "Endcap", "PF neutralHadIso:  Endcap", etBin, etMin, 20.);
  histname = "photonIso";
  h_phoIso_[0] = iBooker.book1D(histname + "All", "PF photonIso:  All Ecal", etBin, etMin, 20.);
  h_phoIso_[1] = iBooker.book1D(histname + "Barrel", "PF photonIso:  Barrel", etBin, etMin, 20.);
  h_phoIso_[2] = iBooker.book1D(histname + "Endcap", "PF photonIso:  Endcap", etBin, etMin, 20.);
  histname = "nCluOutMustache";
  h_nCluOutsideMustache_[0] =
      iBooker.book1D(histname + "All", "PF number of clusters outside Mustache:  All Ecal", 50, 0., 50.);
  h_nCluOutsideMustache_[1] =
      iBooker.book1D(histname + "Barrel", "PF number of clusters outside Mustache:  Barrel", 50, 0., 50.);
  h_nCluOutsideMustache_[2] =
      iBooker.book1D(histname + "Endcap", "PF number of clusters outside Mustache:  Endcap", 50, 0., 50.);
  histname = "etOutMustache";
  h_etOutsideMustache_[0] = iBooker.book1D(histname + "All", "PF et outside Mustache:  All Ecal", etBin, etMin, 20.);
  h_etOutsideMustache_[1] = iBooker.book1D(histname + "Barrel", "PF et outside Mustache:  Barrel", etBin, etMin, 20.);
  h_etOutsideMustache_[2] = iBooker.book1D(histname + "Endcap", "PF et outside Mustache:  Endcap", etBin, etMin, 20.);
  histname = "pfMVA";
  h_pfMva_[0] = iBooker.book1D(histname + "All", "PF MVA output:  All Ecal", 50, -1., 2.);
  h_pfMva_[1] = iBooker.book1D(histname + "Barrel", "PF MVA output:  Barrel", 50, -1., 2.);
  h_pfMva_[2] = iBooker.book1D(histname + "Endcap", "PF MVA output:  Endcap", 50, -1, 2.);
  ////////// particle based isolation from value map
  histname = "SumPtOverPhoPt_ChHad_Cleaned";
  h_SumPtOverPhoPt_ChHad_Cleaned_[0] =
      iBooker.book1D(histname + "All", "Pf Cand SumPt/P_{T}_{#gamma}: Charged Hadrons:  All Ecal", etBin, etMin, 2.);
  h_SumPtOverPhoPt_ChHad_Cleaned_[1] =
      iBooker.book1D(histname + "Barrel", "PF Cand SumPt/P_{T}_{#gamma}: Charged Hadrons:  Barrel", etBin, etMin, 2.);
  h_SumPtOverPhoPt_ChHad_Cleaned_[2] =
      iBooker.book1D(histname + "Endcap", "PF Cand SumPt/P_{T}_{#gamma}: Charged Hadrons:  Endcap", etBin, etMin, 2.);
  histname = "SumPtOverPhoPt_NeuHad_Cleaned";
  h_SumPtOverPhoPt_NeuHad_Cleaned_[0] =
      iBooker.book1D(histname + "All", "Pf Cand  SumPt/P_{T}_{#gamma}: Neutral Hadrons:  All Ecal", etBin, etMin, 2.);
  h_SumPtOverPhoPt_NeuHad_Cleaned_[1] =
      iBooker.book1D(histname + "Barrel", "PF Cand  SumPt/P_{T}_{#gamma}: Neutral Hadrons:  Barrel", etBin, etMin, 2.);
  h_SumPtOverPhoPt_NeuHad_Cleaned_[2] =
      iBooker.book1D(histname + "Endcap", "PF Cand  SumPt/P_{T}_{#gamma}: Neutral Hadrons:  Endcap", etBin, etMin, 2.);
  histname = "SumPtOverPhoPt_Pho_Cleaned";
  h_SumPtOverPhoPt_Pho_Cleaned_[0] =
      iBooker.book1D(histname + "All", "Pf Cand SumPt/P_{T}_{#gamma}: Photons:  All Ecal", etBin, etMin, 2.);
  h_SumPtOverPhoPt_Pho_Cleaned_[1] =
      iBooker.book1D(histname + "Barrel", "PF Cand SumPt/P_{T}_{#gamma}: Photons:  Barrel", etBin, etMin, 2.);
  h_SumPtOverPhoPt_Pho_Cleaned_[2] =
      iBooker.book1D(histname + "Endcap", "PF Cand SumPt/P_{T}_{#gamma}: Photons:  Endcap", etBin, etMin, 2.);

  histname = "dRPhoPFcand_ChHad_Cleaned";
  h_dRPhoPFcand_ChHad_Cleaned_[0] =
      iBooker.book1D(histname + "All", "dR(pho,cand) Charged Hadrons : All Ecal", etBin, etMin, 0.7);
  h_dRPhoPFcand_ChHad_Cleaned_[1] =
      iBooker.book1D(histname + "Barrel", "dR(pho,cand) Charged Hadrons :  Barrel", etBin, etMin, 0.7);
  h_dRPhoPFcand_ChHad_Cleaned_[2] =
      iBooker.book1D(histname + "Endcap", "dR(pho,cand) Charged Hadrons :  Endcap", etBin, etMin, 0.7);
  histname = "dRPhoPFcand_NeuHad_Cleaned";
  h_dRPhoPFcand_NeuHad_Cleaned_[0] =
      iBooker.book1D(histname + "All", "dR(pho,cand) Neutral Hadrons : All Ecal", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_Cleaned_[1] =
      iBooker.book1D(histname + "Barrel", "dR(pho,cand) Neutral Hadrons :  Barrel", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_Cleaned_[2] =
      iBooker.book1D(histname + "Endcap", "dR(pho,cand) Neutral Hadrons :  Endcap", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_Cleaned_[3] =
      iBooker.book1D(histname + "Barrel_1", "dR(pho,cand) Neutral Hadrons :  Barrel |eta| <=1", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_Cleaned_[4] =
      iBooker.book1D(histname + "Barrel_2", "dR(pho,cand) Neutral Hadrons :  Barrel |eta | > 1", etBin, etMin, 0.7);
  histname = "dRPhoPFcand_Pho_Cleaned";
  h_dRPhoPFcand_Pho_Cleaned_[0] =
      iBooker.book1D(histname + "All", "dR(pho,cand) Photons : All Ecal", etBin, etMin, 0.7);
  h_dRPhoPFcand_Pho_Cleaned_[1] =
      iBooker.book1D(histname + "Barrel", "dR(pho,cand) Photons :  Barrel", etBin, etMin, 0.7);
  h_dRPhoPFcand_Pho_Cleaned_[2] =
      iBooker.book1D(histname + "Endcap", "dR(pho,cand) Photons :  Endcap", etBin, etMin, 0.7);

  //
  histname = "SumPtOverPhoPt_ChHad_unCleaned";
  h_SumPtOverPhoPt_ChHad_unCleaned_[0] =
      iBooker.book1D(histname + "All", "Pf Cand Sum Pt Over photon pt Charged Hadrons :  All Ecal", etBin, etMin, 2.);
  h_SumPtOverPhoPt_ChHad_unCleaned_[1] =
      iBooker.book1D(histname + "Barrel", "PF Cand Sum Pt Over photon pt Charged Hadrons:  Barrel", etBin, etMin, 2.);
  h_SumPtOverPhoPt_ChHad_unCleaned_[2] =
      iBooker.book1D(histname + "Endcap", "PF Cand Sum Pt Over photon pt Charged Hadrons:  Endcap", etBin, etMin, 2.);
  histname = "SumPtOverPhoPt_NeuHad_unCleaned";
  h_SumPtOverPhoPt_NeuHad_unCleaned_[0] =
      iBooker.book1D(histname + "All", "Pf Cand Sum Pt Over photon pt Neutral Hadrons :  All Ecal", etBin, etMin, 2.);
  h_SumPtOverPhoPt_NeuHad_unCleaned_[1] =
      iBooker.book1D(histname + "Barrel", "PF Cand Sum Pt Over photon pt Neutral Hadrons:  Barrel", etBin, etMin, 2.);
  h_SumPtOverPhoPt_NeuHad_unCleaned_[2] =
      iBooker.book1D(histname + "Endcap", "PF Cand Sum Pt Over photon pt Neutral Hadrons:  Endcap", etBin, etMin, 2.);
  histname = "SumPtOverPhoPt_Pho_unCleaned";
  h_SumPtOverPhoPt_Pho_unCleaned_[0] =
      iBooker.book1D(histname + "All", "Pf Cand Sum Pt Over photon pt Photons:  All Ecal", etBin, etMin, 2.);
  h_SumPtOverPhoPt_Pho_unCleaned_[1] =
      iBooker.book1D(histname + "Barrel", "PF Cand Sum Pt Over photon pt Photons:  Barrel", etBin, etMin, 2.);
  h_SumPtOverPhoPt_Pho_unCleaned_[2] =
      iBooker.book1D(histname + "Endcap", "PF Cand Sum Pt Over photon pt Photons:  Endcap", etBin, etMin, 2.);
  histname = "dRPhoPFcand_ChHad_unCleaned";
  h_dRPhoPFcand_ChHad_unCleaned_[0] =
      iBooker.book1D(histname + "All", "dR(pho,cand) Charged Hadrons :  All Ecal", etBin, etMin, 0.7);
  h_dRPhoPFcand_ChHad_unCleaned_[1] =
      iBooker.book1D(histname + "Barrel", "dR(pho,cand) Charged Hadrons :  Barrel", etBin, etMin, 0.7);
  h_dRPhoPFcand_ChHad_unCleaned_[2] =
      iBooker.book1D(histname + "Endcap", "dR(pho,cand) Charged Hadrons :  Endcap", etBin, etMin, 0.7);

  histname = "dRPhoPFcand_NeuHad_unCleaned";
  h_dRPhoPFcand_NeuHad_unCleaned_[0] =
      iBooker.book1D(histname + "All", "dR(pho,cand) Neutral Hadrons :  All Ecal", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_unCleaned_[1] =
      iBooker.book1D(histname + "Barrel", "dR(pho,cand) Neutral Hadrons :  Barrel", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_unCleaned_[2] =
      iBooker.book1D(histname + "Endcap", "dR(pho,cand) Neutral Hadrons :  Endcap", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_unCleaned_[3] =
      iBooker.book1D(histname + "Barrel_1", "dR(pho,cand) Neutral Hadrons :  Barrel |eta| <=1  ", etBin, etMin, 0.7);
  h_dRPhoPFcand_NeuHad_unCleaned_[4] =
      iBooker.book1D(histname + "Barrel_2", "dR(pho,cand) Neutral Hadrons :  Barrel |eta| > 1", etBin, etMin, 0.7);

  histname = "dRPhoPFcand_Pho_unCleaned";
  h_dRPhoPFcand_Pho_unCleaned_[0] =
      iBooker.book1D(histname + "All", "dR(pho,cand) Photons:  All Ecal", etBin, etMin, 0.7);
  h_dRPhoPFcand_Pho_unCleaned_[1] =
      iBooker.book1D(histname + "Barrel", "dR(pho,cand) Photons:  Barrel", etBin, etMin, 0.7);
  h_dRPhoPFcand_Pho_unCleaned_[2] =
      iBooker.book1D(histname + "Endcap", "dR(pho,cand) Photons:  Endcap", etBin, etMin, 0.7);

  //    if ( ! isRunCentrally_ ) {
  // Photon pair invariant mass
  histname = "gamgamMass";
  h_gamgamMass_[0][0] =
      iBooker.book1D(histname + "All", "2 photons invariant mass: All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMass_[0][1] =
      iBooker.book1D(histname + "Barrel", "2 photons invariant mass:  Barrel ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMass_[0][2] =
      iBooker.book1D(histname + "Endcap", "2 photons invariant mass:  Endcap ", ggMassBin, ggMassMin, ggMassMax);
  //
  histname = "gamgamMassNoConv";
  h_gamgamMass_[1][0] = iBooker.book1D(
      histname + "All", "2 photons with no conversion invariant mass: All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMass_[1][1] = iBooker.book1D(
      histname + "Barrel", "2 photons with no conversion  invariant mass:  Barrel ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMass_[1][2] = iBooker.book1D(
      histname + "Endcap", "2 photons with no conversion  invariant mass:  Endcap ", ggMassBin, ggMassMin, ggMassMax);
  //
  histname = "gamgamMassConv";
  h_gamgamMass_[2][0] = iBooker.book1D(
      histname + "All", "2 photons with conversion invariant mass: All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMass_[2][1] = iBooker.book1D(
      histname + "Barrel", "2 photons with  conversion  invariant mass:  Barrel ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMass_[2][2] = iBooker.book1D(
      histname + "Endcap", "2 photons with  conversion  invariant mass:  Endcap ", ggMassBin, ggMassMin, ggMassMax);
  // with energy regression1
  histname = "gamgamMassRegr1";
  h_gamgamMassRegr1_[0][0] =
      iBooker.book1D(histname + "All", "2 photons invariant mass Regr1 : All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr1_[0][1] =
      iBooker.book1D(histname + "Barrel", "2 photons invariant mass Regr1 :  Barrel ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr1_[0][2] =
      iBooker.book1D(histname + "Endcap", "2 photons invariant mass Regr1 :  Endcap ", ggMassBin, ggMassMin, ggMassMax);
  //
  histname = "gamgamMassRegr1NoConv";
  h_gamgamMassRegr1_[1][0] = iBooker.book1D(
      histname + "All", "2 photons with no conversion invariant mass Regr1: All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr1_[1][1] = iBooker.book1D(histname + "Barrel",
                                            "2 photons with no conversion  invariant mass Regr1:  Barrel ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);
  h_gamgamMassRegr1_[1][2] = iBooker.book1D(histname + "Endcap",
                                            "2 photons with no conversion  invariant mass Regr1:  Endcap ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);
  //
  histname = "gamgamMassRegr1Conv";
  h_gamgamMassRegr1_[2][0] = iBooker.book1D(
      histname + "All", "2 photons with conversion invariant mass Regr1: All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr1_[2][1] = iBooker.book1D(histname + "Barrel",
                                            "2 photons with  conversion  invariant mass Regr1:  Barrel ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);
  h_gamgamMassRegr1_[2][2] = iBooker.book1D(histname + "Endcap",
                                            "2 photons with  conversion  invariant mass Regr1:  Endcap ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);
  // with energy regression2
  histname = "gamgamMassRegr2";
  h_gamgamMassRegr2_[0][0] =
      iBooker.book1D(histname + "All", "2 photons invariant mass Regr2 : All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr2_[0][1] =
      iBooker.book1D(histname + "Barrel", "2 photons invariant mass Regr2 :  Barrel ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr2_[0][2] =
      iBooker.book1D(histname + "Endcap", "2 photons invariant mass Regr2 :  Endcap ", ggMassBin, ggMassMin, ggMassMax);
  //
  histname = "gamgamMassRegr2NoConv";
  h_gamgamMassRegr2_[1][0] = iBooker.book1D(
      histname + "All", "2 photons with no conversion invariant mass Regr2: All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr2_[1][1] = iBooker.book1D(histname + "Barrel",
                                            "2 photons with no conversion  invariant mass Regr2:  Barrel ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);
  h_gamgamMassRegr2_[1][2] = iBooker.book1D(histname + "Endcap",
                                            "2 photons with no conversion  invariant mass Regr2:  Endcap ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);
  //
  histname = "gamgamMassRegr2Conv";
  h_gamgamMassRegr2_[2][0] = iBooker.book1D(
      histname + "All", "2 photons with conversion invariant mass Regr2: All ecal ", ggMassBin, ggMassMin, ggMassMax);
  h_gamgamMassRegr2_[2][1] = iBooker.book1D(histname + "Barrel",
                                            "2 photons with  conversion  invariant mass Regr2:  Barrel ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);
  h_gamgamMassRegr2_[2][2] = iBooker.book1D(histname + "Endcap",
                                            "2 photons with  conversion  invariant mass Regr2:  Endcap ",
                                            ggMassBin,
                                            ggMassMin,
                                            ggMassMax);

  //}

  ///// Histos to allow comparison with miniAOD

  h_scEta_miniAOD_[0] = iBooker.book1D("scEta_miniAOD", " SC Eta ", etaBin, etaMin, etaMax);
  h_scPhi_miniAOD_[0] = iBooker.book1D("scPhi_miniAOD", " SC Phi ", phiBin, phiMin, phiMax);
  histname = "phoE";
  h_phoE_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", " Photon Energy: All ecal ", eBin, eMin, eMax);
  h_phoE_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " Photon Energy: barrel ", eBin, eMin, eMax);
  h_phoE_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " Photon Energy: Endcap ", eBin, eMin, eMax);

  histname = "phoEt";
  h_phoEt_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", " Photon Transverse Energy: All ecal ", etBin, etMin, etMax);
  h_phoEt_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", " Photon Transverse Energy: Barrel ", etBin, etMin, etMax);
  h_phoEt_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", " Photon Transverse Energy: Endcap ", etBin, etMin, etMax);

  histname = "eRes";
  h_phoERes_miniAOD_[0][0] = iBooker.book1D(
      histname + "All_miniAOD", " Photon E/E_{true}: All ecal;  E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_miniAOD_[0][1] = iBooker.book1D(
      histname + "Barrel_miniAOD", "Photon E/E_{true}: Barrel; E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_miniAOD_[0][2] = iBooker.book1D(
      histname + "Endcap_miniAOD", " Photon E/E_{true}: Endcap; E/E_{true} (GeV)", resBin, resMin, resMax);

  histname = "sigmaEoE";
  h_phoSigmaEoE_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "#sigma_{E}/E: All ecal; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "#sigma_{E}/E: Barrel; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "#sigma_{E}/E: Endcap, #sigma_{E}/E", 100, 0., 0.08);

  histname = "r9";
  h_r9_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", " r9: All Ecal", r9Bin, r9Min, r9Max);
  h_r9_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " r9: Barrel ", r9Bin, r9Min, r9Max);
  h_r9_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " r9: Endcap ", r9Bin, r9Min, r9Max);
  histname = "full5x5_r9";
  h_full5x5_r9_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", " r9: All Ecal", r9Bin, r9Min, r9Max);
  h_full5x5_r9_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " r9: Barrel ", r9Bin, r9Min, r9Max);
  h_full5x5_r9_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " r9: Endcap ", r9Bin, r9Min, r9Max);
  histname = "r1";
  h_r1_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", " e1x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r1_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " e1x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r1_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " e1x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  histname = "r2";
  h_r2_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", " e2x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r2_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " e2x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r2_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " e2x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  histname = "hOverE";
  h_hOverE_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", "H/E: All Ecal", 100, 0., 0.2);
  h_hOverE_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", "H/E: Barrel ", 100, 0., 0.2);
  h_hOverE_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", "H/E: Endcap ", 100, 0., 0.2);
  //
  histname = "newhOverE";
  h_newhOverE_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", "new H/E: All Ecal", 100, 0., 0.2);
  h_newhOverE_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", "new H/E: Barrel ", 100, 0., 0.2);
  h_newhOverE_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", "new H/E: Endcap ", 100, 0., 0.2);
  //
  histname = "sigmaIetaIeta";
  h_sigmaIetaIeta_miniAOD_[0][0] = iBooker.book1D(histname + "All_miniAOD", "sigmaIetaIeta: All Ecal", 100, 0., 0.1);
  h_sigmaIetaIeta_miniAOD_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", "sigmaIetaIeta: Barrel ", 100, 0., 0.05);
  h_sigmaIetaIeta_miniAOD_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", "sigmaIetaIeta: Endcap ", 100, 0., 0.1);
  histname = "full5x5_sigmaIetaIeta";
  h_full5x5_sigmaIetaIeta_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "Full5x5 sigmaIetaIeta: All Ecal", 100, 0., 0.1);
  h_full5x5_sigmaIetaIeta_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "Full5x5 sigmaIetaIeta: Barrel ", 100, 0., 0.05);
  h_full5x5_sigmaIetaIeta_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "Full5x5 sigmaIetaIeta: Endcap ", 100, 0., 0.1);
  //
  histname = "ecalRecHitSumEtConeDR04";
  h_ecalRecHitSumEtConeDR04_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "ecalRecHitSumEtDR04: All Ecal", etBin, etMin, 50.);
  h_ecalRecHitSumEtConeDR04_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "ecalRecHitSumEtDR04: Barrel ", etBin, etMin, 50.);
  h_ecalRecHitSumEtConeDR04_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "ecalRecHitSumEtDR04: Endcap ", etBin, etMin, 50.);
  histname = "hcalTowerSumEtConeDR04";
  h_hcalTowerSumEtConeDR04_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "hcalTowerSumEtConeDR04: All Ecal", etBin, etMin, 50.);
  h_hcalTowerSumEtConeDR04_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "hcalTowerSumEtConeDR04: Barrel ", etBin, etMin, 50.);
  h_hcalTowerSumEtConeDR04_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "hcalTowerSumEtConeDR04: Endcap ", etBin, etMin, 50.);
  //
  histname = "hcalTowerBcSumEtConeDR04";
  h_hcalTowerBcSumEtConeDR04_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "hcalTowerBcSumEtConeDR04: All Ecal", etBin, etMin, 50.);
  h_hcalTowerBcSumEtConeDR04_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "hcalTowerBcSumEtConeDR04: Barrel ", etBin, etMin, 50.);
  h_hcalTowerBcSumEtConeDR04_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "hcalTowerBcSumEtConeDR04: Endcap ", etBin, etMin, 50.);
  histname = "isoTrkSolidConeDR04";
  h_isoTrkSolidConeDR04_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "isoTrkSolidConeDR04: All Ecal", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "isoTrkSolidConeDR04: Barrel ", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "isoTrkSolidConeDR04: Endcap ", etBin, etMin, etMax * 0.1);
  histname = "nTrkSolidConeDR04";
  h_nTrkSolidConeDR04_miniAOD_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "nTrkSolidConeDR04: All Ecal", 20, 0., 20);
  h_nTrkSolidConeDR04_miniAOD_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "nTrkSolidConeDR04: Barrel ", 20, 0., 20);
  h_nTrkSolidConeDR04_miniAOD_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "nTrkSolidConeDR04: Endcap ", 20, 0., 20);

  //  Infos from Particle Flow - isolation and ID
  histname = "chargedHadIso";
  h_chHadIso_miniAOD_[0] = iBooker.book1D(histname + "All_miniAOD", "PF chargedHadIso:  All Ecal", etBin, etMin, 20.);
  h_chHadIso_miniAOD_[1] = iBooker.book1D(histname + "Barrel_miniAOD", "PF chargedHadIso:  Barrel", etBin, etMin, 20.);
  h_chHadIso_miniAOD_[2] = iBooker.book1D(histname + "Endcap_miniAOD", "PF chargedHadIso:  Endcap", etBin, etMin, 20.);
  histname = "neutralHadIso";
  h_nHadIso_miniAOD_[0] = iBooker.book1D(histname + "All_miniAOD", "PF neutralHadIso:  All Ecal", etBin, etMin, 20.);
  h_nHadIso_miniAOD_[1] = iBooker.book1D(histname + "Barrel_miniAOD", "PF neutralHadIso:  Barrel", etBin, etMin, 20.);
  h_nHadIso_miniAOD_[2] = iBooker.book1D(histname + "Endcap_miniAOD", "PF neutralHadIso:  Endcap", etBin, etMin, 20.);
  histname = "photonIso";
  h_phoIso_miniAOD_[0] = iBooker.book1D(histname + "All_miniAOD", "PF photonIso:  All Ecal", etBin, etMin, 20.);
  h_phoIso_miniAOD_[1] = iBooker.book1D(histname + "Barrel_miniAOD", "PF photonIso:  Barrel", etBin, etMin, 20.);
  h_phoIso_miniAOD_[2] = iBooker.book1D(histname + "Endcap_miniAOD", "PF photonIso:  Endcap", etBin, etMin, 20.);

  iBooker.setCurrentFolder("EgammaV/" + fName_ + "/ConversionInfo");

  histname = "nConv";
  h_nConv_[0][0] = iBooker.book1D(histname + "All",
                                  "Number Of two-tracks Conversions per isolated candidates per events: All Ecal  ",
                                  10,
                                  -0.5,
                                  9.5);
  h_nConv_[0][1] = iBooker.book1D(histname + "Barrel",
                                  "Number Of two-tracks Conversions per isolated candidates per events: Ecal Barrel  ",
                                  10,
                                  -0.5,
                                  9.5);
  h_nConv_[0][2] = iBooker.book1D(histname + "Endcap",
                                  "Number Of two-tracks Conversions per isolated candidates per events: Ecal Endcap ",
                                  10,
                                  -0.5,
                                  9.5);
  h_nConv_[1][0] = iBooker.book1D(histname + "OneLegAll",
                                  "Number Of single leg Conversions per isolated candidates per events: All Ecal  ",
                                  10,
                                  -0.5,
                                  9.5);
  h_nConv_[1][1] = iBooker.book1D(histname + "OneLegBarrel",
                                  "Number Of single leg Conversions per isolated candidates per events: Ecal Barrel  ",
                                  10,
                                  -0.5,
                                  9.5);
  h_nConv_[1][2] = iBooker.book1D(histname + "OneLegEndcap",
                                  "Number Of single leg Conversions per isolated candidates per events: Ecal Endcap ",
                                  10,
                                  -0.5,
                                  9.5);

  h_convEta_[0] = iBooker.book1D("convEta1", " converted Photon Eta >1 track", etaBin, etaMin, etaMax);
  h_convEta_[1] = iBooker.book1D("convEta2", " converted Photon Eta =2 tracks ", etaBin, etaMin, etaMax);
  h_convEta_[2] = iBooker.book1D("convEta2ass", " converted Photon Eta =2 tracks, both ass ", etaBin, etaMin, etaMax);
  h_convPhi_[0] = iBooker.book1D("convPhi", " converted Photon  Phi ", phiBin, phiMin, phiMax);

  histname = "convERes";
  h_convERes_[0][0] =
      iBooker.book1D(histname + "All", " Conversion rec/true Energy: All ecal ", resBin, resMin, resMax);
  h_convERes_[0][1] =
      iBooker.book1D(histname + "Barrel", " Conversion rec/true Energy: Barrel ", resBin, resMin, resMax);
  h_convERes_[0][2] =
      iBooker.book1D(histname + "Endcap", " Conversion rec/true Energy: Endcap ", resBin, resMin, resMax);

  histname = "p_EResVsR";
  p_eResVsR_ = iBooker.bookProfile(
      histname + "All", " photons conversion E/Etrue vs R: all Ecal ", rBin, rMin, rMax, 100, 0., 1.5, "");

  histname = "convPtRes";
  h_convPtRes_[1][0] =
      iBooker.book1D(histname + "All", " Conversion Pt rec/true  from tracks : All ecal ", resBin, 0., 1.5);
  h_convPtRes_[1][1] =
      iBooker.book1D(histname + "Barrel", " Conversion Pt rec/true  from tracks: Barrel ", resBin, 0., 1.5);
  h_convPtRes_[1][2] =
      iBooker.book1D(histname + "Endcap", " Conversion Pt rec/true  from tracks: Endcap ", resBin, 0., 1.5);

  if (!isRunCentrally_) {
    histname = "r9VsTracks";
    h_r9VsNofTracks_[0][0] = iBooker.book2D(
        histname + "All", " photons r9 vs nTracks from conversions: All Ecal", r9Bin, r9Min, r9Max, 3, -0.5, 2.5);
    h_r9VsNofTracks_[0][1] = iBooker.book2D(
        histname + "Barrel", " photons r9 vs nTracks from conversions: Barrel Ecal", r9Bin, r9Min, r9Max, 3, -0.5, 2.5);
    h_r9VsNofTracks_[0][2] = iBooker.book2D(
        histname + "Endcap", " photons r9 vs nTracks from conversions: Endcap Ecal", r9Bin, r9Min, r9Max, 3, -0.5, 2.5);
  }

  histname = "mvaOut";
  h_mvaOut_[0] = iBooker.book1D(histname + "All", " mvaOut for all conversions : All Ecal", 100, 0., 1.);
  h_mvaOut_[1] = iBooker.book1D(histname + "Barrel", " mvaOut for all conversions : Barrel Ecal", 100, 0., 1.);
  h_mvaOut_[2] = iBooker.book1D(histname + "Endcap", " mvaOut for all conversions : Endcap Ecal", 100, 0., 1.);

  histname = "EoverPtracks";
  h_EoverPTracks_[0][0] =
      iBooker.book1D(histname + "BarrelPix", " photons conversion E/p: barrel pix", eoverpBin, eoverpMin, eoverpMax);
  h_EoverPTracks_[0][1] =
      iBooker.book1D(histname + "BarrelTib", " photons conversion E/p: barrel tib", eoverpBin, eoverpMin, eoverpMax);
  h_EoverPTracks_[0][2] =
      iBooker.book1D(histname + "BarrelTob", " photons conversion E/p: barrel tob ", eoverpBin, eoverpMin, eoverpMax);

  h_EoverPTracks_[1][0] = iBooker.book1D(histname + "All", " photons conversion E/p: all Ecal ", 100, 0., 5.);
  h_EoverPTracks_[1][1] = iBooker.book1D(histname + "Barrel", " photons conversion E/p: Barrel Ecal", 100, 0., 5.);
  h_EoverPTracks_[1][2] = iBooker.book1D(histname + "Endcap", " photons conversion E/p: Endcap Ecal ", 100, 0., 5.);
  histname = "EoverP_SL";
  h_EoverP_SL_[0] = iBooker.book1D(histname + "All", " photons single leg conversion E/p: all Ecal ", 100, 0., 5.);
  h_EoverP_SL_[1] = iBooker.book1D(histname + "Barrel", " photons single leg conversion E/p: Barrel Ecal", 100, 0., 5.);
  h_EoverP_SL_[2] =
      iBooker.book1D(histname + "Endcap", " photons single leg conversion E/p: Endcap Ecal ", 100, 0., 5.);

  histname = "PoverEtracks";
  h_PoverETracks_[1][0] =
      iBooker.book1D(histname + "All", " photons conversion p/E: all Ecal ", povereBin, povereMin, povereMax);
  h_PoverETracks_[1][1] =
      iBooker.book1D(histname + "Barrel", " photons conversion p/E: Barrel Ecal", povereBin, povereMin, povereMax);
  h_PoverETracks_[1][2] =
      iBooker.book1D(histname + "Endcap", " photons conversion p/E: Endcap Ecal ", povereBin, povereMin, povereMax);

  histname = "pEoverEtrueVsEta";
  p_EoverEtrueVsEta_[0] =
      iBooker.bookProfile(histname + "All",
                          " photons conversion with 2 (associated) reco tracks E/Etrue vs #eta: all Ecal ",
                          etaBin2,
                          etaMin,
                          etaMax,
                          100,
                          0.,
                          2.5,
                          "");

  histname = "pEoverEtrueVsR";
  p_EoverEtrueVsR_[0] = iBooker.bookProfile(
      histname + "All", " photons conversion E/Etrue vs R: all Ecal ", rBin, rMin, rMax, 100, 0., 2.5, "");

  histname = "pEoverEtrueVsEta";
  p_EoverEtrueVsEta_[1] = iBooker.bookProfile(histname + "All2",
                                              " photons conversion  2 reco tracks  E/Etrue vs #eta: all Ecal ",
                                              etaBin2,
                                              etaMin,
                                              etaMax,
                                              100,
                                              0.,
                                              2.5,
                                              "");

  histname = "pPoverPtrueVsEta";
  p_PoverPtrueVsEta_[0] = iBooker.bookProfile(
      histname + "All", " photons conversion P/Ptrue vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 5., "");

  histname = "pEoverPVsEta";
  p_EoverPVsEta_[0] = iBooker.bookProfile(
      histname + "All", " photons conversion E/P vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 5., "");

  if (!isRunCentrally_) {
    histname = "EoverEtrueVsEoverP";
    h2_EoverEtrueVsEoverP_[0] =
        iBooker.book2D(histname + "All", " photons conversion E/Etrue vs E/P: all Ecal ", 100, 0., 5., 100, 0.5, 1.5);
    h2_EoverEtrueVsEoverP_[1] = iBooker.book2D(
        histname + "Barrel", " photons conversion  E/Etrue vs E/: Barrel Ecal", 100, 0., 5., 100, 0.5, 1.5);
    h2_EoverEtrueVsEoverP_[2] = iBooker.book2D(
        histname + "Endcap", " photons conversion  E/Etrue vs E/: Endcap Ecal ", 100, 0., 5., 100, 0.5, 1.5);
    histname = "PoverPtrueVsEoverP";
    h2_PoverPtrueVsEoverP_[0] =
        iBooker.book2D(histname + "All", " photons conversion P/Ptrue vs E/P: all Ecal ", 100, 0., 5., 100, 0., 2.5);
    h2_PoverPtrueVsEoverP_[1] = iBooker.book2D(
        histname + "Barrel", " photons conversion  P/Ptrue vs E/: Barrel Ecal", 100, 0., 5., 100, 0., 2.5);
    h2_PoverPtrueVsEoverP_[2] = iBooker.book2D(
        histname + "Endcap", " photons conversion  P/Ptrue vs E/: Endcap Ecal ", 100, 0., 5., 100, 0., 2.5);

    histname = "EoverEtrueVsEta";
    h2_EoverEtrueVsEta_[0] =
        iBooker.book2D(histname + "All",
                       " photons conversion with 2 (associated) reco tracks  E/Etrue vs #eta: all Ecal ",
                       etaBin2,
                       etaMin,
                       etaMax,
                       100,
                       0.,
                       2.5);

    histname = "EoverEtrueVsEta";
    h2_EoverEtrueVsEta_[1] = iBooker.book2D(histname + "All2",
                                            " photons conversion  2 reco tracks  E/Etrue vs #eta: all Ecal ",
                                            etaBin2,
                                            etaMin,
                                            etaMax,
                                            100,
                                            0.,
                                            2.5);

    histname = "EoverEtrueVsR";
    h2_EoverEtrueVsR_[0] =
        iBooker.book2D(histname + "All", " photons conversion E/Etrue vs R: all Ecal ", rBin, rMin, rMax, 100, 0., 2.5);

    histname = "PoverPtrueVsEta";
    h2_PoverPtrueVsEta_[0] = iBooker.book2D(
        histname + "All", " photons conversion P/Ptrue vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 5.);

    histname = "EoverPVsEta";
    h2_EoverPVsEta_[0] = iBooker.book2D(
        histname + "All", " photons conversion E/P vs #eta: all Ecal ", etaBin2, etaMin, etaMax, 100, 0., 5.);

    histname = "EoverPVsR";
    h2_EoverPVsR_[0] =
        iBooker.book2D(histname + "All", " photons conversion E/P vs R: all Ecal ", rBin, rMin, rMax, 100, 0., 5.);

    histname = "etaVsRsim";
    h2_etaVsRsim_[0] = iBooker.book2D(histname + "All",
                                      " eta(sim) vs R (sim) for associated conversions: all Ecal ",
                                      etaBin,
                                      etaMin,
                                      etaMax,
                                      rBin,
                                      rMin,
                                      rMax);
    histname = "etaVsRreco";
    h2_etaVsRreco_[0] = iBooker.book2D(histname + "All",
                                       " eta(reco) vs R (reco) for associated conversions: all Ecal ",
                                       etaBin,
                                       etaMin,
                                       etaMax,
                                       rBin,
                                       rMin,
                                       rMax);
  }

  histname = "pEoverPVsR";
  p_EoverPVsR_[0] = iBooker.bookProfile(
      histname + "All", " photons conversion E/P vs R: all Ecal ", rBin, rMin, rMax, 100, 0., 5., "");

  histname = "hInvMass";
  h_invMass_[0][0] = iBooker.book1D(
      histname + "All_AllTracks", " Photons:Tracks from conversion: Pair invariant mass: all Ecal ", 100, 0., 1.5);
  h_invMass_[0][1] = iBooker.book1D(
      histname + "Barrel_AllTracks", " Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ", 100, 0., 1.5);
  h_invMass_[0][2] = iBooker.book1D(
      histname + "Endcap_AllTracks", " Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ", 100, 0., 1.5);
  histname = "hInvMass";
  h_invMass_[1][0] = iBooker.book1D(
      histname + "All_AssTracks", " Photons:Tracks from conversion: Pair invariant mass: all Ecal ", 100, 0., 1.5);
  h_invMass_[1][1] = iBooker.book1D(
      histname + "Barrel_AssTracks", " Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ", 100, 0., 1.5);
  h_invMass_[1][2] = iBooker.book1D(
      histname + "Endcap_AssTracks", " Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ", 100, 0., 1.5);

  histname = "hDPhiTracksAtVtx";
  h_DPhiTracksAtVtx_[1][0] = iBooker.book1D(histname + "All",
                                            " Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",
                                            dPhiTracksBin,
                                            dPhiTracksMin,
                                            dPhiTracksMax);
  h_DPhiTracksAtVtx_[1][1] =
      iBooker.book1D(histname + "Barrel",
                     " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",
                     dPhiTracksBin,
                     dPhiTracksMin,
                     dPhiTracksMax);
  h_DPhiTracksAtVtx_[1][2] =
      iBooker.book1D(histname + "Endcap",
                     " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",
                     dPhiTracksBin,
                     dPhiTracksMin,
                     dPhiTracksMax);

  if (!isRunCentrally_) {
    histname = "hDPhiTracksAtVtxVsEta";
    h2_DPhiTracksAtVtxVsEta_ = iBooker.book2D(histname + "All",
                                              "  Photons:Tracks from conversions: #delta#phi Tracks at vertex vs #eta",
                                              etaBin2,
                                              etaMin,
                                              etaMax,
                                              100,
                                              -0.5,
                                              0.5);

    histname = "hDPhiTracksAtVtxVsR";
    h2_DPhiTracksAtVtxVsR_ = iBooker.book2D(histname + "All",
                                            "  Photons:Tracks from conversions: #delta#phi Tracks at vertex vs R",
                                            rBin,
                                            rMin,
                                            rMax,
                                            100,
                                            -0.5,
                                            0.5);

    histname = "hDCotTracksVsEta";
    h2_DCotTracksVsEta_ = iBooker.book2D(histname + "All",
                                         "  Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks vs #eta",
                                         etaBin2,
                                         etaMin,
                                         etaMax,
                                         100,
                                         -0.2,
                                         0.2);

    histname = "hDCotTracksVsR";
    h2_DCotTracksVsR_ = iBooker.book2D(histname + "All",
                                       "  Photons:Tracks from conversions:  #delta cotg(#Theta)  Tracks at vertex vs R",
                                       rBin,
                                       rMin,
                                       rMax,
                                       100,
                                       -0.2,
                                       0.2);

    histname = "h2_DPhiTracksAtEcalVsR";
    if (fName_ != "pfPhotonValidator" && fName_ != "oldpfPhotonValidator")
      h2_DPhiTracksAtEcalVsR_ = iBooker.book2D(histname + "All",
                                               " Photons:Tracks from conversions:  #delta#phi at Ecal vs R : all Ecal ",
                                               rBin,
                                               rMin,
                                               rMax,
                                               dPhiTracksBin,
                                               0.,
                                               dPhiTracksMax);

    histname = "h2_DPhiTracksAtEcalVsEta";
    if (fName_ != "pfPhotonValidator" && fName_ != "oldpfPhotonValidator")
      h2_DPhiTracksAtEcalVsEta_ =
          iBooker.book2D(histname + "All",
                         " Photons:Tracks from conversions:  #delta#phi at Ecal vs #eta : all Ecal ",
                         etaBin2,
                         etaMin,
                         etaMax,
                         dPhiTracksBin,
                         0.,
                         dPhiTracksMax);
  }

  histname = "pDPhiTracksAtVtxVsEta";
  p_DPhiTracksAtVtxVsEta_ =
      iBooker.bookProfile(histname + "All",
                          " Photons:Tracks from conversions: #delta#phi Tracks at vertex vs #eta ",
                          etaBin2,
                          etaMin,
                          etaMax,
                          100,
                          -0.5,
                          0.5,
                          "");

  histname = "pDPhiTracksAtVtxVsR";
  p_DPhiTracksAtVtxVsR_ = iBooker.bookProfile(histname + "All",
                                              " Photons:Tracks from conversions: #delta#phi Tracks at vertex vs R ",
                                              rBin,
                                              rMin,
                                              rMax,
                                              100,
                                              -0.5,
                                              0.5,
                                              "");

  histname = "hDCotTracks";
  h_DCotTracks_[1][0] = iBooker.book1D(histname + "All",
                                       " Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",
                                       dCotTracksBin,
                                       dCotTracksMin,
                                       dCotTracksMax);
  h_DCotTracks_[1][1] = iBooker.book1D(histname + "Barrel",
                                       " Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",
                                       dCotTracksBin,
                                       dCotTracksMin,
                                       dCotTracksMax);
  h_DCotTracks_[1][2] = iBooker.book1D(histname + "Endcap",
                                       " Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",
                                       dCotTracksBin,
                                       dCotTracksMin,
                                       dCotTracksMax);

  histname = "pDCotTracksVsEta";
  p_DCotTracksVsEta_ = iBooker.bookProfile(histname + "All",
                                           " Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks vs #eta ",
                                           etaBin2,
                                           etaMin,
                                           etaMax,
                                           100,
                                           -0.2,
                                           0.2,
                                           "");

  histname = "pDCotTracksVsR";
  p_DCotTracksVsR_ =
      iBooker.bookProfile(histname + "All",
                          " Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks at vertex vs R ",
                          rBin,
                          rMin,
                          rMax,
                          100,
                          -0.2,
                          0.2,
                          "");

  histname = "hDistMinAppTracks";
  h_distMinAppTracks_[1][0] = iBooker.book1D(histname + "All",
                                             " Photons:Tracks from conversions Min Approach Dist Tracks: all Ecal ",
                                             dEtaTracksBin,
                                             -0.1,
                                             0.6);
  h_distMinAppTracks_[1][1] = iBooker.book1D(histname + "Barrel",
                                             " Photons:Tracks from conversions Min Approach Dist Tracks: Barrel Ecal ",
                                             dEtaTracksBin,
                                             -0.1,
                                             0.6);
  h_distMinAppTracks_[1][2] = iBooker.book1D(histname + "Endcap",
                                             " Photons:Tracks from conversions Min Approach Dist Tracks: Endcap Ecal ",
                                             dEtaTracksBin,
                                             -0.1,
                                             0.6);

  // if ( fName_ != "pfPhotonValidator" &&  fName_ != "oldpfPhotonValidator" ) {
  histname = "hDPhiTracksAtEcal";
  h_DPhiTracksAtEcal_[1][0] = iBooker.book1D(histname + "All",
                                             " Photons:Tracks from conversions:  #delta#phi at Ecal : all Ecal ",
                                             dPhiTracksBin,
                                             0.,
                                             dPhiTracksMax);
  h_DPhiTracksAtEcal_[1][1] = iBooker.book1D(histname + "Barrel",
                                             " Photons:Tracks from conversions:  #delta#phi at Ecal : Barrel Ecal ",
                                             dPhiTracksBin,
                                             0.,
                                             dPhiTracksMax);
  h_DPhiTracksAtEcal_[1][2] = iBooker.book1D(histname + "Endcap",
                                             " Photons:Tracks from conversions:  #delta#phi at Ecal : Endcap Ecal ",
                                             dPhiTracksBin,
                                             0.,
                                             dPhiTracksMax);

  histname = "pDPhiTracksAtEcalVsR";
  p_DPhiTracksAtEcalVsR_ = iBooker.bookProfile(histname + "All",
                                               " Photons:Tracks from conversions:  #delta#phi at Ecal  vs R ",
                                               rBin,
                                               rMin,
                                               rMax,
                                               dPhiTracksBin,
                                               0.,
                                               dPhiTracksMax,
                                               "");

  histname = "pDPhiTracksAtEcalVsEta";
  p_DPhiTracksAtEcalVsEta_ = iBooker.bookProfile(histname + "All",
                                                 " Photons:Tracks from conversions:  #delta#phi at Ecal  vs #eta ",
                                                 etaBin2,
                                                 etaMin,
                                                 etaMax,
                                                 dPhiTracksBin,
                                                 0.,
                                                 dPhiTracksMax,
                                                 "");

  histname = "hDEtaTracksAtEcal";
  h_DEtaTracksAtEcal_[1][0] = iBooker.book1D(histname + "All",
                                             " Photons:Tracks from conversions:  #delta#eta at Ecal : all Ecal ",
                                             dEtaTracksBin,
                                             dEtaTracksMin,
                                             dEtaTracksMax);
  h_DEtaTracksAtEcal_[1][1] = iBooker.book1D(histname + "Barrel",
                                             " Photons:Tracks from conversions:  #delta#eta at Ecal : Barrel Ecal ",
                                             dEtaTracksBin,
                                             dEtaTracksMin,
                                             dEtaTracksMax);
  h_DEtaTracksAtEcal_[1][2] = iBooker.book1D(histname + "Endcap",
                                             " Photons:Tracks from conversions:  #delta#eta at Ecal : Endcap Ecal ",
                                             dEtaTracksBin,
                                             dEtaTracksMin,
                                             dEtaTracksMax);

  //  }

  h_convVtxRvsZ_[0] = iBooker.book2D("convVtxRvsZAll",
                                     " Photon Reco conversion vtx position",
                                     zBinForXray,
                                     zMinForXray,
                                     zMaxForXray,
                                     rBinForXray,
                                     rMinForXray,
                                     rMaxForXray);
  h_convVtxRvsZ_[1] = iBooker.book2D("convVtxRvsZBarrel",
                                     " Photon Reco conversion vtx position",
                                     zBinForXray,
                                     zMinForXray,
                                     zMaxForXray,
                                     rBinForXray,
                                     rMinForXray,
                                     rMaxForXray);
  h_convVtxRvsZ_[2] = iBooker.book2D("convVtxRvsZEndcap",
                                     " Photon Reco conversion vtx position",
                                     zBin2ForXray,
                                     zMinForXray,
                                     zMaxForXray,
                                     rBinForXray,
                                     rMinForXray,
                                     rMaxForXray);
  h_convVtxYvsX_ = iBooker.book2D(
      "convVtxYvsXTrkBarrel", " Photon Reco conversion vtx position, (x,y) eta<1 ", 100, -80., 80., 100, -80., 80.);
  //
  h_convSLVtxRvsZ_[0] = iBooker.book2D("convSLVtxRvsZAll",
                                       " Photon Reco single leg conversion innermost hit  position",
                                       zBinForXray,
                                       zMinForXray,
                                       zMaxForXray,
                                       rBinForXray,
                                       rMinForXray,
                                       rMaxForXray);
  h_convSLVtxRvsZ_[1] = iBooker.book2D("convSLVtxRvsZBarrel",
                                       " Photon Reco single leg conversion innermost hit position",
                                       zBinForXray,
                                       zMinForXray,
                                       zMaxForXray,
                                       rBinForXray,
                                       rMinForXray,
                                       rMaxForXray);
  h_convSLVtxRvsZ_[2] = iBooker.book2D("convSLVtxRvsZEndcap",
                                       " Photon Reco single leg conversion innermost hit position",
                                       zBin2ForXray,
                                       zMinForXray,
                                       zMaxForXray,
                                       rBinForXray,
                                       rMinForXray,
                                       rMaxForXray);

  /// zooms
  if (!isRunCentrally_) {
    h_convVtxRvsZ_zoom_[0] = iBooker.book2D("convVtxRvsZBarrelZoom1",
                                            " Photon Reco conversion vtx position",
                                            zBinForXray,
                                            zMinForXray,
                                            zMaxForXray,
                                            rBinForXray,
                                            -10.,
                                            40.);
    h_convVtxRvsZ_zoom_[1] = iBooker.book2D("convVtxRvsZBarrelZoom2",
                                            " Photon Reco conversion vtx position",
                                            zBinForXray,
                                            zMinForXray,
                                            zMaxForXray,
                                            rBinForXray,
                                            -10.,
                                            20.);
    h_convVtxYvsX_zoom_[0] = iBooker.book2D("convVtxYvsXTrkBarrelZoom1",
                                            " Photon Reco conversion vtx position, (x,y) eta<1 ",
                                            100,
                                            -40.,
                                            40.,
                                            100,
                                            -40.,
                                            40.);
    h_convVtxYvsX_zoom_[1] = iBooker.book2D("convVtxYvsXTrkBarrelZoom2",
                                            " Photon Reco conversion vtx position, (x,y) eta<1 ",
                                            100,
                                            -20.,
                                            20.,
                                            100,
                                            -20.,
                                            20.);
  }

  h_convVtxdX_ = iBooker.book1D("convVtxdX", " Photon Reco conversion vtx dX", 100, -20., 20.);
  h_convVtxdY_ = iBooker.book1D("convVtxdY", " Photon Reco conversion vtx dY", 100, -20., 20.);
  h_convVtxdZ_ = iBooker.book1D("convVtxdZ", " Photon Reco conversion vtx dZ", 100, -20., 20.);
  h_convVtxdR_ = iBooker.book1D("convVtxdR", " Photon Reco conversion vtx dR", 100, -20., 20.);

  h_convVtxdX_barrel_ =
      iBooker.book1D("convVtxdX_barrel", " Photon Reco conversion vtx dX, |eta|<=1.2", 100, -20., 20.);
  h_convVtxdY_barrel_ =
      iBooker.book1D("convVtxdY_barrel", " Photon Reco conversion vtx dY, |eta|<=1.2 ", 100, -20., 20.);
  h_convVtxdZ_barrel_ =
      iBooker.book1D("convVtxdZ_barrel", " Photon Reco conversion vtx dZ, |eta|<=1.2,", 100, -20., 20.);
  h_convVtxdR_barrel_ =
      iBooker.book1D("convVtxdR_barrel", " Photon Reco conversion vtx dR, |eta|<=1.2", 100, -20., 20.);
  h_convVtxdX_endcap_ =
      iBooker.book1D("convVtxdX_endcap", " Photon Reco conversion vtx dX,  |eta|>1.2", 100, -20., 20.);
  h_convVtxdY_endcap_ =
      iBooker.book1D("convVtxdY_endcap", " Photon Reco conversion vtx dY,  |eta|>1.2", 100, -20., 20.);
  h_convVtxdZ_endcap_ =
      iBooker.book1D("convVtxdZ_endcap", " Photon Reco conversion vtx dZ,  |eta|>1.2", 100, -20., 20.);
  h_convVtxdR_endcap_ =
      iBooker.book1D("convVtxdR_endcap", " Photon Reco conversion vtx dR,  |eta|>1.2 ", 100, -20., 20.);

  h_convVtxdPhi_ = iBooker.book1D("convVtxdPhi", " Photon Reco conversion vtx dPhi", 100, -0.005, 0.005);
  h_convVtxdEta_ = iBooker.book1D("convVtxdEta", " Photon Reco conversion vtx dEta", 100, -0.5, 0.5);

  if (!isRunCentrally_) {
    h2_convVtxdRVsR_ =
        iBooker.book2D("h2ConvVtxdRVsR", "Photon Reco conversion vtx dR vsR", rBin, rMin, rMax, 100, -20., 20.);
    h2_convVtxdRVsEta_ = iBooker.book2D(
        "h2ConvVtxdRVsEta", "Photon Reco conversion vtx dR vs Eta", etaBin2, etaMin, etaMax, 100, -20., 20.);
  }

  p_convVtxdRVsR_ =
      iBooker.bookProfile("pConvVtxdRVsR", "Photon Reco conversion vtx dR vsR", rBin, rMin, rMax, 100, -20., 20., "");
  p_convVtxdRVsEta_ = iBooker.bookProfile(
      "pConvVtxdRVsEta", "Photon Reco conversion vtx dR vs Eta", etaBin2, etaMin, etaMax, 100, -20., 20., "");
  p_convVtxdXVsX_ = iBooker.bookProfile("pConvVtxdXVsX", "Conversion vtx dX vs X", 120, -60, 60, 100, -20., 20., "");
  p_convVtxdYVsY_ = iBooker.bookProfile("pConvVtxdYVsY", "Conversion vtx dY vs Y", 120, -60, 60, 100, -20., 20., "");
  p_convVtxdZVsZ_ =
      iBooker.bookProfile("pConvVtxdZVsZ", "Conversion vtx dZ vs Z", zBin, zMin, zMax, 100, -20., 20., "");

  if (!isRunCentrally_) {
    h2_convVtxRrecVsTrue_ = iBooker.book2D(
        "h2ConvVtxRrecVsTrue", "Photon Reco conversion vtx R rec vs true", rBin, rMin, rMax, rBin, rMin, rMax);
  }

  histname = "vtxChi2";
  h_vtxChi2_[0] = iBooker.book1D(histname + "All", "vertex #chi^{2} all", 100, chi2Min, chi2Max);
  h_vtxChi2_[1] = iBooker.book1D(histname + "Barrel", "vertex #chi^{2} barrel", 100, chi2Min, chi2Max);
  h_vtxChi2_[2] = iBooker.book1D(histname + "Endcap", "vertex #chi^{2} endcap", 100, chi2Min, chi2Max);
  histname = "vtxChi2Prob";
  h_vtxChi2Prob_[0] = iBooker.book1D(histname + "All", "vertex #chi^{2} all", 100, 0., 1.);
  h_vtxChi2Prob_[1] = iBooker.book1D(histname + "Barrel", "vertex #chi^{2} barrel", 100, 0., 1.);
  h_vtxChi2Prob_[2] = iBooker.book1D(histname + "Endcap", "vertex #chi^{2} endcap", 100, 0., 1.);

  histname = "zPVFromTracks";
  h_zPVFromTracks_[0] = iBooker.book1D(histname + "All", " Photons: PV z from conversion tracks", 100, -30., 30.);
  h_zPVFromTracks_[1] = iBooker.book1D(histname + "Barrel", " Photons: PV z from conversion tracks", 100, -30., 30.);
  h_zPVFromTracks_[2] = iBooker.book1D(histname + "Endcap", " Photons: PV z from conversion tracks", 100, -30., 30.);
  h_zPVFromTracks_[3] = iBooker.book1D(histname + "EndcapP", " Photons: PV z from conversion tracks", 100, -30., 30.);
  h_zPVFromTracks_[4] = iBooker.book1D(histname + "EndcapM", " Photons: PV z from conversion tracks", 100, -30., 30.);
  histname = "dzPVFromTracks";
  h_dzPVFromTracks_[0] =
      iBooker.book1D(histname + "All", " Photons: PV Z_rec - Z_true from conversion tracks", 100, -10., 10.);
  h_dzPVFromTracks_[1] =
      iBooker.book1D(histname + "Barrel", " Photons: PV Z_rec - Z_true from conversion tracks", 100, -10., 10.);
  h_dzPVFromTracks_[2] =
      iBooker.book1D(histname + "Endcap", " Photons: PV Z_rec - Z_true from conversion tracks", 100, -10., 10.);
  h_dzPVFromTracks_[3] =
      iBooker.book1D(histname + "EndcapP", " Photons: PV Z_rec - Z_true from conversion tracks", 100, -10., 10.);
  h_dzPVFromTracks_[4] =
      iBooker.book1D(histname + "EndcapM", " Photons: PV Z_rec - Z_true from conversion tracks", 100, -10., 10.);
  p_dzPVVsR_ =
      iBooker.bookProfile("pdzPVVsR", "Photon Reco conversions: dz(PV) vs R", rBin, rMin, rMax, 100, -3., 3., "");
  p_dzPVVsEta_ = iBooker.bookProfile(
      "pdzPVVsEta", "Photon Reco conversions: dz(PV) vs Eta", etaBin, etaMin, etaMax, 100, -3., 3., "");

  if (!isRunCentrally_) {
    h2_dzPVVsR_ = iBooker.book2D("h2dzPVVsR", "Photon Reco conversions: dz(PV) vs R", rBin, rMin, rMax, 100, -3., 3.);
  }

  //////////////////// plots per track
  if (!isRunCentrally_) {
    histname = "nHitsVsEta";
    nHitsVsEta_[0] = iBooker.book2D(histname + "AllTracks",
                                    "Photons:Tracks from conversions: # of hits vs #eta all tracks",
                                    etaBin,
                                    etaMin,
                                    etaMax,
                                    25,
                                    0.,
                                    25.);

    histname = "nHitsVsEta";
    nHitsVsEta_[1] = iBooker.book2D(histname + "AssTracks",
                                    "Photons:Tracks from conversions: # of hits vs #eta associated tracks",
                                    etaBin,
                                    etaMin,
                                    etaMax,
                                    25,
                                    0.,
                                    25.);

    histname = "nHitsVsR";
    nHitsVsR_[0] = iBooker.book2D(histname + "AllTracks",
                                  "Photons:Tracks from conversions: # of hits vs radius all tracks",
                                  rBin,
                                  rMin,
                                  rMax,
                                  25,
                                  0.,
                                  25);

    histname = "nHitsVsR";
    nHitsVsR_[1] = iBooker.book2D(histname + "AssTracks",
                                  "Photons:Tracks from conversions: # of hits vs radius associated tracks",
                                  rBin,
                                  rMin,
                                  rMax,
                                  25,
                                  0.,
                                  25);

    histname = "h2Chi2VsEta";
    h2_Chi2VsEta_[0] = iBooker.book2D(
        histname + "All", " Reco Track  #chi^{2} vs #eta: All ", etaBin2, etaMin, etaMax, 100, chi2Min, chi2Max);

    histname = "h2Chi2VsR";
    h2_Chi2VsR_[0] =
        iBooker.book2D(histname + "All", " Reco Track  #chi^{2} vs R: All ", rBin, rMin, rMax, 100, chi2Min, chi2Max);
  }

  histname = "h_nHitsVsEta";
  p_nHitsVsEta_[0] = iBooker.bookProfile(histname + "AllTracks",
                                         "Photons:Tracks from conversions: # of hits vs #eta all tracks",
                                         etaBin,
                                         etaMin,
                                         etaMax,
                                         25,
                                         -0.5,
                                         24.5,
                                         "");

  histname = "h_nHitsVsEta";
  p_nHitsVsEta_[1] = iBooker.bookProfile(histname + "AssTracks",
                                         "Photons:Tracks from conversions: # of hits vs #eta associated tracks",
                                         etaBin,
                                         etaMin,
                                         etaMax,
                                         25,
                                         -0.5,
                                         24.5,
                                         "");

  histname = "p_nHitsVsEtaSL";
  p_nHitsVsEtaSL_[0] = iBooker.bookProfile(histname + "AllTracks",
                                           "Photons:Tracks from single leg conversions: # of hits vs #eta all tracks",
                                           etaBin,
                                           etaMin,
                                           etaMax,
                                           25,
                                           -0.5,
                                           24.5,
                                           "");

  histname = "h_nHitsVsR";
  p_nHitsVsR_[0] = iBooker.bookProfile(histname + "AllTracks",
                                       "Photons:Tracks from conversions: # of hits vs radius all tracks",
                                       rBin,
                                       rMin,
                                       rMax,
                                       25,
                                       -0.5,
                                       24.5,
                                       "");
  histname = "p_nHitsVsRSL";
  p_nHitsVsRSL_[0] = iBooker.bookProfile(histname + "AllTracks",
                                         "Photons:Tracks from single leg conversions: # of hits vs radius all tracks",
                                         rBin,
                                         rMin,
                                         rMax,
                                         25,
                                         -0.5,
                                         24.5,
                                         "");

  histname = "tkChi2";
  h_tkChi2_[0] = iBooker.book1D(
      histname + "AllTracks", "Photons:Tracks from conversions: #chi^{2} of all tracks", 100, chi2Min, chi2Max);
  histname = "tkChi2SL";
  h_tkChi2SL_[0] = iBooker.book1D(histname + "AllTracks",
                                  "Photons:Tracks from single leg conversions: #chi^{2} of associated  tracks",
                                  100,
                                  chi2Min,
                                  chi2Max);
  histname = "tkChi2Large";
  h_tkChi2Large_[0] = iBooker.book1D(
      histname + "AllTracks", "Photons:Tracks from conversions: #chi^{2} of all tracks", 1000, 0., 5000.0);

  histname = "h_nHitsVsR";
  p_nHitsVsR_[1] = iBooker.bookProfile(histname + "AssTracks",
                                       "Photons:Tracks from conversions: # of hits vs radius associated tracks",
                                       rBin,
                                       rMin,
                                       rMax,
                                       25,
                                       -0.5,
                                       24.5,
                                       "");

  histname = "tkChi2";
  h_tkChi2_[1] = iBooker.book1D(
      histname + "AssTracks", "Photons:Tracks from conversions: #chi^{2} of associated  tracks", 100, chi2Min, chi2Max);
  histname = "tkChi2Large";
  h_tkChi2Large_[1] = iBooker.book1D(
      histname + "AssTracks", "Photons:Tracks from conversions: #chi^{2} of associated  tracks", 1000, 0., 5000.0);

  histname = "pChi2VsEta";
  p_Chi2VsEta_[0] = iBooker.bookProfile(
      histname + "All", " Reco Track #chi^{2} vs #eta : All ", etaBin2, etaMin, etaMax, 100, chi2Min, chi2Max, "");

  histname = "pChi2VsR";
  p_Chi2VsR_[0] = iBooker.bookProfile(
      histname + "All", " Reco Track #chi^{2} vas R : All ", rBin, rMin, rMax, 100, chi2Min, chi2Max, "");

  histname = "hTkD0";
  h_TkD0_[0] = iBooker.book1D(histname + "All", " Reco Track D0*q: All ", 100, -0.1, 0.6);
  h_TkD0_[1] = iBooker.book1D(histname + "Barrel", " Reco Track D0*q: Barrel ", 100, -0.1, 0.6);
  h_TkD0_[2] = iBooker.book1D(histname + "Endcap", " Reco Track D0*q: Endcap ", 100, -0.1, 0.6);

  histname = "hTkPtPull";
  h_TkPtPull_[0] = iBooker.book1D(histname + "All", " Reco Track Pt pull: All ", 100, -10., 10.);
  histname = "hTkPtPull";
  h_TkPtPull_[1] = iBooker.book1D(histname + "Barrel", " Reco Track Pt pull: Barrel ", 100, -10., 10.);
  histname = "hTkPtPull";
  h_TkPtPull_[2] = iBooker.book1D(histname + "Endcap", " Reco Track Pt pull: Endcap ", 100, -10., 10.);

  histname = "pTkPtPullEta";
  p_TkPtPull_[0] =
      iBooker.bookProfile(histname + "All", " Reco Track Pt pull: All ", etaBin2, etaMin, etaMax, 100, -10., 10., " ");

  if (!isRunCentrally_) {
    histname = "h2TkPtPullEta";
    h2_TkPtPull_[0] =
        iBooker.book2D(histname + "All", " Reco Track Pt pull: All ", etaBin2, etaMin, etaMax, 100, -10., 10.);

    histname = "PtRecVsPtSim";
    h2_PtRecVsPtSim_[0] =
        iBooker.book2D(histname + "All", "Pt Rec vs Pt sim: All ", etBin, etMin, etMax, etBin, etMin, etMax);
    h2_PtRecVsPtSim_[1] =
        iBooker.book2D(histname + "Barrel", "Pt Rec vs Pt sim: Barrel ", etBin, etMin, etMax, etBin, etMin, etMax);
    h2_PtRecVsPtSim_[2] =
        iBooker.book2D(histname + "Endcap", "Pt Rec vs Pt sim: Endcap ", etBin, etMin, etMax, etBin, etMin, etMax);
    histname = "PtRecVsPtSimMixProv";
    h2_PtRecVsPtSimMixProv_ = iBooker.book2D(
        histname + "All", "Pt Rec vs Pt sim All for mix with general tracks ", etBin, etMin, etMax, etBin, etMin, etMax);
  }

  // if ( fName_ != "pfPhotonValidator" &&  fName_ != "oldpfPhotonValidator" ) {
  histname = "eBcOverTkPout";
  hBCEnergyOverTrackPout_[0] = iBooker.book1D(histname + "All", "Matrching BC E/P_out: all Ecal ", 100, 0., 5.);
  hBCEnergyOverTrackPout_[1] = iBooker.book1D(histname + "Barrel", "Matrching BC E/P_out: Barrel ", 100, 0., 5.);
  hBCEnergyOverTrackPout_[2] = iBooker.book1D(histname + "Endcap", "Matrching BC E/P_out: Endcap ", 100, 0., 5.);
  // }

  ////////////// test on OutIn tracks
  h_OIinnermostHitR_ = iBooker.book1D("OIinnermostHitR", " R innermost hit for OI tracks ", 50, 0., 25);
  h_IOinnermostHitR_ = iBooker.book1D("IOinnermostHitR", " R innermost hit for IO tracks ", 50, 0., 25);

  /// test track provenance
  h_trkProv_[0] = iBooker.book1D("allTrkProv", " Track pair provenance ", 4, 0., 4.);
  h_trkProv_[1] = iBooker.book1D("assTrkProv", " Track pair provenance ", 4, 0., 4.);
  //
  h_trkAlgo_ =
      iBooker.book1D("allTrackAlgo", " Track Algo ", reco::TrackBase::algoSize, -0.5, reco::TrackBase::algoSize - 0.5);
  h_convAlgo_ = iBooker.book1D("allConvAlgo", " Conv Algo ", 5, -0.5, 4.5);
  h_convQuality_ = iBooker.book1D("allConvQuality", "Conv quality ", 11, -0.5, 11.);

  // histos for fake rate
  histname = "h_RecoConvTwoTracksEta";
  h_RecoConvTwoTracks_[0] =
      iBooker.book1D(histname, " All reco conversions with 2 reco  tracks: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_RecoConvTwoTracksPhi";
  h_RecoConvTwoTracks_[1] =
      iBooker.book1D(histname, " All reco conversions with 2 reco tracks: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_RecoConvTwoTracksR";
  h_RecoConvTwoTracks_[2] =
      iBooker.book1D(histname, " All reco conversions with 2 reco tracks: simulated R", rBin, rMin, rMax);
  histname = "h_RecoConvTwoTracksZ";
  h_RecoConvTwoTracks_[3] =
      iBooker.book1D(histname, " All reco conversions with 2 reco tracks: simulated Z", zBin, zMin, zMax);
  histname = "h_RecoConvTwoTracksEt";
  h_RecoConvTwoTracks_[4] =
      iBooker.book1D(histname, " All reco conversions with 2 reco tracks: simulated Et", etBin, etMin, etMax);
  //
  histname = "h_RecoConvTwoMTracksEta";
  h_RecoConvTwoMTracks_[0] =
      iBooker.book1D(histname, " All reco conversions with 2 reco-ass tracks: simulated #eta", etaBin2, etaMin, etaMax);
  histname = "h_RecoConvTwoMTracksPhi";
  h_RecoConvTwoMTracks_[1] =
      iBooker.book1D(histname, " All reco conversions with 2 reco-ass tracks: simulated #phi", phiBin, phiMin, phiMax);
  histname = "h_RecoConvTwoMTracksR";
  h_RecoConvTwoMTracks_[2] =
      iBooker.book1D(histname, " All reco conversions with 2 reco-ass tracks: simulated R", rBin, rMin, rMax);
  histname = "h_RecoConvTwoMTracksZ";
  h_RecoConvTwoMTracks_[3] =
      iBooker.book1D(histname, " All reco conversions with 2 reco-ass tracks: simulated Z", zBin, zMin, zMax);
  histname = "h_RecoConvTwoMTracksEt";
  h_RecoConvTwoMTracks_[4] =
      iBooker.book1D(histname, " All reco conversions with 2 reco-ass tracks: simulated Et", etBin, etMin, etMax);
}

void PhotonValidator::dqmBeginRun(edm::Run const& r, edm::EventSetup const& theEventSetup) {
  //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << " get magnetic field"
                                          << "\n";
  theMF_ = theEventSetup.getHandle(magneticFieldToken_);

  thePhotonMCTruthFinder_ = std::make_unique<PhotonMCTruthFinder>();
}

void PhotonValidator::dqmEndRun(edm::Run const& r, edm::EventSetup const&) { thePhotonMCTruthFinder_.reset(); }

void PhotonValidator::analyze(const edm::Event& e, const edm::EventSetup& esup) {
  thePhotonMCTruthFinder_->clear();
  using namespace edm;
  //  const float etaPhiDistance=0.01;
  // Fiducial region
  // const float TRK_BARL =0.9;
  const float BARL = 1.4442;  // DAQ TDR p.290
  //  const float END_LO = 1.566; // unused
  const float END_HI = 2.5;
  // Electron mass
  //const Float_t mElec= 0.000511;

  edm::Handle<reco::TrackToTrackingParticleAssociator> theHitsAssociator;
  e.getByLabel("trackAssociatorByHitsForPhotonValidation", theHitsAssociator);
  reco::TrackToTrackingParticleAssociator const* trackAssociator = theHitsAssociator.product();

  nEvt_++;
  LogInfo("PhotonValidator") << "PhotonValidator Analyzing event number: " << e.id() << " Global Counter " << nEvt_
                             << "\n";

  // get the geometry from the event setup:
  theCaloGeom_ = esup.getHandle(caloGeometryToken_);

  edm::Handle<reco::VertexCollection> vtxH;
  e.getByToken(offline_pvToken_, vtxH);
  h_nRecoVtx_->Fill(float(vtxH->size()));

  // Transform Track into TransientTrack (needed by the Vertex fitter)
  auto theTTB = esup.getHandle(transientTrackBuilderToken_);

  ///// Get the recontructed  photons
  Handle<reco::PhotonCollection> photonHandle;
  e.getByToken(photonCollectionToken_, photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
  if (!photonHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the Photon collection " << std::endl;
    return;
  }

  // Get the  PF refined cluster  collection
  Handle<reco::PFCandidateCollection> pfCandidateHandle;
  e.getByToken(pfCandidates_, pfCandidateHandle);
  if (!pfCandidateHandle.isValid()) {
    edm::LogError("PhotonValidator") << "Error! Can't get the product pfCandidates " << std::endl;
  }

  edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef> > > phoToParticleBasedIsoMapHandle;
  edm::ValueMap<std::vector<reco::PFCandidateRef> > phoToParticleBasedIsoMap;
  if (fName_ == "pfPhotonValidator") {
    e.getByToken(particleBasedIso_token, phoToParticleBasedIsoMapHandle);
    if (!phoToParticleBasedIsoMapHandle.isValid()) {
      edm::LogInfo("PhotonValidator") << "Error! Can't get the product: valueMap photons to particle based iso "
                                      << std::endl;
    }
    phoToParticleBasedIsoMap = *(phoToParticleBasedIsoMapHandle.product());
  }

  Handle<edm::View<reco::Track> > outInTrkHandle;
  Handle<edm::View<reco::Track> > inOutTrkHandle;
  if (!fastSim_) {
    //// Get the Out In CKF tracks from conversions
    e.getByToken(conversionOITrackPr_Token_, outInTrkHandle);
    //// Get the In Out  CKF tracks from conversions
    e.getByToken(conversionIOTrackPr_Token_, inOutTrkHandle);

  }  // if !fastSim

  //////////////////// Get the MC truth
  //get simtrack info
  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertices;
  edm::Handle<SimTrackContainer> SimTk;
  edm::Handle<SimVertexContainer> SimVtx;

  if (!fastSim_) {
    e.getByToken(g4_simTk_Token_, SimTk);
    e.getByToken(g4_simVtx_Token_, SimVtx);
  } else {
    e.getByToken(famos_simTk_Token_, SimTk);
    e.getByToken(famos_simVtx_Token_, SimVtx);
  }

  theSimTracks.insert(theSimTracks.end(), SimTk->begin(), SimTk->end());
  theSimVertices.insert(theSimVertices.end(), SimVtx->begin(), SimVtx->end());
  std::vector<PhotonMCTruth> mcPhotons = thePhotonMCTruthFinder_->find(theSimTracks, theSimVertices);

  edm::Handle<edm::HepMCProduct> hepMC;
  e.getByToken(hepMC_Token_, hepMC);
  const HepMC::GenEvent* myGenEvent = hepMC->GetEvent();

  Handle<reco::GenParticleCollection> genParticles;
  e.getByToken(genpartToken_, genParticles);

  // get generated jets
  Handle<reco::GenJetCollection> GenJetsHandle;
  e.getByToken(genjets_Token_, GenJetsHandle);
  reco::GenJetCollection genJetCollection = *(GenJetsHandle.product());

  // Get electron tracking truth
  bool useTP = parameters_.getParameter<bool>("useTP");
  TrackingParticleCollection trackingParticles;
  edm::Handle<TrackingParticleCollection> ElectronTPHandle;
  if (useTP) {
    if (!fastSim_) {
      e.getByToken(token_tp_, ElectronTPHandle);
      trackingParticles = *(ElectronTPHandle.product());
    }
  }

  //// Track association with TrackingParticles
  std::vector<reco::PhotonCollection::const_iterator> StoRMatchedConvertedPhotons;
  reco::SimToRecoCollection OISimToReco;
  reco::SimToRecoCollection IOSimToReco;
  // Reco to Sim
  reco::RecoToSimCollection OIRecoToSim;
  reco::RecoToSimCollection IORecoToSim;

  if (useTP) {
    if (!fastSim_) {
      // Sim to Reco
      OISimToReco = trackAssociator->associateSimToReco(outInTrkHandle, ElectronTPHandle);
      IOSimToReco = trackAssociator->associateSimToReco(inOutTrkHandle, ElectronTPHandle);
      // Reco to Sim
      OIRecoToSim = trackAssociator->associateRecoToSim(outInTrkHandle, ElectronTPHandle);
      IORecoToSim = trackAssociator->associateRecoToSim(inOutTrkHandle, ElectronTPHandle);
    }
  }
  //
  vector<reco::SimToRecoCollection*> StoRCollPtrs;
  StoRCollPtrs.push_back(&OISimToReco);
  StoRCollPtrs.push_back(&IOSimToReco);
  vector<reco::RecoToSimCollection*> RtoSCollPtrs;
  RtoSCollPtrs.push_back(&OIRecoToSim);
  RtoSCollPtrs.push_back(&IORecoToSim);
  //
  for (int i = 0; i < 2; i++)
    nSimPho_[i] = 0;
  for (int i = 0; i < 2; i++)
    nSimConv_[i] = 0;

  std::vector<reco::PhotonRef> myPhotons;

  for (unsigned int iPho = 0; iPho < photonHandle->size(); iPho++) {
    reco::PhotonRef phoRef(reco::PhotonRef(photonHandle, iPho));
    //  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
    if (fabs(phoRef->eta()) > 2.5)
      continue;
    myPhotons.push_back(phoRef);
  }

  std::sort(myPhotons.begin(), myPhotons.end(), sortPhotons());
  // if ( ! isRunCentrally_ ) {
  if (myPhotons.size() >= 2) {
    if (myPhotons[0]->et() > 40 && myPhotons[1]->et() > 25) {
      math::XYZTLorentzVector p12 = myPhotons[0]->p4() + myPhotons[1]->p4();
      math::XYZTLorentzVector p12_regr1 =
          myPhotons[0]->p4(reco::Photon::regression1) + myPhotons[1]->p4(reco::Photon::regression1);
      math::XYZTLorentzVector p12_regr2 =
          myPhotons[0]->p4(reco::Photon::regression2) + myPhotons[1]->p4(reco::Photon::regression2);
      float gamgamMass2 = p12.Dot(p12);
      float gamgamMass2_regr1 = p12_regr1.Dot(p12_regr1);
      float gamgamMass2_regr2 = p12_regr2.Dot(p12_regr2);

      //// standard ecal energy corrections
      if (gamgamMass2 > 0) {
        // total
        h_gamgamMass_[0][0]->Fill(sqrt(gamgamMass2));
        if (myPhotons[0]->isEB() && myPhotons[1]->isEB())
          h_gamgamMass_[0][1]->Fill(sqrt(gamgamMass2));
        if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
            (myPhotons[0]->isEB() && myPhotons[1]->isEE()))
          h_gamgamMass_[0][2]->Fill(sqrt(gamgamMass2));
        // Golden photons
        if (myPhotons[0]->r9() > 0.94 && myPhotons[1]->r9() > 0.94) {
          h_gamgamMass_[1][0]->Fill(sqrt(gamgamMass2));
          if (myPhotons[0]->isEB() && myPhotons[1]->isEB())
            h_gamgamMass_[1][1]->Fill(sqrt(gamgamMass2));
          if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
              (myPhotons[0]->isEB() && myPhotons[1]->isEE()))
            h_gamgamMass_[1][2]->Fill(sqrt(gamgamMass2));
        }
        // both photons converted
        if (!myPhotons[0]->conversions().empty() && !myPhotons[1]->conversions().empty()) {
          if (myPhotons[0]->conversions()[0]->nTracks() == 2 && myPhotons[1]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[0]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[0]->conversions()[0]->conversionVertex().ndof());
            float chi2Prob2 = ChiSquaredProbability(myPhotons[1]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[1]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005 && chi2Prob2 > 0.0005) {
              h_gamgamMass_[2][0]->Fill(sqrt(gamgamMass2));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMass_[2][1]->Fill(sqrt(gamgamMass2));
              }
              if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
                  (myPhotons[0]->isEB() && myPhotons[1]->isEE())) {
                h_gamgamMass_[2][2]->Fill(sqrt(gamgamMass2));
              }
            }
          }
        } else if (!myPhotons[0]->conversions().empty() && myPhotons[1]->conversions().empty() &&
                   myPhotons[1]->r9() > 0.93) {  // one photon converted
          if (myPhotons[0]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[0]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[0]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005) {
              h_gamgamMass_[2][0]->Fill(sqrt(gamgamMass2));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMass_[2][1]->Fill(sqrt(gamgamMass2));
              }
              if (myPhotons[0]->isEE() || myPhotons[1]->isEE()) {
                h_gamgamMass_[2][2]->Fill(sqrt(gamgamMass2));
              }
            }
          }
        } else if (!myPhotons[1]->conversions().empty() && myPhotons[0]->conversions().empty() &&
                   myPhotons[0]->r9() > 0.93) {  // one photon converted
          if (myPhotons[1]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[1]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[1]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005) {
              h_gamgamMass_[2][0]->Fill(sqrt(gamgamMass2));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMass_[2][1]->Fill(sqrt(gamgamMass2));
              }
              if (myPhotons[0]->isEE() || myPhotons[1]->isEE()) {
                h_gamgamMass_[2][2]->Fill(sqrt(gamgamMass2));
              }
            }
          }
        }
      }  // gamgamMass2 > 0

      ////  energy from regression1
      if (gamgamMass2_regr1 > 0) {
        // total
        h_gamgamMassRegr1_[0][0]->Fill(sqrt(gamgamMass2_regr1));
        if (myPhotons[0]->isEB() && myPhotons[1]->isEB())
          h_gamgamMassRegr1_[0][1]->Fill(sqrt(gamgamMass2_regr1));
        if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
            (myPhotons[0]->isEB() && myPhotons[1]->isEE()))
          h_gamgamMassRegr1_[0][2]->Fill(sqrt(gamgamMass2_regr1));
        // Golden photons
        if (myPhotons[0]->r9() > 0.94 && myPhotons[1]->r9() > 0.94) {
          h_gamgamMassRegr1_[1][0]->Fill(sqrt(gamgamMass2_regr1));
          if (myPhotons[0]->isEB() && myPhotons[1]->isEB())
            h_gamgamMassRegr1_[1][1]->Fill(sqrt(gamgamMass2_regr1));
          if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
              (myPhotons[0]->isEB() && myPhotons[1]->isEE()))
            h_gamgamMassRegr1_[1][2]->Fill(sqrt(gamgamMass2_regr1));
        }

        // both photons converted
        if (!myPhotons[0]->conversions().empty() && !myPhotons[1]->conversions().empty()) {
          if (myPhotons[0]->conversions()[0]->nTracks() == 2 && myPhotons[1]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[0]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[0]->conversions()[0]->conversionVertex().ndof());
            float chi2Prob2 = ChiSquaredProbability(myPhotons[1]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[1]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005 && chi2Prob2 > 0.0005) {
              h_gamgamMassRegr1_[2][0]->Fill(sqrt(gamgamMass2_regr1));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMassRegr1_[2][1]->Fill(sqrt(gamgamMass2_regr1));
              }
              if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
                  (myPhotons[0]->isEB() && myPhotons[1]->isEE())) {
                h_gamgamMassRegr1_[2][2]->Fill(sqrt(gamgamMass2_regr1));
              }
            }
          }
        } else if (!myPhotons[0]->conversions().empty() && myPhotons[1]->conversions().empty() &&
                   myPhotons[1]->r9() > 0.93) {  // one photon converted
          if (myPhotons[0]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[0]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[0]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005) {
              h_gamgamMassRegr1_[2][0]->Fill(sqrt(gamgamMass2_regr1));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMassRegr1_[2][1]->Fill(sqrt(gamgamMass2_regr1));
              }
              if (myPhotons[0]->isEE() || myPhotons[1]->isEE()) {
                h_gamgamMassRegr1_[2][2]->Fill(sqrt(gamgamMass2_regr1));
              }
            }
          }
        } else if (!myPhotons[1]->conversions().empty() && myPhotons[0]->conversions().empty() &&
                   myPhotons[0]->r9() > 0.93) {  // one photon converted
          if (myPhotons[1]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[1]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[1]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005) {
              h_gamgamMassRegr1_[2][0]->Fill(sqrt(gamgamMass2_regr1));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMassRegr1_[2][1]->Fill(sqrt(gamgamMass2_regr1));
              }
              if (myPhotons[0]->isEE() || myPhotons[1]->isEE()) {
                h_gamgamMassRegr1_[2][2]->Fill(sqrt(gamgamMass2_regr1));
              }
            }
          }
        }
      }  // gamgamMass2_regr1 > 0

      ////  energy from regression2
      if (gamgamMass2_regr2 > 0) {
        // total
        h_gamgamMassRegr2_[0][0]->Fill(sqrt(gamgamMass2_regr2));
        if (myPhotons[0]->isEB() && myPhotons[1]->isEB())
          h_gamgamMassRegr2_[0][1]->Fill(sqrt(gamgamMass2_regr2));
        if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
            (myPhotons[0]->isEB() && myPhotons[1]->isEE()))
          h_gamgamMassRegr2_[0][2]->Fill(sqrt(gamgamMass2_regr2));
        // Golden photons
        if (myPhotons[0]->r9() > 0.94 && myPhotons[1]->r9() > 0.94) {
          h_gamgamMassRegr2_[1][0]->Fill(sqrt(gamgamMass2_regr2));
          if (myPhotons[0]->isEB() && myPhotons[1]->isEB())
            h_gamgamMassRegr2_[1][1]->Fill(sqrt(gamgamMass2_regr2));
          if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
              (myPhotons[0]->isEB() && myPhotons[1]->isEE()))
            h_gamgamMassRegr2_[1][2]->Fill(sqrt(gamgamMass2_regr2));
        }

        // both photons converted
        if (!myPhotons[0]->conversions().empty() && !myPhotons[1]->conversions().empty()) {
          if (myPhotons[0]->conversions()[0]->nTracks() == 2 && myPhotons[1]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[0]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[0]->conversions()[0]->conversionVertex().ndof());
            float chi2Prob2 = ChiSquaredProbability(myPhotons[1]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[1]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005 && chi2Prob2 > 0.0005) {
              h_gamgamMassRegr2_[2][0]->Fill(sqrt(gamgamMass2_regr2));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMassRegr2_[2][1]->Fill(sqrt(gamgamMass2_regr2));
              }
              if ((myPhotons[0]->isEE() && myPhotons[1]->isEE()) || (myPhotons[0]->isEE() && myPhotons[1]->isEB()) ||
                  (myPhotons[0]->isEB() && myPhotons[1]->isEE())) {
                h_gamgamMassRegr2_[2][2]->Fill(sqrt(gamgamMass2_regr2));
              }
            }
          }
        } else if (!myPhotons[0]->conversions().empty() && myPhotons[1]->conversions().empty() &&
                   myPhotons[1]->r9() > 0.93) {  // one photon converted
          if (myPhotons[0]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[0]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[0]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005) {
              h_gamgamMassRegr2_[2][0]->Fill(sqrt(gamgamMass2_regr2));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMassRegr2_[2][1]->Fill(sqrt(gamgamMass2_regr2));
              }
              if (myPhotons[0]->isEE() || myPhotons[1]->isEE()) {
                h_gamgamMassRegr2_[2][2]->Fill(sqrt(gamgamMass2_regr2));
              }
            }
          }
        } else if (!myPhotons[1]->conversions().empty() && myPhotons[0]->conversions().empty() &&
                   myPhotons[0]->r9() > 0.93) {  // one photon converted
          if (myPhotons[1]->conversions()[0]->nTracks() == 2) {
            float chi2Prob1 = ChiSquaredProbability(myPhotons[1]->conversions()[0]->conversionVertex().chi2(),
                                                    myPhotons[1]->conversions()[0]->conversionVertex().ndof());
            if (chi2Prob1 > 0.0005) {
              h_gamgamMassRegr2_[2][0]->Fill(sqrt(gamgamMass2_regr2));
              if (myPhotons[0]->isEB() && myPhotons[1]->isEB()) {
                h_gamgamMassRegr2_[2][1]->Fill(sqrt(gamgamMass2_regr2));
              }
              if (myPhotons[0]->isEE() || myPhotons[1]->isEE()) {
                h_gamgamMassRegr2_[2][2]->Fill(sqrt(gamgamMass2_regr2));
              }
            }
          }
        }
      }  // gamgamMass2_regr2 > 0
    }
  }
  // }

  for (std::vector<PhotonMCTruth>::const_iterator mcPho = mcPhotons.begin(); mcPho != mcPhotons.end(); mcPho++) {
    if ((*mcPho).fourMomentum().et() < minPhoEtCut_)
      continue;

    for (HepMC::GenEvent::particle_const_iterator mcIter = myGenEvent->particles_begin();
         mcIter != myGenEvent->particles_end();
         mcIter++) {
      if ((*mcIter)->pdg_id() != 22)
        continue;
      bool isTheSame = false;
      HepMC::GenParticle* mother = nullptr;
      if ((*mcIter)->production_vertex()) {
        if ((*mcIter)->production_vertex()->particles_begin(HepMC::parents) !=
            (*mcIter)->production_vertex()->particles_end(HepMC::parents))
          mother = *((*mcIter)->production_vertex()->particles_begin(HepMC::parents));
      }

      float mcPhi = (*mcPho).fourMomentum().phi();
      mcPhi_ = phiNormalization(mcPhi);
      mcEta_ = (*mcPho).fourMomentum().pseudoRapidity();
      mcEta_ = etaTransformation(mcEta_, (*mcPho).primaryVertex().z());

      mcConvR_ = (*mcPho).vertex().perp();
      mcConvX_ = (*mcPho).vertex().x();
      mcConvY_ = (*mcPho).vertex().y();
      mcConvZ_ = (*mcPho).vertex().z();
      mcConvEta_ = (*mcPho).vertex().eta();
      mcConvPhi_ = (*mcPho).vertex().phi();

      if (fabs(mcEta_) > END_HI)
        continue;

      if (mother == nullptr || (mother != nullptr && mother->pdg_id() == 22) ||
          (mother != nullptr && mother->pdg_id() == 25) || (mother != nullptr && mother->pdg_id() == 35)) {
        double dPt = fabs((*mcIter)->momentum().perp() - (*mcPho).fourMomentum().et());
        float phiMother = (*mcIter)->momentum().phi();
        double dPhi = phiNormalization(phiMother) - mcPhi_;
        double dEta = fabs((*mcIter)->momentum().eta() - (*mcPho).fourMomentum().pseudoRapidity());

        if (dEta <= 0.0001 && dPhi <= 0.0001 && dPt <= 0.0001)
          isTheSame = true;
      }
      if (!isTheSame)
        continue;

      nSimPho_[0]++;
      if (!isRunCentrally_) {
        h_SimPhoMotherEt_[0]->Fill((*mcPho).motherMomentum().et());
        h_SimPhoMotherEta_[0]->Fill((*mcPho).motherMomentum().pseudoRapidity());
      }

      h_SimPho_[0]->Fill(mcEta_);
      h_SimPho_[1]->Fill(mcPhi_);
      h_SimPho_[2]->Fill((*mcPho).fourMomentum().et());

      ////////////////////////////////// extract info about simulated conversions

      bool goodSimConversion = false;
      bool visibleConversion = false;
      bool visibleConversionsWithTwoSimTracks = false;
      if ((*mcPho).isAConversion() == 1) {
        nSimConv_[0]++;
        h_AllSimConv_[0]->Fill(mcEta_);
        h_AllSimConv_[1]->Fill(mcPhi_);
        h_AllSimConv_[2]->Fill(mcConvR_);
        h_AllSimConv_[3]->Fill(mcConvZ_);
        h_AllSimConv_[4]->Fill((*mcPho).fourMomentum().et());

        if (!isRunCentrally_) {
          if (mcConvR_ < 51)
            h_SimConvEtaPix_[0]->Fill(mcEta_);
        }

        if ((fabs(mcEta_) <= BARL && mcConvR_ < 85) ||
            (fabs(mcEta_) > BARL && fabs(mcEta_) <= END_HI && fabs((*mcPho).vertex().z()) < 210))
          visibleConversion = true;

        theConvTP_.clear();
        for (size_t i = 0; i < trackingParticles.size(); ++i) {
          TrackingParticleRef tp(ElectronTPHandle, i);
          if (fabs(tp->vx() - (*mcPho).vertex().x()) < 0.001 && fabs(tp->vy() - (*mcPho).vertex().y()) < 0.001 &&
              fabs(tp->vz() - (*mcPho).vertex().z()) < 0.001) {
            theConvTP_.push_back(tp);
          }
        }
        if (theConvTP_.size() == 2)
          visibleConversionsWithTwoSimTracks = true;
        goodSimConversion = false;

        if (visibleConversion && visibleConversionsWithTwoSimTracks)
          goodSimConversion = true;
        if (goodSimConversion) {
          nSimConv_[1]++;
          h_VisSimConv_[0]->Fill(mcEta_);
          h_VisSimConv_[1]->Fill(mcPhi_);
          h_VisSimConv_[2]->Fill(mcConvR_);
          h_VisSimConv_[3]->Fill(mcConvZ_);
          h_VisSimConv_[4]->Fill((*mcPho).fourMomentum().et());

          if (useTP) {
            if (!isRunCentrally_) {
              for (edm::RefVector<TrackingParticleCollection>::iterator iTrk = theConvTP_.begin();
                   iTrk != theConvTP_.end();
                   ++iTrk) {
                h_simTkPt_->Fill((*iTrk)->pt());
                h_simTkEta_->Fill((*iTrk)->eta());
              }
            }
          }
        }
      }  ////////////// End of info from sim conversions //////////////////////////////////////////////////

      float minDelta = 10000.;
      std::vector<reco::PhotonRef> thePhotons;
      int index = 0;
      int iMatch = -1;
      bool matched = false;

      for (unsigned int iPho = 0; iPho < photonHandle->size(); iPho++) {
        reco::PhotonRef aPho(reco::PhotonRef(photonHandle, iPho));
        thePhotons.push_back(aPho);
        float phiPho = aPho->phi();
        float etaPho = aPho->eta();
        float deltaPhi = phiPho - mcPhi_;
        float deltaEta = etaPho - mcEta_;
        if (deltaPhi > pi)
          deltaPhi -= twopi;
        if (deltaPhi < -pi)
          deltaPhi += twopi;
        deltaPhi = pow(deltaPhi, 2);
        deltaEta = pow(deltaEta, 2);
        float delta = sqrt(deltaPhi + deltaEta);
        if (delta < 0.1 && delta < minDelta) {
          minDelta = delta;
          iMatch = index;
        }
        index++;
      }  // end loop over reco photons
      if (iMatch > -1)
        matched = true;

      if (matched) {
        nSimPho_[1]++;
        if (!isRunCentrally_) {
          h_SimPhoMotherEt_[1]->Fill((*mcPho).motherMomentum().et());
          h_SimPhoMotherEta_[1]->Fill((*mcPho).motherMomentum().pseudoRapidity());
        }
        h_MatchedSimPho_[0]->Fill(mcEta_);
        h_MatchedSimPho_[1]->Fill(mcPhi_);
        h_MatchedSimPho_[2]->Fill((*mcPho).fourMomentum().et());
      }

      if (!matched)
        continue;

      bool phoIsInBarrel = false;   // full barrel
      bool phoIsInBarrel1 = false;  // |eta| <=1
      bool phoIsInBarrel2 = false;  // |eta| >1
      bool phoIsInEndcap = false;
      bool phoIsInEndcapP = false;
      bool phoIsInEndcapM = false;

      reco::PhotonRef matchingPho = thePhotons[iMatch];

      if (fabs(matchingPho->superCluster()->position().eta()) < 1.479) {
        phoIsInBarrel = true;
      } else {
        phoIsInEndcap = true;
        if (matchingPho->superCluster()->position().eta() > 0)
          phoIsInEndcapP = true;
        if (matchingPho->superCluster()->position().eta() < 0)
          phoIsInEndcapM = true;
      }
      if (fabs(matchingPho->superCluster()->position().eta()) <= 1) {
        phoIsInBarrel1 = true;
      } else if (fabs(matchingPho->superCluster()->position().eta()) > 1) {
        phoIsInBarrel2 = true;
      }

      edm::Handle<EcalRecHitCollection> ecalRecHitHandle;
      if (phoIsInBarrel) {
        // Get handle to rec hits ecal barrel
        e.getByToken(barrelEcalHits_, ecalRecHitHandle);
        if (!ecalRecHitHandle.isValid()) {
          Labels l;
          labelsForToken(barrelEcalHits_, l);
          edm::LogError("PhotonProducer") << "Error! Can't get the product " << l.module;
          return;
        }

      } else if (phoIsInEndcap) {
        // Get handle to rec hits ecal encap
        e.getByToken(endcapEcalHits_, ecalRecHitHandle);
        if (!ecalRecHitHandle.isValid()) {
          Labels l;
          labelsForToken(barrelEcalHits_, l);
          edm::LogError("PhotonProducer") << "Error! Can't get the product " << l.module;
          return;
        }
      }

      int type = 0;
      const EcalRecHitCollection ecalRecHitCollection = *(ecalRecHitHandle.product());
      float photonE = matchingPho->energy();
      float sigmaEoE = matchingPho->getCorrectedEnergyError(matchingPho->getCandidateP4type()) / matchingPho->energy();
      //float photonEt= matchingPho->energy()/cosh( matchingPho->eta()) ;
      float photonEt = matchingPho->pt();
      float photonERegr1 = matchingPho->getCorrectedEnergy(reco::Photon::regression1);
      float photonERegr2 = matchingPho->getCorrectedEnergy(reco::Photon::regression2);
      float r9 = matchingPho->r9();
      //     float full5x5_r9 = matchingPho->full5x5_r9();
      float r1 = matchingPho->r1x5();
      float r2 = matchingPho->r2x5();
      float sigmaIetaIeta = matchingPho->sigmaIetaIeta();
      //float full5x5_sieie =  matchingPho->full5x5_sigmaIetaIeta();
      float hOverE = matchingPho->hadronicOverEm();
      float newhOverE = matchingPho->hadTowOverEm();
      float ecalIso = matchingPho->ecalRecHitSumEtConeDR04();
      float hcalIso = matchingPho->hcalTowerSumEtConeDR04();
      float newhcalIso = matchingPho->hcalTowerSumEtBcConeDR04();
      float trkIso = matchingPho->trkSumPtSolidConeDR04();
      float nIsoTrk = matchingPho->nTrkSolidConeDR04();
      // PF related quantities
      float chargedHadIso = matchingPho->chargedHadronIso();
      float neutralHadIso = matchingPho->neutralHadronIso();
      float photonIso = matchingPho->photonIso();
      float etOutsideMustache = matchingPho->etOutsideMustache();
      int nClusterOutsideMustache = matchingPho->nClusterOutsideMustache();
      float pfMVA = matchingPho->pfMVA();

      std::vector<std::pair<DetId, float> >::const_iterator rhIt;
      bool atLeastOneDeadChannel = false;
      for (reco::CaloCluster_iterator bcIt = matchingPho->superCluster()->clustersBegin();
           bcIt != matchingPho->superCluster()->clustersEnd();
           ++bcIt) {
        for (rhIt = (*bcIt)->hitsAndFractions().begin(); rhIt != (*bcIt)->hitsAndFractions().end(); ++rhIt) {
          for (EcalRecHitCollection::const_iterator it = ecalRecHitCollection.begin(); it != ecalRecHitCollection.end();
               ++it) {
            if (rhIt->first == (*it).id()) {
              if ((*it).recoFlag() == 9) {
                atLeastOneDeadChannel = true;
                break;
              }
            }
          }
        }
      }

      if (atLeastOneDeadChannel) {
        h_MatchedSimPhoBadCh_[0]->Fill(mcEta_);
        h_MatchedSimPhoBadCh_[1]->Fill(mcPhi_);
        h_MatchedSimPhoBadCh_[2]->Fill((*mcPho).fourMomentum().et());
      }

      if (phoIsInBarrel)
        h_phoPixSeedSize_[0]->Fill(matchingPho->electronPixelSeeds().size());
      else
        h_phoPixSeedSize_[1]->Fill(matchingPho->electronPixelSeeds().size());

      h_scEta_[type]->Fill(matchingPho->superCluster()->eta());
      h_scPhi_[type]->Fill(matchingPho->superCluster()->phi());
      if (!isRunCentrally_) {
        h_scEtaWidth_[type]->Fill(matchingPho->superCluster()->etaWidth());
        h_scPhiWidth_[type]->Fill(matchingPho->superCluster()->phiWidth());
      }
      h_scE_[type][0]->Fill(matchingPho->superCluster()->energy());
      h_scEt_[type][0]->Fill(matchingPho->superCluster()->energy() / cosh(matchingPho->superCluster()->eta()));
      if (phoIsInEndcap)
        h_psE_->Fill(matchingPho->superCluster()->preshowerEnergy());
      //
      h_r9_[type][0]->Fill(r9);
      //
      h_r1_[type][0]->Fill(r1);
      //
      h_r2_[type][0]->Fill(r2);
      //
      h_sigmaIetaIeta_[type][0]->Fill(sigmaIetaIeta);
      //
      h_hOverE_[type][0]->Fill(hOverE);
      p_r9VsEta_[0]->Fill(mcEta_, r9);

      if (!isRunCentrally_) {
        h2_r9VsEt_[0]->Fill((*mcPho).fourMomentum().et(), r9);
        h2_r1VsEta_[0]->Fill(mcEta_, r1);
        h2_r1VsEt_[0]->Fill((*mcPho).fourMomentum().et(), r1);
        h2_r2VsEta_[0]->Fill(mcEta_, r2);
        h2_r2VsEt_[0]->Fill((*mcPho).fourMomentum().et(), r2);
        h2_sigmaIetaIetaVsEta_[0]->Fill(mcEta_, sigmaIetaIeta);
        h2_sigmaIetaIetaVsEt_[0]->Fill((*mcPho).fourMomentum().et(), sigmaIetaIeta);
        h2_hOverEVsEta_[0]->Fill(mcEta_, hOverE);
        h2_hOverEVsEt_[0]->Fill((*mcPho).fourMomentum().et(), hOverE);
      }
      p_hOverEVsEta_[0]->Fill(mcEta_, hOverE);
      p_hOverEVsEt_[0]->Fill((*mcPho).fourMomentum().et(), hOverE);
      //
      h_newhOverE_[type][0]->Fill(newhOverE);
      p_newhOverEVsEta_[0]->Fill(mcEta_, newhOverE);
      p_newhOverEVsEt_[0]->Fill((*mcPho).fourMomentum().et(), newhOverE);

      //
      h_ecalRecHitSumEtConeDR04_[type][0]->Fill(ecalIso);
      if (!isRunCentrally_) {
        h2_ecalRecHitSumEtConeDR04VsEta_[0]->Fill(mcEta_, ecalIso);
        h2_ecalRecHitSumEtConeDR04VsEt_[0]->Fill((*mcPho).fourMomentum().et(), ecalIso);
        h2_hcalTowerSumEtConeDR04VsEta_[0]->Fill(mcEta_, hcalIso);
        h2_hcalTowerSumEtConeDR04VsEt_[0]->Fill((*mcPho).fourMomentum().et(), hcalIso);
      }
      p_ecalRecHitSumEtConeDR04VsEta_[0]->Fill(mcEta_, ecalIso);
      if (!isRunCentrally_)
        p_ecalRecHitSumEtConeDR04VsEt_[0]->Fill((*mcPho).fourMomentum().et(), ecalIso);
      //
      h_hcalTowerSumEtConeDR04_[type][0]->Fill(hcalIso);
      p_hcalTowerSumEtConeDR04VsEta_[0]->Fill(mcEta_, hcalIso);
      if (!isRunCentrally_)
        p_hcalTowerSumEtConeDR04VsEt_[0]->Fill((*mcPho).fourMomentum().et(), hcalIso);
      //
      if (!isRunCentrally_)
        h_hcalTowerBcSumEtConeDR04_[type][0]->Fill(newhcalIso);
      p_hcalTowerBcSumEtConeDR04VsEta_[0]->Fill(mcEta_, newhcalIso);
      if (!isRunCentrally_)
        p_hcalTowerBcSumEtConeDR04VsEt_[0]->Fill((*mcPho).fourMomentum().et(), newhcalIso);
      //
      h_isoTrkSolidConeDR04_[type][0]->Fill(trkIso);
      h_nTrkSolidConeDR04_[type][0]->Fill(nIsoTrk);

      if (!isRunCentrally_) {
        h2_isoTrkSolidConeDR04VsEta_[0]->Fill(mcEta_, trkIso);
        h2_isoTrkSolidConeDR04VsEt_[0]->Fill((*mcPho).fourMomentum().et(), trkIso);
        h2_nTrkSolidConeDR04VsEta_[0]->Fill(mcEta_, nIsoTrk);
        h2_nTrkSolidConeDR04VsEt_[0]->Fill((*mcPho).fourMomentum().et(), nIsoTrk);
      }

      h_chHadIso_[0]->Fill(chargedHadIso);
      h_nHadIso_[0]->Fill(neutralHadIso);
      h_phoIso_[0]->Fill(photonIso);
      h_nCluOutsideMustache_[0]->Fill(float(nClusterOutsideMustache));
      h_etOutsideMustache_[0]->Fill(etOutsideMustache);
      h_pfMva_[0]->Fill(pfMVA);
      //
      h_phoEta_[type]->Fill(matchingPho->eta());
      h_phoPhi_[type]->Fill(matchingPho->phi());
      h_phoDEta_[0]->Fill(matchingPho->eta() - (*mcPho).fourMomentum().eta());
      h_phoDPhi_[0]->Fill(matchingPho->phi() - mcPhi_);
      h_phoE_[type][0]->Fill(photonE);
      h_phoEt_[type][0]->Fill(photonEt);
      h_nConv_[0][0]->Fill(float(matchingPho->conversions().size()));
      h_nConv_[1][0]->Fill(float(matchingPho->conversionsOneLeg().size()));

      //
      h_phoERes_[0][0]->Fill(photonE / (*mcPho).fourMomentum().e());
      h_phoSigmaEoE_[0][0]->Fill(sigmaEoE);
      h_phoEResRegr1_[0][0]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
      h_phoEResRegr2_[0][0]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());

      p_eResVsEta_[0]->Fill(mcEta_, photonE / (*mcPho).fourMomentum().e());
      p_sigmaEoEVsEta_[0]->Fill(mcEta_, sigmaEoE);
      p_eResVsEt_[0][0]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());

      if (!isRunCentrally_)
        h2_eResVsEta_[0]->Fill(mcEta_, photonE / (*mcPho).fourMomentum().e());
      if (!isRunCentrally_)
        h2_eResVsEt_[0][0]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
      if (!isRunCentrally_)
        h2_eResVsR9_[0]->Fill(r9, photonE / (*mcPho).fourMomentum().e());
      if (!isRunCentrally_)
        h2_sceResVsR9_[0]->Fill(r9, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
      if (!isRunCentrally_)
        p_eResVsR9_[0]->Fill(r9, photonE / (*mcPho).fourMomentum().e());
      if (!isRunCentrally_)
        p_sceResVsR9_[0]->Fill(r9, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
      //
      if ((*mcPho).isAConversion() == 0) {
        if (!isRunCentrally_) {
          h2_eResVsEta_[1]->Fill(mcEta_, photonE / (*mcPho).fourMomentum().e());
          h2_r9VsEt_[1]->Fill((*mcPho).fourMomentum().et(), r9);
          //
          h2_r1VsEta_[1]->Fill(mcEta_, r1);
          h2_r1VsEt_[1]->Fill((*mcPho).fourMomentum().et(), r1);
          //
          h2_r2VsEta_[1]->Fill(mcEta_, r2);
          h2_r2VsEt_[1]->Fill((*mcPho).fourMomentum().et(), r2);
          //
          h2_sigmaIetaIetaVsEta_[1]->Fill(mcEta_, sigmaIetaIeta);
          h2_sigmaIetaIetaVsEt_[1]->Fill((*mcPho).fourMomentum().et(), sigmaIetaIeta);
          //
          h2_hOverEVsEta_[1]->Fill(mcEta_, hOverE);
          h2_hOverEVsEt_[1]->Fill((*mcPho).fourMomentum().et(), hOverE);
        }

        if (!isRunCentrally_) {
          h2_ecalRecHitSumEtConeDR04VsEta_[1]->Fill(mcEta_, ecalIso);
          h2_hcalTowerSumEtConeDR04VsEta_[1]->Fill(mcEta_, hcalIso);
          h2_isoTrkSolidConeDR04VsEta_[1]->Fill(mcEta_, trkIso);
          h2_isoTrkSolidConeDR04VsEt_[1]->Fill((*mcPho).fourMomentum().et(), trkIso);
          h2_nTrkSolidConeDR04VsEta_[1]->Fill(mcEta_, nIsoTrk);
          h2_nTrkSolidConeDR04VsEt_[1]->Fill((*mcPho).fourMomentum().et(), nIsoTrk);
        }
        p_ecalRecHitSumEtConeDR04VsEta_[1]->Fill(mcEta_, ecalIso);
        if (!isRunCentrally_)
          p_hcalTowerSumEtConeDR04VsEta_[1]->Fill(mcEta_, hcalIso);
      }

      if (photonE / (*mcPho).fourMomentum().e() < 0.3 && photonE / (*mcPho).fourMomentum().e() > 0.1) {
      }

      if ((r9 > 0.94 && phoIsInBarrel) || (r9 > 0.95 && phoIsInEndcap)) {
        h_phoERes_[1][0]->Fill(photonE / (*mcPho).fourMomentum().e());
        h_phoSigmaEoE_[1][0]->Fill(sigmaEoE);
        h_phoEResRegr1_[1][0]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
        h_phoEResRegr2_[1][0]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
        if (!isRunCentrally_)
          h2_eResVsEt_[0][1]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
        p_eResVsEt_[0][1]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
        p_eResVsEta_[1]->Fill(mcEta_, photonE / (*mcPho).fourMomentum().e());
        p_r9VsEta_[1]->Fill(mcEta_, r9);
        p_sigmaEoEVsEta_[1]->Fill(mcEta_, sigmaEoE);

      } else if ((r9 <= 0.94 && phoIsInBarrel) || (r9 <= 0.95 && phoIsInEndcap)) {
        h_phoERes_[2][0]->Fill(photonE / (*mcPho).fourMomentum().e());
        h_phoSigmaEoE_[2][0]->Fill(sigmaEoE);
        h_phoEResRegr1_[2][0]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
        h_phoEResRegr2_[2][0]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
        p_eResVsEt_[0][2]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
        p_eResVsEta_[2]->Fill(mcEta_, photonE / (*mcPho).fourMomentum().e());
        p_r9VsEta_[2]->Fill(mcEta_, r9);
        p_sigmaEoEVsEta_[2]->Fill(mcEta_, sigmaEoE);

        if (!isRunCentrally_) {
          h2_eResVsEt_[0][2]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
          h_EtR9Less093_[0][0]->Fill(photonEt);
        }
      }

      if (phoIsInBarrel) {
        h_scE_[type][1]->Fill(matchingPho->superCluster()->energy());
        h_scEt_[type][1]->Fill(matchingPho->superCluster()->energy() / cosh(matchingPho->superCluster()->eta()));
        h_r9_[type][1]->Fill(r9);
        h_r1_[type][1]->Fill(r1);
        h_r2_[type][1]->Fill(r2);
        h_sigmaIetaIeta_[type][1]->Fill(sigmaIetaIeta);
        h_hOverE_[type][1]->Fill(hOverE);
        h_newhOverE_[type][1]->Fill(newhOverE);
        h_ecalRecHitSumEtConeDR04_[type][1]->Fill(ecalIso);
        p_ecalRecHitSumEtConeDR04VsEt_[1]->Fill((*mcPho).fourMomentum().et(), ecalIso);
        h_hcalTowerSumEtConeDR04_[type][1]->Fill(hcalIso);
        p_hcalTowerSumEtConeDR04VsEt_[1]->Fill((*mcPho).fourMomentum().et(), hcalIso);
        h_hcalTowerBcSumEtConeDR04_[type][1]->Fill(newhcalIso);
        p_hcalTowerBcSumEtConeDR04VsEt_[1]->Fill((*mcPho).fourMomentum().et(), newhcalIso);
        h_isoTrkSolidConeDR04_[type][1]->Fill(trkIso);
        h_nTrkSolidConeDR04_[type][1]->Fill(nIsoTrk);
        h_chHadIso_[1]->Fill(chargedHadIso);
        h_nHadIso_[1]->Fill(neutralHadIso);
        h_phoIso_[1]->Fill(photonIso);
        h_nCluOutsideMustache_[1]->Fill(float(nClusterOutsideMustache));
        h_etOutsideMustache_[1]->Fill(etOutsideMustache);
        h_pfMva_[1]->Fill(pfMVA);
        h_phoE_[type][1]->Fill(photonE);
        h_phoEt_[type][1]->Fill(photonEt);
        h_nConv_[type][1]->Fill(float(matchingPho->conversions().size()));
        h_nConv_[1][1]->Fill(float(matchingPho->conversionsOneLeg().size()));
        h_phoERes_[0][1]->Fill(photonE / (*mcPho).fourMomentum().e());
        h_phoSigmaEoE_[0][1]->Fill(sigmaEoE);
        h_phoEResRegr1_[0][1]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
        h_phoEResRegr2_[0][1]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
        p_eResVsR9_[1]->Fill(r9, photonE / (*mcPho).fourMomentum().e());
        p_sceResVsR9_[1]->Fill(r9, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
        if (!isRunCentrally_) {
          h2_eResVsR9_[1]->Fill(r9, photonE / (*mcPho).fourMomentum().e());
          h2_sceResVsR9_[1]->Fill(r9, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
          h2_ecalRecHitSumEtConeDR04VsEt_[1]->Fill((*mcPho).fourMomentum().et(), ecalIso);
          h2_hcalTowerSumEtConeDR04VsEt_[1]->Fill((*mcPho).fourMomentum().et(), hcalIso);
          h2_eResVsEt_[1][0]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
        }
        p_eResVsEt_[1][0]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
        p_eResVsNVtx_[1][0]->Fill(float(vtxH->size()), photonE / (*mcPho).fourMomentum().e());
        p_sigmaEoEVsEt_[1][0]->Fill((*mcPho).fourMomentum().et(), sigmaEoE);
        p_sigmaEoEVsNVtx_[1][0]->Fill(float(vtxH->size()), sigmaEoE);

        if (r9 > 0.94) {
          h_phoERes_[1][1]->Fill(photonE / (*mcPho).fourMomentum().e());
          h_phoSigmaEoE_[1][1]->Fill(sigmaEoE);
          h_phoEResRegr1_[1][1]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
          h_phoEResRegr2_[1][1]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
          if (!isRunCentrally_)
            h2_eResVsEt_[1][1]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
          p_eResVsEt_[1][1]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
          p_eResVsNVtx_[1][1]->Fill(float(vtxH->size()), photonE / (*mcPho).fourMomentum().e());
          p_sigmaEoEVsEt_[1][1]->Fill((*mcPho).fourMomentum().et(), sigmaEoE);
          p_sigmaEoEVsNVtx_[1][1]->Fill(float(vtxH->size()), sigmaEoE);
        }
        if (r9 <= 0.94) {
          h_phoERes_[2][1]->Fill(photonE / (*mcPho).fourMomentum().e());
          h_phoSigmaEoE_[2][1]->Fill(sigmaEoE);
          h_phoEResRegr1_[2][1]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
          h_phoEResRegr2_[2][1]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
          p_eResVsEt_[1][2]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
          p_eResVsNVtx_[1][2]->Fill(float(vtxH->size()), photonE / (*mcPho).fourMomentum().e());
          p_sigmaEoEVsEt_[1][2]->Fill((*mcPho).fourMomentum().et(), sigmaEoE);
          p_sigmaEoEVsNVtx_[1][2]->Fill(float(vtxH->size()), sigmaEoE);
          if (!isRunCentrally_) {
            h2_eResVsEt_[1][2]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
            h_EtR9Less093_[0][1]->Fill(photonEt);
          }
        }
      }
      if (phoIsInEndcap) {
        h_scE_[type][2]->Fill(matchingPho->superCluster()->energy());
        h_scEt_[type][2]->Fill(matchingPho->superCluster()->energy() / cosh(matchingPho->superCluster()->eta()));
        h_r9_[type][2]->Fill(r9);
        h_r1_[type][2]->Fill(r1);
        h_r2_[type][2]->Fill(r2);
        h_sigmaIetaIeta_[type][2]->Fill(sigmaIetaIeta);
        h_hOverE_[type][2]->Fill(hOverE);
        h_newhOverE_[type][2]->Fill(newhOverE);
        h_ecalRecHitSumEtConeDR04_[type][2]->Fill(ecalIso);
        p_ecalRecHitSumEtConeDR04VsEt_[2]->Fill((*mcPho).fourMomentum().et(), ecalIso);
        h_hcalTowerSumEtConeDR04_[type][2]->Fill(hcalIso);
        p_hcalTowerSumEtConeDR04VsEt_[2]->Fill((*mcPho).fourMomentum().et(), hcalIso);
        h_hcalTowerBcSumEtConeDR04_[type][2]->Fill(newhcalIso);
        p_hcalTowerBcSumEtConeDR04VsEt_[2]->Fill((*mcPho).fourMomentum().et(), newhcalIso);
        h_isoTrkSolidConeDR04_[type][2]->Fill(trkIso);
        h_nTrkSolidConeDR04_[type][2]->Fill(nIsoTrk);
        h_chHadIso_[2]->Fill(chargedHadIso);
        h_nHadIso_[2]->Fill(neutralHadIso);
        h_phoIso_[2]->Fill(photonIso);
        h_nCluOutsideMustache_[2]->Fill(float(nClusterOutsideMustache));
        h_etOutsideMustache_[2]->Fill(etOutsideMustache);
        h_pfMva_[2]->Fill(pfMVA);
        h_phoE_[type][2]->Fill(photonE);
        h_phoEt_[type][2]->Fill(photonEt);
        h_nConv_[type][2]->Fill(float(matchingPho->conversions().size()));
        h_nConv_[1][2]->Fill(float(matchingPho->conversionsOneLeg().size()));
        h_phoERes_[0][2]->Fill(photonE / (*mcPho).fourMomentum().e());
        h_phoSigmaEoE_[0][2]->Fill(sigmaEoE);
        h_phoEResRegr1_[0][2]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
        h_phoEResRegr2_[0][2]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
        p_eResVsR9_[2]->Fill(r9, photonE / (*mcPho).fourMomentum().e());
        p_sceResVsR9_[2]->Fill(r9, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
        if (!isRunCentrally_) {
          h2_eResVsR9_[2]->Fill(r9, photonE / (*mcPho).fourMomentum().e());
          h2_sceResVsR9_[2]->Fill(r9, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
          h2_ecalRecHitSumEtConeDR04VsEt_[2]->Fill((*mcPho).fourMomentum().et(), ecalIso);
          h2_hcalTowerSumEtConeDR04VsEt_[2]->Fill((*mcPho).fourMomentum().et(), hcalIso);
          h2_eResVsEt_[2][0]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
        }

        p_eResVsEt_[2][0]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
        p_eResVsNVtx_[2][0]->Fill(float(vtxH->size()), photonE / (*mcPho).fourMomentum().e());
        p_sigmaEoEVsEt_[2][0]->Fill((*mcPho).fourMomentum().et(), sigmaEoE);
        p_sigmaEoEVsNVtx_[2][0]->Fill(float(vtxH->size()), sigmaEoE);

        if (r9 > 0.95) {
          h_phoERes_[1][2]->Fill(photonE / (*mcPho).fourMomentum().e());
          h_phoSigmaEoE_[1][2]->Fill(sigmaEoE);
          h_phoEResRegr1_[1][2]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
          h_phoEResRegr2_[1][2]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
          if (!isRunCentrally_)
            h2_eResVsEt_[2][1]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
          p_eResVsEt_[2][1]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
          p_eResVsNVtx_[2][1]->Fill(float(vtxH->size()), photonE / (*mcPho).fourMomentum().e());
          p_sigmaEoEVsEt_[2][1]->Fill((*mcPho).fourMomentum().et(), sigmaEoE);
          p_sigmaEoEVsNVtx_[2][1]->Fill(float(vtxH->size()), sigmaEoE);
        }
        if (r9 <= 0.95) {
          h_phoERes_[2][2]->Fill(photonE / (*mcPho).fourMomentum().e());
          h_phoSigmaEoE_[2][2]->Fill(sigmaEoE);
          h_phoEResRegr1_[2][2]->Fill(photonERegr1 / (*mcPho).fourMomentum().e());
          h_phoEResRegr2_[2][2]->Fill(photonERegr2 / (*mcPho).fourMomentum().e());
          p_eResVsEt_[2][2]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
          p_eResVsNVtx_[2][2]->Fill(float(vtxH->size()), photonE / (*mcPho).fourMomentum().e());
          p_sigmaEoEVsEt_[2][2]->Fill((*mcPho).fourMomentum().et(), sigmaEoE);
          p_sigmaEoEVsNVtx_[2][2]->Fill(float(vtxH->size()), sigmaEoE);

          if (!isRunCentrally_) {
            h2_eResVsEt_[2][2]->Fill((*mcPho).fourMomentum().et(), photonE / (*mcPho).fourMomentum().e());
            h_EtR9Less093_[0][2]->Fill(photonEt);
          }
        }
      }

      ///////////////////////   Particle based isolation
      if (fName_ == "pfPhotonValidator") {
        float SumPtIsoValCh = 0.;
        float SumPtIsoValNh = 0.;
        float SumPtIsoValPh = 0.;

        float SumPtIsoValCleanCh = 0.;
        float SumPtIsoValCleanNh = 0.;
        float SumPtIsoValCleanPh = 0.;

        for (unsigned int lCand = 0; lCand < pfCandidateHandle->size(); lCand++) {
          reco::PFCandidateRef pfCandRef(reco::PFCandidateRef(pfCandidateHandle, lCand));
          float dR = deltaR(matchingPho->eta(), matchingPho->phi(), pfCandRef->eta(), pfCandRef->phi());
          if (dR < 0.4) {
            /// uncleaned
            reco::PFCandidate::ParticleType type = pfCandRef->particleId();
            if (type == reco::PFCandidate::e)
              continue;
            if (type == reco::PFCandidate::gamma && pfCandRef->mva_nothing_gamma() > 0.)
              continue;

            if (type == reco::PFCandidate::h) {
              SumPtIsoValCh += pfCandRef->pt();
              h_dRPhoPFcand_ChHad_unCleaned_[0]->Fill(dR);
              if (phoIsInBarrel)
                h_dRPhoPFcand_ChHad_unCleaned_[1]->Fill(dR);
              else
                h_dRPhoPFcand_ChHad_unCleaned_[2]->Fill(dR);
            }
            if (type == reco::PFCandidate::h0) {
              SumPtIsoValNh += pfCandRef->pt();
              h_dRPhoPFcand_NeuHad_unCleaned_[0]->Fill(dR);
              if (phoIsInBarrel) {
                h_dRPhoPFcand_NeuHad_unCleaned_[1]->Fill(dR);
                if (phoIsInBarrel1) {
                  h_dRPhoPFcand_NeuHad_unCleaned_[3]->Fill(dR);
                }
                if (phoIsInBarrel2) {
                  h_dRPhoPFcand_NeuHad_unCleaned_[4]->Fill(dR);
                }
              } else {
                h_dRPhoPFcand_NeuHad_Cleaned_[2]->Fill(dR);
              }
            }

            if (type == reco::PFCandidate::gamma) {
              SumPtIsoValPh += pfCandRef->pt();
              h_dRPhoPFcand_Pho_unCleaned_[0]->Fill(dR);
              if (phoIsInBarrel)
                h_dRPhoPFcand_Pho_unCleaned_[1]->Fill(dR);
              else
                h_dRPhoPFcand_Pho_unCleaned_[2]->Fill(dR);
            }

            ////////// acces the value map to access the PFCandidates in overlap with the photon which need to be excluded from the isolation
            bool skip = false;
            for (std::vector<reco::PFCandidateRef>::const_iterator i = phoToParticleBasedIsoMap[matchingPho].begin();
                 i != phoToParticleBasedIsoMap[matchingPho].end();
                 ++i) {
              if ((*i) == pfCandRef) {
                skip = true;
              }
            }  // loop over the PFCandidates flagged as overlapping with the photon

            if (skip)
              continue;
            if (type == reco::PFCandidate::h) {
              SumPtIsoValCleanCh += pfCandRef->pt();
              h_dRPhoPFcand_ChHad_Cleaned_[0]->Fill(dR);
              if (phoIsInBarrel)
                h_dRPhoPFcand_ChHad_Cleaned_[1]->Fill(dR);
              else
                h_dRPhoPFcand_ChHad_Cleaned_[2]->Fill(dR);
            }
            if (type == reco::PFCandidate::h0) {
              SumPtIsoValCleanNh += pfCandRef->pt();
              h_dRPhoPFcand_NeuHad_Cleaned_[0]->Fill(dR);
              if (phoIsInBarrel) {
                h_dRPhoPFcand_NeuHad_Cleaned_[1]->Fill(dR);
                if (phoIsInBarrel1) {
                  h_dRPhoPFcand_NeuHad_Cleaned_[3]->Fill(dR);
                }
                if (phoIsInBarrel2) {
                  h_dRPhoPFcand_NeuHad_Cleaned_[4]->Fill(dR);
                }
              } else {
                h_dRPhoPFcand_NeuHad_Cleaned_[2]->Fill(dR);
              }
            }
            if (type == reco::PFCandidate::gamma) {
              SumPtIsoValCleanPh += pfCandRef->pt();
              h_dRPhoPFcand_Pho_Cleaned_[0]->Fill(dR);
              if (phoIsInBarrel)
                h_dRPhoPFcand_Pho_Cleaned_[1]->Fill(dR);
              else
                h_dRPhoPFcand_Pho_Cleaned_[2]->Fill(dR);
            }

          }  // dr=0.4
        }    // loop over all PF Candidates

        h_SumPtOverPhoPt_ChHad_Cleaned_[0]->Fill(SumPtIsoValCleanCh / matchingPho->pt());
        h_SumPtOverPhoPt_NeuHad_Cleaned_[0]->Fill(SumPtIsoValCleanNh / matchingPho->pt());
        h_SumPtOverPhoPt_Pho_Cleaned_[0]->Fill(SumPtIsoValCleanPh / matchingPho->pt());
        h_SumPtOverPhoPt_ChHad_unCleaned_[0]->Fill(SumPtIsoValCh / matchingPho->pt());
        h_SumPtOverPhoPt_NeuHad_unCleaned_[0]->Fill(SumPtIsoValNh / matchingPho->pt());
        h_SumPtOverPhoPt_Pho_unCleaned_[0]->Fill(SumPtIsoValPh / matchingPho->pt());
        if (phoIsInBarrel) {
          h_SumPtOverPhoPt_ChHad_Cleaned_[1]->Fill(SumPtIsoValCleanCh / matchingPho->pt());
          h_SumPtOverPhoPt_NeuHad_Cleaned_[1]->Fill(SumPtIsoValCleanNh / matchingPho->pt());
          h_SumPtOverPhoPt_Pho_Cleaned_[1]->Fill(SumPtIsoValCleanPh / matchingPho->pt());
          h_SumPtOverPhoPt_ChHad_unCleaned_[1]->Fill(SumPtIsoValCh / matchingPho->pt());
          h_SumPtOverPhoPt_NeuHad_unCleaned_[1]->Fill(SumPtIsoValNh / matchingPho->pt());
          h_SumPtOverPhoPt_Pho_unCleaned_[1]->Fill(SumPtIsoValPh / matchingPho->pt());
        } else {
          h_SumPtOverPhoPt_ChHad_Cleaned_[2]->Fill(SumPtIsoValCleanCh / matchingPho->pt());
          h_SumPtOverPhoPt_NeuHad_Cleaned_[2]->Fill(SumPtIsoValCleanNh / matchingPho->pt());
          h_SumPtOverPhoPt_Pho_Cleaned_[2]->Fill(SumPtIsoValCleanPh / matchingPho->pt());
          h_SumPtOverPhoPt_ChHad_unCleaned_[2]->Fill(SumPtIsoValCh / matchingPho->pt());
          h_SumPtOverPhoPt_NeuHad_unCleaned_[2]->Fill(SumPtIsoValNh / matchingPho->pt());
          h_SumPtOverPhoPt_Pho_unCleaned_[2]->Fill(SumPtIsoValPh / matchingPho->pt());
        }

      }  // only for pfPhotonValidator

      if (!(visibleConversion && visibleConversionsWithTwoSimTracks))
        continue;

      if (!isRunCentrally_) {
        h_r9_[1][0]->Fill(r9);
        if (phoIsInBarrel)
          h_r9_[1][1]->Fill(r9);
        if (phoIsInEndcap)
          h_r9_[1][2]->Fill(r9);

        h_simConvVtxRvsZ_[0]->Fill(fabs(mcConvZ_), mcConvR_);
        if (fabs(mcEta_) <= 1.) {
          h_simConvVtxRvsZ_[1]->Fill(fabs(mcConvZ_), mcConvR_);
          h_simConvVtxYvsX_->Fill(mcConvX_, mcConvY_);
        } else
          h_simConvVtxRvsZ_[2]->Fill(fabs(mcConvZ_), mcConvR_);
      }

      if (!fastSim_) {
        ////////////////// plot quantities related to conversions
        reco::ConversionRefVector conversions = matchingPho->conversions();
        bool atLeastOneRecoTwoTrackConversion = false;
        for (unsigned int iConv = 0; iConv < conversions.size(); iConv++) {
          reco::ConversionRef aConv = conversions[iConv];
          double like = aConv->MVAout();
          if (like < likelihoodCut_)
            continue;

          if (!isRunCentrally_)
            h2_EoverEtrueVsEta_[1]->Fill(mcEta_, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
          p_EoverEtrueVsEta_[1]->Fill(mcEta_, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());

          //std::vector<reco::TrackRef> tracks = aConv->tracks();
          const std::vector<edm::RefToBase<reco::Track> > tracks = aConv->tracks();
          if (tracks.size() < 2)
            continue;
          atLeastOneRecoTwoTrackConversion = true;

          h_mvaOut_[0]->Fill(like);

          if (tracks.size() == 2) {
            if (sqrt(aConv->tracksPin()[0].Perp2()) < convTrackMinPtCut_ ||
                sqrt(aConv->tracksPin()[1].Perp2()) < convTrackMinPtCut_)
              continue;
          }

          if (dCotCutOn_) {
            if ((fabs(mcEta_) > 1.1 && fabs(mcEta_) < 1.4) && fabs(aConv->pairCotThetaSeparation()) > dCotHardCutValue_)
              continue;
            if (fabs(aConv->pairCotThetaSeparation()) > dCotCutValue_)
              continue;
          }

          nRecConv_++;

          std::map<const reco::Track*, TrackingParticleRef> myAss;
          std::map<const reco::Track*, TrackingParticleRef>::const_iterator itAss;
          std::map<reco::TrackRef, TrackingParticleRef>::const_iterator itAssMin;
          std::map<reco::TrackRef, TrackingParticleRef>::const_iterator itAssMax;
          //

          int nAssT2 = 0;
          for (unsigned int i = 0; i < tracks.size(); i++) {
            //	    reco::TrackRef track = tracks[i].castTo<reco::TrackRef>();

            type = 0;
            if (!isRunCentrally_)
              nHitsVsEta_[type]->Fill(mcEta_, float(tracks[i]->numberOfValidHits()) - 0.0001);
            if (!isRunCentrally_)
              nHitsVsR_[type]->Fill(mcConvR_, float(tracks[i]->numberOfValidHits()) - 0.0001);
            p_nHitsVsEta_[type]->Fill(mcEta_, float(tracks[i]->numberOfValidHits() - 0.0001));
            p_nHitsVsR_[type]->Fill(mcConvR_, float(tracks[i]->numberOfValidHits() - 0.0001));
            h_tkChi2_[type]->Fill(tracks[i]->normalizedChi2());

            const RefToBase<reco::Track>& tfrb = tracks[i];
            RefToBaseVector<reco::Track> tc;
            tc.push_back(tfrb);
            // reco::RecoToSimCollection q = trackAssociator->associateRecoToSim(tc,theConvTP_);
            reco::SimToRecoCollection q = trackAssociator->associateSimToReco(tc, theConvTP_);
            std::vector<std::pair<RefToBase<reco::Track>, double> > trackV;
            int tpI = 0;

            if (q.find(theConvTP_[0]) != q.end()) {
              trackV = (std::vector<std::pair<RefToBase<reco::Track>, double> >)q[theConvTP_[0]];
            } else if (q.find(theConvTP_[1]) != q.end()) {
              trackV = (std::vector<std::pair<RefToBase<reco::Track>, double> >)q[theConvTP_[1]];
              tpI = 1;
            }

            if (trackV.empty())
              continue;
            edm::RefToBase<reco::Track> tr = trackV.front().first;
            myAss.insert(std::make_pair(tr.get(), theConvTP_[tpI]));
            nAssT2++;
          }

          type = 0;

          //	  float totP = sqrt(aConv->pairMomentum().Mag2());
          float refP = -99999.;
          float refPt = -99999.;
          if (aConv->conversionVertex().isValid()) {
            refP = sqrt(aConv->refittedPairMomentum().Mag2());
            refPt = sqrt(aConv->refittedPairMomentum().perp2());
          }
          float invM = aConv->pairInvariantMass();

          h_invMass_[type][0]->Fill(invM);
          if (phoIsInBarrel)
            h_invMass_[type][1]->Fill(invM);
          if (phoIsInEndcap)
            h_invMass_[type][2]->Fill(invM);

          ////////// Numerators for conversion absolute efficiency
          if (tracks.size() == 2) {
            h_SimConvTwoTracks_[0]->Fill(mcEta_);
            h_SimConvTwoTracks_[1]->Fill(mcPhi_);
            h_SimConvTwoTracks_[2]->Fill(mcConvR_);
            h_SimConvTwoTracks_[3]->Fill(mcConvZ_);
            h_SimConvTwoTracks_[4]->Fill((*mcPho).fourMomentum().et());

            if (!aConv->caloCluster().empty())
              h_convEta_[1]->Fill(aConv->caloCluster()[0]->eta());

            float trkProvenance = 3;
            if (tracks[0]->algoName() == "outInEcalSeededConv" && tracks[1]->algoName() == "outInEcalSeededConv")
              trkProvenance = 0;
            if (tracks[0]->algoName() == "inOutEcalSeededConv" && tracks[1]->algoName() == "inOutEcalSeededConv")
              trkProvenance = 1;
            if ((tracks[0]->algoName() == "outInEcalSeededConv" && tracks[1]->algoName() == "inOutEcalSeededConv") ||
                (tracks[1]->algoName() == "outInEcalSeededConv" && tracks[0]->algoName() == "inOutEcalSeededConv"))
              trkProvenance = 2;
            if (trkProvenance == 3) {
            }
            h_trkProv_[0]->Fill(trkProvenance);
            h_trkAlgo_->Fill(tracks[0]->algo());
            h_trkAlgo_->Fill(tracks[1]->algo());
            h_convAlgo_->Fill(aConv->algo());

            ////////// Numerators for conversion efficiencies: both tracks are associated
            if (nAssT2 == 2) {
              if (!isRunCentrally_) {
                h_r9_[2][0]->Fill(r9);
                if (phoIsInBarrel)
                  h_r9_[2][1]->Fill(r9);
                if (phoIsInEndcap)
                  h_r9_[2][2]->Fill(r9);
              }

              if (!aConv->caloCluster().empty())
                h_convEta_[2]->Fill(aConv->caloCluster()[0]->eta());
              nRecConvAss_++;

              h_SimConvTwoMTracks_[0]->Fill(mcEta_);
              h_SimConvTwoMTracks_[1]->Fill(mcPhi_);
              h_SimConvTwoMTracks_[2]->Fill(mcConvR_);
              h_SimConvTwoMTracks_[3]->Fill(mcConvZ_);
              h_SimConvTwoMTracks_[4]->Fill((*mcPho).fourMomentum().et());

              if (aConv->conversionVertex().isValid()) {
                float chi2Prob =
                    ChiSquaredProbability(aConv->conversionVertex().chi2(), aConv->conversionVertex().ndof());
                if (chi2Prob > 0) {
                  h_SimConvTwoMTracksAndVtxPGT0_[0]->Fill(mcEta_);
                  h_SimConvTwoMTracksAndVtxPGT0_[1]->Fill(mcPhi_);
                  h_SimConvTwoMTracksAndVtxPGT0_[2]->Fill(mcConvR_);
                  h_SimConvTwoMTracksAndVtxPGT0_[3]->Fill(mcConvZ_);
                  h_SimConvTwoMTracksAndVtxPGT0_[4]->Fill((*mcPho).fourMomentum().et());
                }
                if (chi2Prob > 0.0005) {
                  h_SimConvTwoMTracksAndVtxPGT0005_[0]->Fill(mcEta_);
                  h_SimConvTwoMTracksAndVtxPGT0005_[1]->Fill(mcPhi_);
                  h_SimConvTwoMTracksAndVtxPGT0005_[2]->Fill(mcConvR_);
                  h_SimConvTwoMTracksAndVtxPGT0005_[3]->Fill(mcConvZ_);
                  h_SimConvTwoMTracksAndVtxPGT0005_[4]->Fill((*mcPho).fourMomentum().et());
                }

                if (chi2Prob > 0.0005) {
                  if (!aConv->caloCluster().empty()) {
                    h_convEta_[0]->Fill(aConv->caloCluster()[0]->eta());
                    h_convPhi_[0]->Fill(aConv->caloCluster()[0]->phi());
                    h_convERes_[0][0]->Fill(aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e());
                  }
                  if (!isRunCentrally_) {
                    h_r9VsNofTracks_[0][0]->Fill(r9, aConv->nTracks());
                    h_EtR9Less093_[1][0]->Fill(photonEt);
                    if (phoIsInBarrel)
                      h_EtR9Less093_[1][1]->Fill(photonEt);
                    if (phoIsInEndcap)
                      h_EtR9Less093_[1][2]->Fill(photonEt);
                  }
                  if (phoIsInBarrel) {
                    if (!aConv->caloCluster().empty())
                      h_convERes_[0][1]->Fill(aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e());
                    if (!isRunCentrally_)
                      h_r9VsNofTracks_[0][1]->Fill(r9, aConv->nTracks());
                    h_mvaOut_[1]->Fill(like);
                  }
                  if (phoIsInEndcap) {
                    if (!aConv->caloCluster().empty())
                      h_convERes_[0][2]->Fill(aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e());
                    if (!isRunCentrally_)
                      h_r9VsNofTracks_[0][2]->Fill(r9, aConv->nTracks());
                    h_mvaOut_[2]->Fill(like);
                  }
                }
              }

              ///////////  Quantities per conversion
              type = 1;

              h_trkProv_[1]->Fill(trkProvenance);
              h_invMass_[type][0]->Fill(invM);

              float eoverp = -99999.;

              if (aConv->conversionVertex().isValid()) {
                eoverp = photonE / sqrt(aConv->refittedPairMomentum().Mag2());
                //eoverp= aConv->EoverPrefittedTracks();
                h_convPtRes_[type][0]->Fill(refPt / (*mcPho).fourMomentum().et());
                h_EoverPTracks_[type][0]->Fill(eoverp);
                h_PoverETracks_[type][0]->Fill(1. / eoverp);
                if (!isRunCentrally_)
                  h2_EoverEtrueVsEoverP_[0]->Fill(eoverp,
                                                  matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
                if (!isRunCentrally_)
                  h2_PoverPtrueVsEoverP_[0]->Fill(eoverp, refP / (*mcPho).fourMomentum().e());
                if (!isRunCentrally_)
                  h2_EoverPVsEta_[0]->Fill(mcEta_, eoverp);
                if (!isRunCentrally_)
                  h2_EoverPVsR_[0]->Fill(mcConvR_, eoverp);
                p_EoverPVsEta_[0]->Fill(mcEta_, eoverp);
                p_EoverPVsR_[0]->Fill(mcConvR_, eoverp);
                p_eResVsR_->Fill(mcConvR_, photonE / (*mcPho).fourMomentum().e());
                if (!isRunCentrally_)
                  h2_PoverPtrueVsEta_[0]->Fill(mcEta_, refP / (*mcPho).fourMomentum().e());
                p_PoverPtrueVsEta_[0]->Fill(mcEta_, refP / (*mcPho).fourMomentum().e());
              }

              if (!isRunCentrally_)
                h2_EoverEtrueVsEta_[0]->Fill(mcEta_,
                                             matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
              if (!isRunCentrally_)
                h2_EoverEtrueVsR_[0]->Fill(mcConvR_,
                                           matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
              p_EoverEtrueVsEta_[0]->Fill(mcEta_, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
              p_EoverEtrueVsR_[0]->Fill(mcConvR_, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());

              if (!isRunCentrally_)
                h2_etaVsRsim_[0]->Fill(mcEta_, mcConvR_);

              //  here original tracks and their inner momentum is considered
              float dPhiTracksAtVtx = aConv->dPhiTracksAtVtx();
              h_DPhiTracksAtVtx_[type][0]->Fill(dPhiTracksAtVtx);
              if (!isRunCentrally_)
                h2_DPhiTracksAtVtxVsEta_->Fill(mcEta_, dPhiTracksAtVtx);
              if (!isRunCentrally_)
                h2_DPhiTracksAtVtxVsR_->Fill(mcConvR_, dPhiTracksAtVtx);
              p_DPhiTracksAtVtxVsEta_->Fill(mcEta_, dPhiTracksAtVtx);
              p_DPhiTracksAtVtxVsR_->Fill(mcConvR_, dPhiTracksAtVtx);

              h_DCotTracks_[type][0]->Fill(aConv->pairCotThetaSeparation());
              if (!isRunCentrally_)
                h2_DCotTracksVsEta_->Fill(mcEta_, aConv->pairCotThetaSeparation());
              if (!isRunCentrally_)
                h2_DCotTracksVsR_->Fill(mcConvR_, aConv->pairCotThetaSeparation());
              p_DCotTracksVsEta_->Fill(mcEta_, aConv->pairCotThetaSeparation());
              p_DCotTracksVsR_->Fill(mcConvR_, aConv->pairCotThetaSeparation());

              if (phoIsInBarrel) {
                h_invMass_[type][1]->Fill(invM);
                if (aConv->conversionVertex().isValid()) {
                  h_convPtRes_[type][1]->Fill(refPt / (*mcPho).fourMomentum().et());
                  h_EoverPTracks_[type][1]->Fill(eoverp);
                  if (mcConvR_ < 15)
                    h_EoverPTracks_[0][0]->Fill(eoverp);
                  if (mcConvR_ > 15 && mcConvR_ < 58)
                    h_EoverPTracks_[0][1]->Fill(eoverp);
                  if (mcConvR_ > 58)
                    h_EoverPTracks_[0][2]->Fill(eoverp);
                  h_PoverETracks_[type][1]->Fill(1. / eoverp);
                  if (!isRunCentrally_)
                    h2_EoverEtrueVsEoverP_[1]->Fill(
                        eoverp, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
                  if (!isRunCentrally_)
                    h2_PoverPtrueVsEoverP_[1]->Fill(eoverp, refP / (*mcPho).fourMomentum().e());
                }
                h_DPhiTracksAtVtx_[type][1]->Fill(dPhiTracksAtVtx);
                h_DCotTracks_[type][1]->Fill(aConv->pairCotThetaSeparation());
              }

              if (phoIsInEndcap) {
                h_invMass_[type][2]->Fill(invM);
                if (aConv->conversionVertex().isValid()) {
                  h_convPtRes_[type][2]->Fill(refPt / (*mcPho).fourMomentum().et());
                  h_EoverPTracks_[type][2]->Fill(eoverp);
                  h_PoverETracks_[type][2]->Fill(1. / eoverp);
                  if (!isRunCentrally_)
                    h2_EoverEtrueVsEoverP_[2]->Fill(
                        eoverp, matchingPho->superCluster()->energy() / (*mcPho).fourMomentum().e());
                  if (!isRunCentrally_)
                    h2_PoverPtrueVsEoverP_[2]->Fill(eoverp, refP / (*mcPho).fourMomentum().e());
                }
                h_DPhiTracksAtVtx_[type][2]->Fill(dPhiTracksAtVtx);
                h_DCotTracks_[type][2]->Fill(aConv->pairCotThetaSeparation());
              }

              if (aConv->conversionVertex().isValid()) {
                h_convVtxdX_->Fill(aConv->conversionVertex().position().x() - mcConvX_);
                h_convVtxdY_->Fill(aConv->conversionVertex().position().y() - mcConvY_);
                h_convVtxdZ_->Fill(aConv->conversionVertex().position().z() - mcConvZ_);
                h_convVtxdR_->Fill(sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);

                if (fabs(mcConvEta_) <= 1.2) {
                  h_convVtxdX_barrel_->Fill(aConv->conversionVertex().position().x() - mcConvX_);
                  h_convVtxdY_barrel_->Fill(aConv->conversionVertex().position().y() - mcConvY_);
                  h_convVtxdZ_barrel_->Fill(aConv->conversionVertex().position().z() - mcConvZ_);
                  h_convVtxdR_barrel_->Fill(sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);
                } else {
                  h_convVtxdX_endcap_->Fill(aConv->conversionVertex().position().x() - mcConvX_);
                  h_convVtxdY_endcap_->Fill(aConv->conversionVertex().position().y() - mcConvY_);
                  h_convVtxdZ_endcap_->Fill(aConv->conversionVertex().position().z() - mcConvZ_);
                  h_convVtxdR_endcap_->Fill(sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);
                }

                h_convVtxdPhi_->Fill(aConv->conversionVertex().position().phi() - mcConvPhi_);
                h_convVtxdEta_->Fill(aConv->conversionVertex().position().eta() - mcConvEta_);
                if (!isRunCentrally_)
                  h2_convVtxdRVsR_->Fill(mcConvR_, sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);
                if (!isRunCentrally_)
                  h2_convVtxdRVsEta_->Fill(mcEta_, sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);
                p_convVtxdRVsR_->Fill(mcConvR_, sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);
                p_convVtxdRVsEta_->Fill(mcEta_, sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);
                float signX = aConv->refittedPairMomentum().x() / fabs(aConv->refittedPairMomentum().x());
                float signY = aConv->refittedPairMomentum().y() / fabs(aConv->refittedPairMomentum().y());
                float signZ = aConv->refittedPairMomentum().z() / fabs(aConv->refittedPairMomentum().z());
                p_convVtxdXVsX_->Fill(mcConvX_, (aConv->conversionVertex().position().x() - mcConvX_) * signX);
                p_convVtxdYVsY_->Fill(mcConvY_, (aConv->conversionVertex().position().y() - mcConvY_) * signY);
                p_convVtxdZVsZ_->Fill(mcConvZ_, (aConv->conversionVertex().position().z() - mcConvZ_) * signZ);

                if (!isRunCentrally_)
                  h2_convVtxRrecVsTrue_->Fill(mcConvR_, sqrt(aConv->conversionVertex().position().perp2()));

                //float zPV = aConv->zOfPrimaryVertexFromTracks();
                float thetaConv = aConv->refittedPairMomentum().Theta();
                float thetaSC = matchingPho->superCluster()->position().theta();
                float rSC =
                    sqrt(matchingPho->superCluster()->position().x() * matchingPho->superCluster()->position().x() +
                         matchingPho->superCluster()->position().y() * matchingPho->superCluster()->position().y());
                float zSC = matchingPho->superCluster()->position().z();
                float zPV = sqrt(rSC * rSC + zSC * zSC) * sin(thetaConv - thetaSC) / sin(thetaConv);

                h_zPVFromTracks_[0]->Fill(zPV);
                h_dzPVFromTracks_[0]->Fill(zPV - (*mcPho).primaryVertex().z());

                if (phoIsInBarrel) {
                  h_zPVFromTracks_[1]->Fill(zPV);
                  h_dzPVFromTracks_[1]->Fill(zPV - (*mcPho).primaryVertex().z());
                } else if (phoIsInEndcap) {
                  h_zPVFromTracks_[2]->Fill(zPV);
                  h_dzPVFromTracks_[2]->Fill(zPV - (*mcPho).primaryVertex().z());
                } else if (phoIsInEndcapP) {
                  h_zPVFromTracks_[3]->Fill(zPV);
                  h_dzPVFromTracks_[3]->Fill(zPV - (*mcPho).primaryVertex().z());
                } else if (phoIsInEndcapM) {
                  h_zPVFromTracks_[4]->Fill(zPV);
                  h_dzPVFromTracks_[4]->Fill(zPV - (*mcPho).primaryVertex().z());
                }

                p_dzPVVsR_->Fill(mcConvR_, zPV - (*mcPho).primaryVertex().z());
                p_dzPVVsEta_->Fill(mcConvEta_, zPV - (*mcPho).primaryVertex().z());
                if (!isRunCentrally_)
                  h2_dzPVVsR_->Fill(mcConvR_, zPV - (*mcPho).primaryVertex().z());
              }

              float dPhiTracksAtEcal = -99;
              float dEtaTracksAtEcal = -99;
              if (!aConv->bcMatchingWithTracks().empty() && aConv->bcMatchingWithTracks()[0].isNonnull() &&
                  aConv->bcMatchingWithTracks()[1].isNonnull()) {
                nRecConvAssWithEcal_++;
                float recoPhi1 = aConv->ecalImpactPosition()[0].phi();
                float recoPhi2 = aConv->ecalImpactPosition()[1].phi();
                float recoEta1 = aConv->ecalImpactPosition()[0].eta();
                float recoEta2 = aConv->ecalImpactPosition()[1].eta();
                // unused   float bcPhi1 = aConv->bcMatchingWithTracks()[0]->phi();
                // unused   float bcPhi2 = aConv->bcMatchingWithTracks()[1]->phi();
                // unused   float bcEta1 = aConv->bcMatchingWithTracks()[0]->eta();
                // unused   float bcEta2 = aConv->bcMatchingWithTracks()[1]->eta();
                recoPhi1 = phiNormalization(recoPhi1);
                recoPhi2 = phiNormalization(recoPhi2);
                dPhiTracksAtEcal = recoPhi1 - recoPhi2;
                dPhiTracksAtEcal = phiNormalization(dPhiTracksAtEcal);
                dEtaTracksAtEcal = recoEta1 - recoEta2;

                h_DPhiTracksAtEcal_[type][0]->Fill(fabs(dPhiTracksAtEcal));
                if (!isRunCentrally_)
                  h2_DPhiTracksAtEcalVsR_->Fill(mcConvR_, fabs(dPhiTracksAtEcal));
                if (!isRunCentrally_)
                  h2_DPhiTracksAtEcalVsEta_->Fill(mcEta_, fabs(dPhiTracksAtEcal));
                p_DPhiTracksAtEcalVsR_->Fill(mcConvR_, fabs(dPhiTracksAtEcal));
                p_DPhiTracksAtEcalVsEta_->Fill(mcEta_, fabs(dPhiTracksAtEcal));

                h_DEtaTracksAtEcal_[type][0]->Fill(dEtaTracksAtEcal);

                if (phoIsInBarrel) {
                  h_DPhiTracksAtEcal_[type][1]->Fill(fabs(dPhiTracksAtEcal));
                  h_DEtaTracksAtEcal_[type][1]->Fill(dEtaTracksAtEcal);
                }
                if (phoIsInEndcap) {
                  h_DPhiTracksAtEcal_[type][2]->Fill(fabs(dPhiTracksAtEcal));
                  h_DEtaTracksAtEcal_[type][2]->Fill(dEtaTracksAtEcal);
                }
              }
              ///////////  Quantities per track
              for (unsigned int i = 0; i < tracks.size(); i++) {
                const RefToBase<reco::Track>& tfrb(tracks[i]);
                itAss = myAss.find(tfrb.get());
                if (itAss == myAss.end())
                  continue;

                float trkProvenance = 3;
                if (tracks[0]->algoName() == "outInEcalSeededConv" && tracks[1]->algoName() == "outInEcalSeededConv")
                  trkProvenance = 0;
                if (tracks[0]->algoName() == "inOutEcalSeededConv" && tracks[1]->algoName() == "inOutEcalSeededConv")
                  trkProvenance = 1;
                if ((tracks[0]->algoName() == "outInEcalSeededConv" &&
                     tracks[1]->algoName() == "inOutEcalSeededConv") ||
                    (tracks[1]->algoName() == "outInEcalSeededConv" && tracks[0]->algoName() == "inOutEcalSeededConv"))
                  trkProvenance = 2;

                if (!isRunCentrally_)
                  nHitsVsEta_[type]->Fill(mcEta_, float(tracks[i]->numberOfValidHits()));
                if (!isRunCentrally_)
                  nHitsVsR_[type]->Fill(mcConvR_, float(tracks[i]->numberOfValidHits()));
                p_nHitsVsEta_[type]->Fill(mcEta_, float(tracks[i]->numberOfValidHits()) - 0.0001);
                p_nHitsVsR_[type]->Fill(mcConvR_, float(tracks[i]->numberOfValidHits()) - 0.0001);
                h_tkChi2_[type]->Fill(tracks[i]->normalizedChi2());
                h_tkChi2Large_[type]->Fill(tracks[i]->normalizedChi2());
                if (!isRunCentrally_)
                  h2_Chi2VsEta_[0]->Fill(mcEta_, tracks[i]->normalizedChi2());
                if (!isRunCentrally_)
                  h2_Chi2VsR_[0]->Fill(mcConvR_, tracks[i]->normalizedChi2());
                p_Chi2VsEta_[0]->Fill(mcEta_, tracks[i]->normalizedChi2());
                p_Chi2VsR_[0]->Fill(mcConvR_, tracks[i]->normalizedChi2());

                float simPt = sqrt(((*itAss).second)->momentum().perp2());
                //		float recPt =   sqrt( aConv->tracks()[i]->innerMomentum().Perp2() ) ;
                float refPt = -9999.;
                float px = 0, py = 0;

                if (aConv->conversionVertex().isValid()) {
                  reco::Track refTrack = aConv->conversionVertex().refittedTracks()[i];
                  px = refTrack.momentum().x();
                  py = refTrack.momentum().y();
                  refPt = sqrt(px * px + py * py);

                  float ptres = refPt - simPt;
                  // float pterror = aConv->tracks()[i]->ptError();
                  float pterror = aConv->conversionVertex().refittedTracks()[i].ptError();
                  if (!isRunCentrally_) {
                    h2_PtRecVsPtSim_[0]->Fill(simPt, refPt);
                    if (trkProvenance == 3)
                      h2_PtRecVsPtSimMixProv_->Fill(simPt, refPt);
                  }

                  h_TkPtPull_[0]->Fill(ptres / pterror);
                  if (!isRunCentrally_)
                    h2_TkPtPull_[0]->Fill(mcEta_, ptres / pterror);

                  h_TkD0_[0]->Fill(tracks[i]->d0() * tracks[i]->charge());

                  // if ( fName_ != "pfPhotonValidator" &&  fName_ != "oldpfPhotonValidator" )
                  if (!aConv->bcMatchingWithTracks().empty() && aConv->bcMatchingWithTracks()[i].isNonnull())
                    hBCEnergyOverTrackPout_[0]->Fill(aConv->bcMatchingWithTracks()[i]->energy() /
                                                     sqrt(aConv->tracks()[i]->outerMomentum().Mag2()));

                  if (phoIsInBarrel) {
                    h_TkD0_[1]->Fill(tracks[i]->d0() * tracks[i]->charge());
                    h_TkPtPull_[1]->Fill(ptres / pterror);
                    if (!isRunCentrally_)
                      h2_PtRecVsPtSim_[1]->Fill(simPt, refPt);
                    //if ( fName_ != "pfPhotonValidator"  &&  fName_ != "oldpfPhotonValidator")
                    if (!aConv->bcMatchingWithTracks().empty() && aConv->bcMatchingWithTracks()[i].isNonnull())
                      hBCEnergyOverTrackPout_[1]->Fill(aConv->bcMatchingWithTracks()[i]->energy() /
                                                       sqrt(aConv->tracks()[i]->outerMomentum().Mag2()));
                  }
                  if (phoIsInEndcap) {
                    h_TkD0_[2]->Fill(tracks[i]->d0() * tracks[i]->charge());
                    h_TkPtPull_[2]->Fill(ptres / pterror);
                    if (!isRunCentrally_)
                      h2_PtRecVsPtSim_[2]->Fill(simPt, refPt);
                    //		    if ( fName_ != "pfPhotonValidator" &&  fName_ != "oldpfPhotonValidator")
                    if (!aConv->bcMatchingWithTracks().empty() && aConv->bcMatchingWithTracks()[i].isNonnull())
                      hBCEnergyOverTrackPout_[2]->Fill(aConv->bcMatchingWithTracks()[i]->energy() /
                                                       sqrt(aConv->tracks()[i]->outerMomentum().Mag2()));
                  }
                }

              }  // end loop over track
            }    // end analysis of two associated tracks
          }      // end analysis of two  tracks

        }  // loop over conversions

        //////////////////// Monitor singleLeg conversions

        reco::ConversionRefVector conversionsOneLeg = matchingPho->conversionsOneLeg();
        if (!atLeastOneRecoTwoTrackConversion) {
          for (unsigned int iConv = 0; iConv < conversionsOneLeg.size(); iConv++) {
            reco::ConversionRef aConv = conversionsOneLeg[iConv];
            const std::vector<edm::RefToBase<reco::Track> > tracks = aConv->tracks();

            h_trkAlgo_->Fill(tracks[0]->algo());
            h_convAlgo_->Fill(aConv->algo());

            int nAssT = 0;
            std::map<const reco::Track*, TrackingParticleRef> myAss;
            for (unsigned int i = 0; i < tracks.size(); i++) {
              p_nHitsVsEtaSL_[0]->Fill(mcEta_, float(tracks[0]->numberOfValidHits() - 0.0001));
              p_nHitsVsRSL_[0]->Fill(mcConvR_, float(tracks[0]->numberOfValidHits() - 0.0001));
              h_tkChi2SL_[0]->Fill(tracks[0]->normalizedChi2());

              float eoverp = photonE / tracks[0]->p();
              h_EoverP_SL_[0]->Fill(eoverp);
              if (phoIsInBarrel) {
                h_EoverP_SL_[1]->Fill(eoverp);
              } else {
                h_EoverP_SL_[2]->Fill(eoverp);
              }
              h_convSLVtxRvsZ_[0]->Fill(tracks[0]->innerPosition().z(), sqrt(tracks[0]->innerPosition().Perp2()));
              if (fabs(mcEta_) <= 1.) {
                h_convSLVtxRvsZ_[1]->Fill(tracks[0]->innerPosition().z(), sqrt(tracks[0]->innerPosition().Perp2()));
              } else {
                h_convSLVtxRvsZ_[2]->Fill(tracks[0]->innerPosition().z(), sqrt(tracks[0]->innerPosition().Perp2()));
              }

              const RefToBase<reco::Track>& tfrb = tracks[i];
              RefToBaseVector<reco::Track> tc;
              tc.push_back(tfrb);
              reco::SimToRecoCollection q = trackAssociator->associateSimToReco(tc, theConvTP_);
              std::vector<std::pair<RefToBase<reco::Track>, double> > trackV;
              int tpI = 0;

              if (q.find(theConvTP_[0]) != q.end()) {
                trackV = (std::vector<std::pair<RefToBase<reco::Track>, double> >)q[theConvTP_[0]];
              } else if (q.find(theConvTP_[1]) != q.end()) {
                trackV = (std::vector<std::pair<RefToBase<reco::Track>, double> >)q[theConvTP_[1]];
                tpI = 1;
              }

              if (trackV.empty())
                continue;
              edm::RefToBase<reco::Track> tr = trackV.front().first;
              myAss.insert(std::make_pair(tr.get(), theConvTP_[tpI]));
              nAssT++;
            }

            if (nAssT > 0) {
              h_SimConvOneMTracks_[0]->Fill(mcEta_);
              h_SimConvOneMTracks_[1]->Fill(mcPhi_);
              h_SimConvOneMTracks_[2]->Fill(mcConvR_);
              h_SimConvOneMTracks_[3]->Fill(mcConvZ_);
              h_SimConvOneMTracks_[4]->Fill((*mcPho).fourMomentum().et());
            }
          }  // End loop over single leg conversions
        }

      }  // if !fastSim
    }    // End loop over generated particles
  }      // End loop over simulated Photons

  if (!isRunCentrally_) {
    h_nSimPho_[0]->Fill(float(nSimPho_[0]));
    h_nSimPho_[1]->Fill(float(nSimPho_[1]));
    h_nSimConv_[0]->Fill(float(nSimConv_[0]));
    h_nSimConv_[1]->Fill(float(nSimConv_[1]));
  }

  if (!fastSim_) {
    ///////////////////  Measure fake rate
    for (reco::PhotonCollection::const_iterator iPho = photonCollection.begin(); iPho != photonCollection.end();
         iPho++) {
      reco::Photon aPho = reco::Photon(*iPho);
      //    float et= aPho.superCluster()->energy()/cosh( aPho.superCluster()->eta()) ;
      reco::ConversionRefVector conversions = aPho.conversions();
      for (unsigned int iConv = 0; iConv < conversions.size(); iConv++) {
        reco::ConversionRef aConv = conversions[iConv];
        double like = aConv->MVAout();
        if (like < likelihoodCut_)
          continue;
        //std::vector<reco::TrackRef> tracks = aConv->tracks();
        const std::vector<edm::RefToBase<reco::Track> > tracks = aConv->tracks();
        if (tracks.size() < 2)
          continue;

        RefToBase<reco::Track> tk1 = aConv->tracks().front();
        RefToBase<reco::Track> tk2 = aConv->tracks().back();
        RefToBaseVector<reco::Track> tc1, tc2;
        tc1.push_back(tk1);
        tc2.push_back(tk2);

        bool phoIsInBarrel = false;
        bool phoIsInEndcap = false;
        if (fabs(aPho.superCluster()->position().eta()) < 1.479) {
          phoIsInBarrel = true;
        } else {
          phoIsInEndcap = true;
        }

        if (dCotCutOn_) {
          if ((fabs(mcEta_) > 1.1 && fabs(mcEta_) < 1.4) && fabs(aConv->pairCotThetaSeparation()) > dCotHardCutValue_)
            continue;
          if (fabs(aConv->pairCotThetaSeparation()) > dCotCutValue_)
            continue;
        }

        h_RecoConvTwoTracks_[0]->Fill(aPho.eta());
        h_RecoConvTwoTracks_[1]->Fill(aPho.phi());
        if (aConv->conversionVertex().isValid())
          h_RecoConvTwoTracks_[2]->Fill(aConv->conversionVertex().position().perp2());
        h_RecoConvTwoTracks_[3]->Fill(aConv->conversionVertex().position().z());
        h_RecoConvTwoTracks_[4]->Fill(aPho.et());

        int nAssT2 = 0;
        for (std::vector<PhotonMCTruth>::const_iterator mcPho = mcPhotons.begin(); mcPho != mcPhotons.end(); mcPho++) {
          // mcConvPt_= (*mcPho).fourMomentum().et();
          float mcPhi = (*mcPho).fourMomentum().phi();
          //simPV_Z = (*mcPho).primaryVertex().z();
          mcPhi_ = phiNormalization(mcPhi);
          mcEta_ = (*mcPho).fourMomentum().pseudoRapidity();
          mcEta_ = etaTransformation(mcEta_, (*mcPho).primaryVertex().z());
          //mcConvR_= (*mcPho).vertex().perp();
          //mcConvX_= (*mcPho).vertex().x();
          //mcConvY_= (*mcPho).vertex().y();
          //mcConvZ_= (*mcPho).vertex().z();
          //mcConvEta_= (*mcPho).vertex().eta();
          //mcConvPhi_= (*mcPho).vertex().phi();
          if (fabs(mcEta_) > END_HI)
            continue;
          //	    if (mcConvPt_<minPhoPtForPurity) continue;
          //if (fabs(mcEta_)>maxPhoEtaForPurity) continue;
          //if (fabs(mcConvZ_)>maxPhoZForPurity) continue;
          //if (mcConvR_>maxPhoRForEffic) continue;

          if ((*mcPho).isAConversion() != 1)
            continue;
          if (!((fabs(mcEta_) <= BARL && mcConvR_ < 85) ||
                (fabs(mcEta_) > BARL && fabs(mcEta_) <= END_HI && fabs((*mcPho).vertex().z()) < 210)))
            continue;

          theConvTP_.clear();
          for (size_t i = 0; i < trackingParticles.size(); ++i) {
            TrackingParticleRef tp(ElectronTPHandle, i);
            if (fabs(tp->vx() - (*mcPho).vertex().x()) < 0.0001 && fabs(tp->vy() - (*mcPho).vertex().y()) < 0.0001 &&
                fabs(tp->vz() - (*mcPho).vertex().z()) < 0.0001) {
              theConvTP_.push_back(tp);
            }
          }

          if (theConvTP_.size() < 2)
            continue;

          reco::RecoToSimCollection const& p1 = trackAssociator->associateRecoToSim(tc1, theConvTP_);
          reco::RecoToSimCollection const& p2 = trackAssociator->associateRecoToSim(tc2, theConvTP_);
          std::vector<std::pair<RefToBase<reco::Track>, double> > trackV1, trackV2;

          auto itP1 = p1.find(tk1);
          auto itP2 = p2.find(tk2);
          bool good = (itP1 != p1.end()) and (not itP1->val.empty()) and (itP2 != p2.end()) and (not itP2->val.empty());
          if (not good) {
            itP1 = p1.find(tk2);
            itP2 = p2.find(tk1);
            good = (itP1 != p1.end()) and (not itP1->val.empty()) and (itP2 != p2.end()) and (not itP2->val.empty());
          }
          if (good) {
            std::vector<std::pair<TrackingParticleRef, double> > const& tp1 = itP1->val;
            std::vector<std::pair<TrackingParticleRef, double> > const& tp2 = itP2->val;

            TrackingParticleRef tpr1 = tp1.front().first;
            TrackingParticleRef tpr2 = tp2.front().first;

            if (abs(tpr1->pdgId()) == 11 && abs(tpr2->pdgId()) == 11) {
              if ((tpr1->parentVertex()->sourceTracks_end() - tpr1->parentVertex()->sourceTracks_begin() == 1) &&
                  (tpr2->parentVertex()->sourceTracks_end() - tpr2->parentVertex()->sourceTracks_begin() == 1)) {
                if (tpr1->parentVertex().key() == tpr2->parentVertex().key() &&
                    ((*tpr1->parentVertex()->sourceTracks_begin())->pdgId() == 22)) {
                  nAssT2 = 2;
                  break;
                }
              }
            }
          }

        }  // end loop over simulated photons

        if (nAssT2 == 2) {
          h_RecoConvTwoMTracks_[0]->Fill(aPho.eta());
          h_RecoConvTwoMTracks_[1]->Fill(aPho.phi());
          if (aConv->conversionVertex().isValid())
            h_RecoConvTwoMTracks_[2]->Fill(aConv->conversionVertex().position().perp2());
          h_RecoConvTwoMTracks_[3]->Fill(aConv->conversionVertex().position().z());
          h_RecoConvTwoMTracks_[4]->Fill(aPho.et());
        }

        ///////////////////////////// xray
        if (aConv->conversionVertex().isValid()) {
          float chi2Prob = ChiSquaredProbability(aConv->conversionVertex().chi2(), aConv->conversionVertex().ndof());
          double convR = sqrt(aConv->conversionVertex().position().perp2());
          double scalar = aConv->conversionVertex().position().x() * aConv->pairMomentum().x() +
                          aConv->conversionVertex().position().y() * aConv->pairMomentum().y();

          if (scalar < 0)
            convR = -sqrt(aConv->conversionVertex().position().perp2());
          h_convVtxRvsZ_[0]->Fill(fabs(aConv->conversionVertex().position().z()),
                                  sqrt(aConv->conversionVertex().position().perp2()));

          if (!aConv->caloCluster().empty()) {
            if (!isRunCentrally_)
              h2_etaVsRreco_[0]->Fill(aConv->caloCluster()[0]->eta(),
                                      sqrt(aConv->conversionVertex().position().perp2()));
            if (fabs(aConv->caloCluster()[0]->eta()) <= 1.) {
              h_convVtxYvsX_->Fill(aConv->conversionVertex().position().y(), aConv->conversionVertex().position().x());
              h_convVtxRvsZ_[1]->Fill(fabs(aConv->conversionVertex().position().z()), convR);

              if (!isRunCentrally_) {
                h_convVtxYvsX_zoom_[0]->Fill(aConv->conversionVertex().position().y(),
                                             aConv->conversionVertex().position().x());
                h_convVtxYvsX_zoom_[1]->Fill(aConv->conversionVertex().position().y(),
                                             aConv->conversionVertex().position().x());
                h_convVtxRvsZ_zoom_[0]->Fill(fabs(aConv->conversionVertex().position().z()), convR);
                h_convVtxRvsZ_zoom_[1]->Fill(fabs(aConv->conversionVertex().position().z()), convR);
              }
            }
            if (fabs(aConv->caloCluster()[0]->eta()) > 1.)
              h_convVtxRvsZ_[2]->Fill(fabs(aConv->conversionVertex().position().z()), convR);
          }

          h_vtxChi2Prob_[0]->Fill(chi2Prob);
          h_vtxChi2_[0]->Fill(aConv->conversionVertex().normalizedChi2());
          if (phoIsInBarrel) {
            h_vtxChi2Prob_[1]->Fill(chi2Prob);
            h_vtxChi2_[1]->Fill(aConv->conversionVertex().normalizedChi2());
          }
          if (phoIsInEndcap) {
            h_vtxChi2Prob_[2]->Fill(chi2Prob);
            h_vtxChi2_[2]->Fill(aConv->conversionVertex().normalizedChi2());
          }

        }  // end conversion vertex valid
      }    // end loop over reco conversions
    }      // end loop on all reco photons
  }        // if !fastSim

  ///////////////// histograms for background
  float nPho = 0;
  for (reco::GenJetCollection::const_iterator genJetIter = genJetCollection.begin();
       genJetIter != genJetCollection.end();
       ++genJetIter) {
    if (genJetIter->pt() < minPhoEtCut_)
      continue;
    if (fabs(genJetIter->eta()) > 2.5)
      continue;

    float mcJetPhi = genJetIter->phi();
    mcJetPhi_ = phiNormalization(mcJetPhi);
    mcJetEta_ = genJetIter->eta();
    float mcJetPt = genJetIter->pt();

    h_SimJet_[0]->Fill(mcJetEta_);
    h_SimJet_[1]->Fill(mcJetPhi_);
    h_SimJet_[2]->Fill(mcJetPt);

    std::vector<reco::Photon> thePhotons;
    bool matched = false;

    reco::Photon matchingPho;
    for (reco::PhotonCollection::const_iterator iPho = photonCollection.begin(); iPho != photonCollection.end();
         iPho++) {
      reco::Photon aPho = reco::Photon(*iPho);
      float phiPho = aPho.phi();
      float etaPho = aPho.eta();
      float deltaPhi = phiPho - mcJetPhi_;
      float deltaEta = etaPho - mcJetEta_;
      if (deltaPhi > pi)
        deltaPhi -= twopi;
      if (deltaPhi < -pi)
        deltaPhi += twopi;
      deltaPhi = pow(deltaPhi, 2);
      deltaEta = pow(deltaEta, 2);
      float delta = sqrt(deltaPhi + deltaEta);
      if (delta < 0.3) {
        matchingPho = *iPho;
        matched = true;
      }
    }  // end loop over reco photons

    if (!matched)
      continue;
    nPho++;

    h_MatchedSimJet_[0]->Fill(mcJetEta_);
    h_MatchedSimJet_[1]->Fill(mcJetPhi_);
    h_MatchedSimJet_[2]->Fill(mcJetPt);

    bool phoIsInBarrel = false;
    bool phoIsInEndcap = false;
    if (fabs(matchingPho.superCluster()->position().eta()) < 1.479) {
      phoIsInBarrel = true;
    } else {
      phoIsInEndcap = true;
    }
    edm::Handle<EcalRecHitCollection> ecalRecHitHandle;
    if (phoIsInBarrel) {
      // Get handle to rec hits ecal barrel
      e.getByToken(barrelEcalHits_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
        Labels l;
        labelsForToken(barrelEcalHits_, l);
        edm::LogError("PhotonProducer") << "Error! Can't get the product " << l.module;
        return;
      }

    } else if (phoIsInEndcap) {
      // Get handle to rec hits ecal encap
      e.getByToken(endcapEcalHits_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
        Labels l;
        labelsForToken(barrelEcalHits_, l);
        edm::LogError("PhotonProducer") << "Error! Can't get the product " << l.module;
        return;
      }
    }

    const EcalRecHitCollection ecalRecHitCollection = *(ecalRecHitHandle.product());
    float photonE = matchingPho.energy();
    float photonEt = matchingPho.et();
    float r9 = matchingPho.r9();
    float r1 = matchingPho.r1x5();
    float r2 = matchingPho.r2x5();
    float sigmaIetaIeta = matchingPho.sigmaIetaIeta();
    float hOverE = matchingPho.hadronicOverEm();
    float ecalIso = matchingPho.ecalRecHitSumEtConeDR04();
    float hcalIso = matchingPho.hcalTowerSumEtConeDR04();
    float trkIso = matchingPho.trkSumPtSolidConeDR04();
    float nIsoTrk = matchingPho.nTrkSolidConeDR04();
    std::vector<std::pair<DetId, float> >::const_iterator rhIt;

    bool atLeastOneDeadChannel = false;
    for (reco::CaloCluster_iterator bcIt = matchingPho.superCluster()->clustersBegin();
         bcIt != matchingPho.superCluster()->clustersEnd();
         ++bcIt) {
      for (rhIt = (*bcIt)->hitsAndFractions().begin(); rhIt != (*bcIt)->hitsAndFractions().end(); ++rhIt) {
        for (EcalRecHitCollection::const_iterator it = ecalRecHitCollection.begin(); it != ecalRecHitCollection.end();
             ++it) {
          if (rhIt->first == (*it).id()) {
            if ((*it).recoFlag() == 9) {
              atLeastOneDeadChannel = true;
              break;
            }
          }
        }
      }
    }

    if (atLeastOneDeadChannel) {
      h_MatchedSimJetBadCh_[0]->Fill(mcJetEta_);
      h_MatchedSimJetBadCh_[1]->Fill(mcJetPhi_);
      h_MatchedSimJetBadCh_[2]->Fill(mcJetPt);
    }

    h_scBkgEta_->Fill(matchingPho.superCluster()->eta());
    h_scBkgPhi_->Fill(matchingPho.superCluster()->phi());
    h_scBkgE_[0]->Fill(matchingPho.superCluster()->energy());
    h_scBkgEt_[0]->Fill(matchingPho.superCluster()->energy() / cosh(matchingPho.superCluster()->eta()));
    //
    h_phoBkgEta_->Fill(matchingPho.eta());
    h_phoBkgPhi_->Fill(matchingPho.phi());
    h_phoBkgE_[0]->Fill(photonE);
    h_phoBkgEt_[0]->Fill(photonEt);
    h_phoBkgDEta_->Fill(matchingPho.eta() - mcJetEta_);
    h_phoBkgDPhi_->Fill(matchingPho.phi() - mcJetPhi_);

    h_r9Bkg_[0]->Fill(r9);
    h_r1Bkg_[0]->Fill(r1);
    h_r2Bkg_[0]->Fill(r2);
    h_sigmaIetaIetaBkg_[0]->Fill(sigmaIetaIeta);
    h_hOverEBkg_[0]->Fill(hOverE);
    h_ecalRecHitSumEtConeDR04Bkg_[0]->Fill(ecalIso);
    h_hcalTowerSumEtConeDR04Bkg_[0]->Fill(hcalIso);
    h_isoTrkSolidConeDR04Bkg_[0]->Fill(trkIso);
    h_nTrkSolidConeDR04Bkg_[0]->Fill(nIsoTrk);

    if (!isRunCentrally_) {
      h2_r9VsEtaBkg_->Fill(mcJetEta_, r9);
      h2_r9VsEtBkg_->Fill(mcJetPt, r9);
      h2_r1VsEtaBkg_->Fill(mcJetEta_, r1);
      h2_r1VsEtBkg_->Fill(mcJetPt, r1);
      h2_r2VsEtaBkg_->Fill(mcJetEta_, r2);
      h2_r2VsEtBkg_->Fill(mcJetPt, r2);
      h2_sigmaIetaIetaVsEtaBkg_->Fill(mcJetEta_, sigmaIetaIeta);
      h2_sigmaIetaIetaVsEtBkg_[0]->Fill(mcJetPt, sigmaIetaIeta);
      h2_hOverEVsEtaBkg_->Fill(mcJetEta_, hOverE);
      h2_hOverEVsEtBkg_->Fill(mcJetPt, hOverE);

      p_r1VsEtaBkg_->Fill(mcJetEta_, r1);
      p_r1VsEtBkg_->Fill(mcJetPt, r1);
      p_r2VsEtaBkg_->Fill(mcJetEta_, r2);
      p_r2VsEtBkg_->Fill(mcJetPt, r2);
      p_sigmaIetaIetaVsEtaBkg_->Fill(mcJetEta_, sigmaIetaIeta);
      p_sigmaIetaIetaVsEtBkg_[0]->Fill(mcJetPt, sigmaIetaIeta);
      p_hOverEVsEtaBkg_->Fill(mcJetEta_, hOverE);
      p_hOverEVsEtBkg_->Fill(mcJetPt, hOverE);
    }

    if (!isRunCentrally_) {
      h2_ecalRecHitSumEtConeDR04VsEtaBkg_->Fill(mcJetEta_, ecalIso);
      h2_ecalRecHitSumEtConeDR04VsEtBkg_[0]->Fill(mcJetPt, ecalIso);
      h2_hcalTowerSumEtConeDR04VsEtaBkg_->Fill(mcJetEta_, hcalIso);
      h2_hcalTowerSumEtConeDR04VsEtBkg_[0]->Fill(mcJetPt, hcalIso);
      p_ecalRecHitSumEtConeDR04VsEtaBkg_->Fill(mcJetEta_, ecalIso);
      p_ecalRecHitSumEtConeDR04VsEtBkg_[0]->Fill(mcJetPt, ecalIso);
      p_hcalTowerSumEtConeDR04VsEtaBkg_->Fill(mcJetEta_, hcalIso);
      p_hcalTowerSumEtConeDR04VsEtBkg_[0]->Fill(mcJetPt, hcalIso);
      p_isoTrkSolidConeDR04VsEtaBkg_->Fill(mcJetEta_, trkIso);
      p_isoTrkSolidConeDR04VsEtBkg_[0]->Fill(mcJetPt, trkIso);
      p_nTrkSolidConeDR04VsEtaBkg_->Fill(mcJetEta_, nIsoTrk);
      p_nTrkSolidConeDR04VsEtBkg_[0]->Fill(mcJetPt, nIsoTrk);
      h2_isoTrkSolidConeDR04VsEtaBkg_->Fill(mcJetEta_, trkIso);
      h2_isoTrkSolidConeDR04VsEtBkg_[0]->Fill(mcJetPt, trkIso);
      h2_nTrkSolidConeDR04VsEtaBkg_->Fill(mcJetEta_, nIsoTrk);
      h2_nTrkSolidConeDR04VsEtBkg_[0]->Fill(mcJetPt, nIsoTrk);
    }

    if (phoIsInBarrel) {
      h_r9Bkg_[1]->Fill(r9);
      h_r1Bkg_[1]->Fill(r1);
      h_r2Bkg_[1]->Fill(r2);

      h_sigmaIetaIetaBkg_[1]->Fill(sigmaIetaIeta);
      h_hOverEBkg_[1]->Fill(hOverE);
      h_ecalRecHitSumEtConeDR04Bkg_[1]->Fill(ecalIso);
      h_hcalTowerSumEtConeDR04Bkg_[1]->Fill(hcalIso);
      h_isoTrkSolidConeDR04Bkg_[1]->Fill(trkIso);
      h_nTrkSolidConeDR04Bkg_[1]->Fill(nIsoTrk);

      if (!isRunCentrally_) {
        h2_sigmaIetaIetaVsEtBkg_[1]->Fill(mcJetPt, sigmaIetaIeta);
        h2_isoTrkSolidConeDR04VsEtBkg_[1]->Fill(mcJetPt, trkIso);
        h2_nTrkSolidConeDR04VsEtBkg_[1]->Fill(mcJetPt, nIsoTrk);
        h2_ecalRecHitSumEtConeDR04VsEtBkg_[1]->Fill(mcJetPt, ecalIso);
        h2_hcalTowerSumEtConeDR04VsEtBkg_[1]->Fill(mcJetPt, hcalIso);
        p_sigmaIetaIetaVsEtBkg_[1]->Fill(mcJetPt, sigmaIetaIeta);
        p_ecalRecHitSumEtConeDR04VsEtBkg_[1]->Fill(mcJetPt, ecalIso);
        p_hcalTowerSumEtConeDR04VsEtBkg_[1]->Fill(mcJetPt, hcalIso);
        p_isoTrkSolidConeDR04VsEtBkg_[1]->Fill(mcJetPt, trkIso);
        p_nTrkSolidConeDR04VsEtBkg_[1]->Fill(mcJetPt, nIsoTrk);
      }
    } else if (phoIsInEndcap) {
      h_r9Bkg_[2]->Fill(r9);
      h_r1Bkg_[2]->Fill(r1);
      h_r2Bkg_[2]->Fill(r2);

      h_sigmaIetaIetaBkg_[2]->Fill(sigmaIetaIeta);
      h_hOverEBkg_[2]->Fill(hOverE);
      h_ecalRecHitSumEtConeDR04Bkg_[2]->Fill(ecalIso);
      h_hcalTowerSumEtConeDR04Bkg_[2]->Fill(hcalIso);
      h_isoTrkSolidConeDR04Bkg_[2]->Fill(trkIso);
      h_nTrkSolidConeDR04Bkg_[2]->Fill(nIsoTrk);

      if (!isRunCentrally_) {
        h2_sigmaIetaIetaVsEtBkg_[2]->Fill(mcJetPt, sigmaIetaIeta);
        h2_isoTrkSolidConeDR04VsEtBkg_[2]->Fill(mcJetPt, trkIso);
        h2_nTrkSolidConeDR04VsEtBkg_[2]->Fill(mcJetPt, nIsoTrk);
        h2_ecalRecHitSumEtConeDR04VsEtBkg_[2]->Fill(mcJetPt, ecalIso);
        h2_hcalTowerSumEtConeDR04VsEtBkg_[2]->Fill(mcJetPt, hcalIso);
        p_sigmaIetaIetaVsEtBkg_[2]->Fill(mcJetPt, sigmaIetaIeta);
        p_ecalRecHitSumEtConeDR04VsEtBkg_[2]->Fill(mcJetPt, ecalIso);
        p_hcalTowerSumEtConeDR04VsEtBkg_[2]->Fill(mcJetPt, hcalIso);
        p_isoTrkSolidConeDR04VsEtBkg_[2]->Fill(mcJetPt, trkIso);
        p_nTrkSolidConeDR04VsEtBkg_[2]->Fill(mcJetPt, nIsoTrk);
      }
    }

    if (!fastSim_) {
      ////////////////// plot quantities related to conversions
      reco::ConversionRefVector conversions = matchingPho.conversions();
      for (unsigned int iConv = 0; iConv < conversions.size(); iConv++) {
        reco::ConversionRef aConv = conversions[iConv];
        //std::vector<reco::TrackRef> tracks = aConv->tracks();
        const std::vector<edm::RefToBase<reco::Track> > tracks = aConv->tracks();
        double like = aConv->MVAout();
        if (like < likelihoodCut_)
          continue;
        if (tracks.size() < 2)
          continue;
        if (!aConv->caloCluster().empty()) {
          h_convEtaBkg_->Fill(aConv->caloCluster()[0]->eta());
          h_convPhiBkg_->Fill(aConv->caloCluster()[0]->phi());
        }
        h_mvaOutBkg_[0]->Fill(like);
        float eoverp = aConv->EoverP();
        h_EoverPTracksBkg_[0]->Fill(eoverp);
        h_PoverETracksBkg_[0]->Fill(1. / eoverp);
        h_DCotTracksBkg_[0]->Fill(aConv->pairCotThetaSeparation());
        float dPhiTracksAtVtx = aConv->dPhiTracksAtVtx();
        h_DPhiTracksAtVtxBkg_[0]->Fill(dPhiTracksAtVtx);

        if (phoIsInBarrel) {
          h_mvaOutBkg_[1]->Fill(like);
          h_EoverPTracksBkg_[1]->Fill(eoverp);
          h_PoverETracksBkg_[1]->Fill(1. / eoverp);
          h_DCotTracksBkg_[1]->Fill(aConv->pairCotThetaSeparation());
          h_DPhiTracksAtVtxBkg_[1]->Fill(dPhiTracksAtVtx);
        } else if (phoIsInEndcap) {
          h_mvaOutBkg_[2]->Fill(like);
          h_EoverPTracksBkg_[2]->Fill(eoverp);
          h_PoverETracksBkg_[2]->Fill(1. / eoverp);
          h_DCotTracksBkg_[2]->Fill(aConv->pairCotThetaSeparation());
          h_DPhiTracksAtVtxBkg_[2]->Fill(dPhiTracksAtVtx);
        }

        if (aConv->conversionVertex().isValid()) {
          double convR = sqrt(aConv->conversionVertex().position().perp2());
          double scalar = aConv->conversionVertex().position().x() * aConv->pairMomentum().x() +
                          aConv->conversionVertex().position().y() * aConv->pairMomentum().y();
          if (scalar < 0)
            convR = -sqrt(aConv->conversionVertex().position().perp2());

          if (!isRunCentrally_) {
            h_convVtxRvsZBkg_[0]->Fill(fabs(aConv->conversionVertex().position().z()),
                                       sqrt(aConv->conversionVertex().position().perp2()));
            if (!aConv->caloCluster().empty() && fabs(aConv->caloCluster()[0]->eta()) <= 1.) {
              h_convVtxYvsXBkg_->Fill(aConv->conversionVertex().position().y(),
                                      aConv->conversionVertex().position().x());
              h_convVtxRvsZBkg_[1]->Fill(fabs(aConv->conversionVertex().position().z()), convR);
            }
          }

        }  // end vertex valid

      }  // end loop over conversions
    }    // if !fastSim
  }      // end loop over sim jets

  /////// separate loop to compare with miniAOD
  for (reco::GenParticleCollection::const_iterator mcIter = genParticles->begin(); mcIter != genParticles->end();
       mcIter++) {
    if (!(mcIter->pdgId() == 22))
      continue;
    if (mcIter->mother() != nullptr and !(mcIter->mother()->pdgId() == 25))
      continue;
    if (fabs(mcIter->eta()) > 2.5)
      continue;

    float mcPhi = mcIter->phi();
    float mcEta = mcIter->eta();
    //mcEta = etaTransformation(mcEta, (*mcPho).primaryVertex().z() );
    float mcEnergy = mcIter->energy();

    double dR = 9999999.;
    float minDr = 10000.;
    int iMatch = -1;
    bool matched = false;

    for (unsigned int ipho = 0; ipho < photonHandle->size(); ipho++) {
      reco::PhotonRef pho(reco::PhotonRef(photonHandle, ipho));

      double dphi = pho->phi() - mcPhi;
      if (std::fabs(dphi) > CLHEP::pi) {
        dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
      }
      double deta = pho->superCluster()->position().eta() - mcEta;

      dR = sqrt(pow((deta), 2) + pow(dphi, 2));
      if (dR < 0.1 && dR < minDr) {
        minDr = dR;
        iMatch = ipho;
      }
    }

    if (iMatch > -1)
      matched = true;
    if (!matched)
      continue;

    reco::PhotonRef matchingPho(reco::PhotonRef(photonHandle, iMatch));

    bool phoIsInBarrel = false;
    bool phoIsInEndcap = false;

    float phoEta = matchingPho->superCluster()->position().eta();
    if (fabs(phoEta) < 1.479) {
      phoIsInBarrel = true;
    } else {
      phoIsInEndcap = true;
    }

    float photonE = matchingPho->energy();
    float sigmaEoE = matchingPho->getCorrectedEnergyError(matchingPho->getCandidateP4type()) / matchingPho->energy();
    float photonEt = matchingPho->energy() / cosh(matchingPho->eta());
    //	float photonERegr1 = matchingPho->getCorrectedEnergy(reco::Photon::regression1);
    //float photonERegr2 = matchingPho->getCorrectedEnergy(reco::Photon::regression2);
    float r9 = matchingPho->r9();
    float full5x5_r9 = matchingPho->full5x5_r9();
    float r1 = matchingPho->r1x5();
    float r2 = matchingPho->r2x5();
    float sigmaIetaIeta = matchingPho->sigmaIetaIeta();
    float full5x5_sieie = matchingPho->full5x5_sigmaIetaIeta();
    float hOverE = matchingPho->hadronicOverEm();
    float newhOverE = matchingPho->hadTowOverEm();
    float ecalIso = matchingPho->ecalRecHitSumEtConeDR04();
    float hcalIso = matchingPho->hcalTowerSumEtConeDR04();
    float newhcalIso = matchingPho->hcalTowerSumEtBcConeDR04();
    float trkIso = matchingPho->trkSumPtSolidConeDR04();
    float nIsoTrk = matchingPho->nTrkSolidConeDR04();
    // PF related quantities
    float chargedHadIso = matchingPho->chargedHadronIso();
    float neutralHadIso = matchingPho->neutralHadronIso();
    float photonIso = matchingPho->photonIso();
    //	float etOutsideMustache = matchingPho->etOutsideMustache();
    //	int   nClusterOutsideMustache = matchingPho->nClusterOutsideMustache();
    //float pfMVA = matchingPho->pfMVA();

    if ((photonEt > 14 && newhOverE < 0.15) || (photonEt > 10 && photonEt < 14 && chargedHadIso < 10)) {
      h_scEta_miniAOD_[0]->Fill(matchingPho->superCluster()->eta());
      h_scPhi_miniAOD_[0]->Fill(matchingPho->superCluster()->phi());

      h_phoE_miniAOD_[0][0]->Fill(photonE);
      h_phoEt_miniAOD_[0][0]->Fill(photonEt);

      h_phoERes_miniAOD_[0][0]->Fill(photonE / mcEnergy);
      h_phoSigmaEoE_miniAOD_[0][0]->Fill(sigmaEoE);

      h_r9_miniAOD_[0][0]->Fill(r9);
      h_full5x5_r9_miniAOD_[0][0]->Fill(full5x5_r9);
      h_r1_miniAOD_[0][0]->Fill(r1);
      h_r2_miniAOD_[0][0]->Fill(r2);

      h_sigmaIetaIeta_miniAOD_[0][0]->Fill(sigmaIetaIeta);
      h_full5x5_sigmaIetaIeta_miniAOD_[0][0]->Fill(full5x5_sieie);
      h_hOverE_miniAOD_[0][0]->Fill(hOverE);
      h_newhOverE_miniAOD_[0][0]->Fill(newhOverE);

      h_ecalRecHitSumEtConeDR04_miniAOD_[0][0]->Fill(ecalIso);
      h_hcalTowerSumEtConeDR04_miniAOD_[0][0]->Fill(hcalIso);
      h_hcalTowerBcSumEtConeDR04_miniAOD_[0][0]->Fill(newhcalIso);
      h_isoTrkSolidConeDR04_miniAOD_[0][0]->Fill(trkIso);
      h_nTrkSolidConeDR04_miniAOD_[0][0]->Fill(nIsoTrk);

      //
      h_chHadIso_miniAOD_[0]->Fill(chargedHadIso);
      h_nHadIso_miniAOD_[0]->Fill(neutralHadIso);
      h_phoIso_miniAOD_[0]->Fill(photonIso);

      //
      if (phoIsInBarrel) {
        h_phoE_miniAOD_[0][1]->Fill(photonE);
        h_phoEt_miniAOD_[0][1]->Fill(photonEt);

        h_phoERes_miniAOD_[0][1]->Fill(photonE / mcEnergy);
        h_phoSigmaEoE_miniAOD_[0][1]->Fill(sigmaEoE);

        h_r9_miniAOD_[0][1]->Fill(r9);
        h_full5x5_r9_miniAOD_[0][1]->Fill(full5x5_r9);
        h_r1_miniAOD_[0][1]->Fill(r1);
        h_r2_miniAOD_[0][1]->Fill(r2);
        h_sigmaIetaIeta_miniAOD_[0][1]->Fill(sigmaIetaIeta);
        h_full5x5_sigmaIetaIeta_miniAOD_[0][1]->Fill(full5x5_sieie);
        h_hOverE_miniAOD_[0][1]->Fill(hOverE);
        h_newhOverE_miniAOD_[0][1]->Fill(newhOverE);
        h_ecalRecHitSumEtConeDR04_miniAOD_[0][1]->Fill(ecalIso);
        h_hcalTowerSumEtConeDR04_miniAOD_[0][1]->Fill(hcalIso);
        h_hcalTowerBcSumEtConeDR04_miniAOD_[0][1]->Fill(newhcalIso);
        h_isoTrkSolidConeDR04_miniAOD_[0][1]->Fill(trkIso);
        h_nTrkSolidConeDR04_miniAOD_[0][1]->Fill(nIsoTrk);
        h_chHadIso_miniAOD_[1]->Fill(chargedHadIso);
        h_nHadIso_miniAOD_[1]->Fill(neutralHadIso);
        h_phoIso_miniAOD_[1]->Fill(photonIso);
      }
      if (phoIsInEndcap) {
        h_phoE_miniAOD_[0][2]->Fill(photonE);
        h_phoEt_miniAOD_[0][2]->Fill(photonEt);

        h_phoERes_miniAOD_[0][2]->Fill(photonE / mcEnergy);
        h_phoSigmaEoE_miniAOD_[0][2]->Fill(sigmaEoE);
        h_r9_miniAOD_[0][2]->Fill(r9);
        h_full5x5_r9_miniAOD_[0][2]->Fill(full5x5_r9);
        h_r1_miniAOD_[0][2]->Fill(r1);
        h_r2_miniAOD_[0][2]->Fill(r2);
        h_sigmaIetaIeta_miniAOD_[0][2]->Fill(sigmaIetaIeta);
        h_full5x5_sigmaIetaIeta_miniAOD_[0][2]->Fill(full5x5_sieie);
        h_hOverE_miniAOD_[0][2]->Fill(hOverE);
        h_newhOverE_miniAOD_[0][2]->Fill(newhOverE);
        h_ecalRecHitSumEtConeDR04_miniAOD_[0][2]->Fill(ecalIso);
        h_hcalTowerSumEtConeDR04_miniAOD_[0][2]->Fill(hcalIso);
        h_hcalTowerBcSumEtConeDR04_miniAOD_[0][2]->Fill(newhcalIso);
        h_isoTrkSolidConeDR04_miniAOD_[0][2]->Fill(trkIso);
        h_nTrkSolidConeDR04_miniAOD_[0][2]->Fill(nIsoTrk);
        h_chHadIso_miniAOD_[2]->Fill(chargedHadIso);
        h_nHadIso_miniAOD_[2]->Fill(neutralHadIso);
        h_phoIso_miniAOD_[2]->Fill(photonIso);
      }
    }  // end histos for comparing with miniAOD

  }  // end loop over gen photons

  h_nPho_->Fill(float(nPho));
}

float PhotonValidator::phiNormalization(float& phi) {
  //---Definitions
  const float PI = 3.1415927;
  const float TWOPI = 2.0 * PI;

  if (phi > PI) {
    phi = phi - TWOPI;
  }
  if (phi < -PI) {
    phi = phi + TWOPI;
  }

  return phi;
}

float PhotonValidator::etaTransformation(float EtaParticle, float Zvertex) {
  //---Definitions
  const float PI = 3.1415927;

  //---Definitions for ECAL
  const float R_ECAL = 136.5;
  const float Z_Endcap = 328.0;
  const float etaBarrelEndcap = 1.479;

  //---ETA correction

  float Theta = 0.0;
  float ZEcal = R_ECAL * sinh(EtaParticle) + Zvertex;

  if (ZEcal != 0.0)
    Theta = atan(R_ECAL / ZEcal);
  if (Theta < 0.0)
    Theta = Theta + PI;
  float ETA = -log(tan(0.5 * Theta));

  if (fabs(ETA) > etaBarrelEndcap) {
    float Zend = Z_Endcap;
    if (EtaParticle < 0.0)
      Zend = -Zend;
    float Zlen = Zend - Zvertex;
    float RR = Zlen / sinh(EtaParticle);
    Theta = atan(RR / Zend);
    if (Theta < 0.0)
      Theta = Theta + PI;
    ETA = -log(tan(0.5 * Theta));
  }
  //---Return the result
  return ETA;
  //---end
}

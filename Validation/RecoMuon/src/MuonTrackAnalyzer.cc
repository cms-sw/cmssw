/** \class MuonTrackAnalyzer
 *  Analyzer of the Muon tracks
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "Validation/RecoMuon/src/MuonTrackAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"

#include "Validation/RecoMuon/src/Histograms.h"
#include "Validation/RecoMuon/src/HTrack.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;

/// Constructor
MuonTrackAnalyzer::MuonTrackAnalyzer(const ParameterSet &ps) {
  // service parameters
  pset = ps;
  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  // the services
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());

  theSimTracksLabel = edm::InputTag("g4SimHits");
  theSimTracksToken = consumes<edm::SimTrackContainer>(theSimTracksLabel);

  theTracksLabel = pset.getParameter<InputTag>("Tracks");
  theTracksToken = consumes<reco::TrackCollection>(theTracksLabel);
  doTracksAnalysis = pset.getUntrackedParameter<bool>("DoTracksAnalysis", true);

  doSeedsAnalysis = pset.getUntrackedParameter<bool>("DoSeedsAnalysis", false);
  if (doSeedsAnalysis) {
    theSeedsLabel = pset.getParameter<InputTag>("MuonSeed");
    theSeedsToken = consumes<TrajectorySeedCollection>(theSeedsLabel);
    ParameterSet updatorPar = pset.getParameter<ParameterSet>("MuonUpdatorAtVertexParameters");
    theSeedPropagatorName = updatorPar.getParameter<string>("Propagator");

    theUpdator = new MuonUpdatorAtVertex(updatorPar, theService);
  }

  theCSCSimHitLabel = pset.getParameter<InputTag>("CSCSimHit");
  theDTSimHitLabel = pset.getParameter<InputTag>("DTSimHit");
  theRPCSimHitLabel = pset.getParameter<InputTag>("RPCSimHit");
  theCSCSimHitToken = consumes<std::vector<PSimHit> >(theCSCSimHitLabel);
  theDTSimHitToken = consumes<std::vector<PSimHit> >(theDTSimHitLabel);
  theRPCSimHitToken = consumes<std::vector<PSimHit> >(theRPCSimHitLabel);

  theEtaRange = (EtaRange)pset.getParameter<int>("EtaRange");

  // number of sim tracks
  numberOfSimTracks = 0;
  // number of reco tracks
  numberOfRecTracks = 0;

  dbe_ = edm::Service<DQMStore>().operator->();
  out = pset.getUntrackedParameter<string>("rootFileName");
  dirName_ = pset.getUntrackedParameter<std::string>("dirName");
  subsystemname_ = pset.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem");
}

/// Destructor
MuonTrackAnalyzer::~MuonTrackAnalyzer() {
  if (theService)
    delete theService;
}

void MuonTrackAnalyzer::beginJob() {
  //theFile->cd();
}
void MuonTrackAnalyzer::bookHistograms(DQMStore::IBooker &ibooker,
                                       edm::Run const &iRun,
                                       edm::EventSetup const & /* iSetup */) {
  ibooker.cd();

  InputTag algo = theTracksLabel;
  string dirName = dirName_;
  if (!algo.process().empty())
    dirName += algo.process() + "_";
  if (!algo.label().empty())
    dirName += algo.label() + "_";
  if (!algo.instance().empty())
    dirName += algo.instance() + "";
  if (dirName.find("Tracks") < dirName.length()) {
    dirName.replace(dirName.find("Tracks"), 6, "");
  }
  std::replace(dirName.begin(), dirName.end(), ':', '_');
  ibooker.setCurrentFolder(dirName);

  //ibooker.goUp();
  std::string simName = dirName;
  simName += "/SimTracks";
  hSimTracks = new HTrackVariables(ibooker, simName, "SimTracks");

  ibooker.cd();
  ibooker.setCurrentFolder(dirName);

  // Create the root file
  //theFile = new TFile(theRootFileName.c_str(), "RECREATE");

  if (doSeedsAnalysis) {
    ibooker.cd();
    ibooker.setCurrentFolder(dirName);
    hRecoSeedInner = new HTrack(ibooker, dirName, "RecoSeed", "Inner");
    hRecoSeedPCA = new HTrack(ibooker, dirName, "RecoSeed", "PCA");
  }

  if (doTracksAnalysis) {
    ibooker.cd();
    ibooker.setCurrentFolder(dirName);
    hRecoTracksPCA = new HTrack(ibooker, dirName, "RecoTracks", "PCA");
    hRecoTracksInner = new HTrack(ibooker, dirName, "RecoTracks", "Inner");
    hRecoTracksOuter = new HTrack(ibooker, dirName, "RecoTracks", "Outer");

    ibooker.cd();
    ibooker.setCurrentFolder(dirName);

    // General Histos

    hChi2 = ibooker.book1D("chi2", "#chi^2", 200, 0, 200);
    hChi2VsEta = ibooker.book2D("chi2VsEta", "#chi^2 VS #eta", 120, -3., 3., 200, 0, 200);

    hChi2Norm = ibooker.book1D("chi2Norm", "Normalized #chi^2", 400, 0, 100);
    hChi2NormVsEta = ibooker.book2D("chi2NormVsEta", "Normalized #chi^2 VS #eta", 120, -3., 3., 400, 0, 100);

    hHitsPerTrack = ibooker.book1D("HitsPerTrack", "Number of hits per track", 55, 0, 55);
    hHitsPerTrackVsEta =
        ibooker.book2D("HitsPerTrackVsEta", "Number of hits per track VS #eta", 120, -3., 3., 55, 0, 55);

    hDof = ibooker.book1D("dof", "Number of Degree of Freedom", 55, 0, 55);
    hDofVsEta = ibooker.book2D("dofVsEta", "Number of Degree of Freedom VS #eta", 120, -3., 3., 55, 0, 55);

    hChi2Prob = ibooker.book1D("chi2Prob", "#chi^2 probability", 200, 0, 1);
    hChi2ProbVsEta = ibooker.book2D("chi2ProbVsEta", "#chi^2 probability VS #eta", 120, -3., 3., 200, 0, 1);

    hNumberOfTracks = ibooker.book1D("NumberOfTracks", "Number of reconstructed tracks per event", 200, 0, 200);
    hNumberOfTracksVsEta = ibooker.book2D(
        "NumberOfTracksVsEta", "Number of reconstructed tracks per event VS #eta", 120, -3., 3., 10, 0, 10);

    hChargeVsEta = ibooker.book2D("ChargeVsEta", "Charge vs #eta gen", 120, -3., 3., 4, -2., 2.);
    hChargeVsPt = ibooker.book2D("ChargeVsPt", "Charge vs P_{T} gen", 250, 0, 200, 4, -2., 2.);
    hPtRecVsPtGen = ibooker.book2D("PtRecVsPtGen", "P_{T} rec vs P_{T} gen", 250, 0, 200, 250, 0, 200);

    hDeltaPtVsEta = ibooker.book2D("DeltaPtVsEta", "#Delta P_{t} vs #eta gen", 120, -3., 3., 500, -250., 250.);
    hDeltaPt_In_Out_VsEta =
        ibooker.book2D("DeltaPt_In_Out_VsEta_", "P^{in}_{t} - P^{out}_{t} vs #eta gen", 120, -3., 3., 500, -250., 250.);
  }
}

void MuonTrackAnalyzer::analyze(const Event &event, const EventSetup &eventSetup) {
  LogDebug("MuonTrackAnalyzer") << "Run: " << event.id().run() << " Event: " << event.id().event();

  // Update the services
  theService->update(eventSetup);

  Handle<SimTrackContainer> simTracks;
  event.getByToken(theSimTracksToken, simTracks);
  fillPlots(event, simTracks);

  if (doTracksAnalysis)
    tracksAnalysis(event, eventSetup, simTracks);

  if (doSeedsAnalysis)
    seedsAnalysis(event, eventSetup, simTracks);
}

void MuonTrackAnalyzer::seedsAnalysis(const Event &event,
                                      const EventSetup &eventSetup,
                                      Handle<SimTrackContainer> simTracks) {
  MuonPatternRecoDumper debug;

  // Get the RecTrack collection from the event
  Handle<TrajectorySeedCollection> seeds;
  event.getByToken(theSeedsToken, seeds);

  LogTrace("MuonTrackAnalyzer") << "Number of reconstructed seeds: " << seeds->size() << endl;

  for (TrajectorySeedCollection::const_iterator seed = seeds->begin(); seed != seeds->end(); ++seed) {
    TrajectoryStateOnSurface seedTSOS = getSeedTSOS(*seed);
    pair<SimTrack, double> sim = getSimTrack(seedTSOS, simTracks);
    fillPlots(seedTSOS, sim.first, hRecoSeedInner, debug);

    std::pair<bool, FreeTrajectoryState> propSeed = theUpdator->propagateToNominalLine(seedTSOS);
    if (propSeed.first)
      fillPlots(propSeed.second, sim.first, hRecoSeedPCA, debug);
    else
      LogTrace("MuonTrackAnalyzer") << "Error in seed propagation" << endl;
  }
}

void MuonTrackAnalyzer::tracksAnalysis(const Event &event,
                                       const EventSetup &eventSetup,
                                       Handle<SimTrackContainer> simTracks) {
  MuonPatternRecoDumper debug;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> tracks;
  event.getByToken(theTracksToken, tracks);

  LogTrace("MuonTrackAnalyzer") << "Reconstructed tracks: " << tracks->size() << endl;
  hNumberOfTracks->Fill(tracks->size());

  if (!tracks->empty())
    numberOfRecTracks++;

  // Loop over the Rec tracks
  for (reco::TrackCollection::const_iterator t = tracks->begin(); t != tracks->end(); ++t) {
    reco::TransientTrack track(*t, &*theService->magneticField(), theService->trackingGeometry());

    TrajectoryStateOnSurface outerTSOS = track.outermostMeasurementState();
    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();
    TrajectoryStateOnSurface pcaTSOS = track.impactPointState();

    pair<SimTrack, double> sim = getSimTrack(pcaTSOS, simTracks);
    SimTrack simTrack = sim.first;
    hNumberOfTracksVsEta->Fill(simTrack.momentum().eta(), tracks->size());
    fillPlots(track, simTrack);

    LogTrace("MuonTrackAnalyzer") << "State at the outer surface: " << endl;
    fillPlots(outerTSOS, simTrack, hRecoTracksOuter, debug);

    LogTrace("MuonTrackAnalyzer") << "State at the inner surface: " << endl;
    fillPlots(innerTSOS, simTrack, hRecoTracksInner, debug);

    LogTrace("MuonTrackAnalyzer") << "State at PCA: " << endl;
    fillPlots(pcaTSOS, simTrack, hRecoTracksPCA, debug);

    double deltaPt_in_out = innerTSOS.globalMomentum().perp() - outerTSOS.globalMomentum().perp();
    hDeltaPt_In_Out_VsEta->Fill(simTrack.momentum().eta(), deltaPt_in_out);

    double deltaPt_pca_sim = pcaTSOS.globalMomentum().perp() - sqrt(simTrack.momentum().Perp2());
    hDeltaPtVsEta->Fill(simTrack.momentum().eta(), deltaPt_pca_sim);

    hChargeVsEta->Fill(simTrack.momentum().eta(), pcaTSOS.charge());

    hChargeVsPt->Fill(sqrt(simTrack.momentum().perp2()), pcaTSOS.charge());

    hPtRecVsPtGen->Fill(sqrt(simTrack.momentum().perp2()), pcaTSOS.globalMomentum().perp());
  }
  LogTrace("MuonTrackAnalyzer") << "--------------------------------------------" << endl;
}

void MuonTrackAnalyzer::fillPlots(const Event &event, edm::Handle<edm::SimTrackContainer> &simTracks) {
  if (!checkMuonSimHitPresence(event, simTracks))
    return;

  // Loop over the Sim tracks
  SimTrackContainer::const_iterator simTrack;
  LogTrace("MuonTrackAnalyzer") << "Simulated tracks: " << simTracks->size() << endl;

  for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
    if (abs((*simTrack).type()) == 13) {
      if (!isInTheAcceptance((*simTrack).momentum().eta()))
        continue;  // FIXME!!

      numberOfSimTracks++;

      LogTrace("MuonTrackAnalyzer") << "Simualted muon:" << endl;
      LogTrace("MuonTrackAnalyzer") << "Sim pT: " << sqrt((*simTrack).momentum().perp2()) << endl;
      LogTrace("MuonTrackAnalyzer") << "Sim Eta: " << (*simTrack).momentum().eta() << endl;  // FIXME

      hSimTracks->Fill((*simTrack).momentum().mag(),
                       sqrt((*simTrack).momentum().perp2()),
                       (*simTrack).momentum().eta(),
                       (*simTrack).momentum().phi(),
                       -(*simTrack).type() / abs((*simTrack).type()));  // Double FIXME
      LogTrace("MuonTrackAnalyzer") << "hSimTracks filled" << endl;
    }

  LogTrace("MuonTrackAnalyzer") << endl;
}

void MuonTrackAnalyzer::fillPlots(reco::TransientTrack &track, SimTrack &simTrack) {
  LogTrace("MuonTrackAnalyzer") << "Analizer: New track, chi2: " << track.chi2() << " dof: " << track.ndof() << endl;
  hChi2->Fill(track.chi2());
  hDof->Fill(track.ndof());
  hChi2Norm->Fill(track.normalizedChi2());
  hHitsPerTrack->Fill(track.recHitsSize());

  hChi2Prob->Fill(ChiSquaredProbability(track.chi2(), track.ndof()));

  hChi2VsEta->Fill(simTrack.momentum().eta(), track.chi2());
  hChi2NormVsEta->Fill(simTrack.momentum().eta(), track.normalizedChi2());
  hChi2ProbVsEta->Fill(simTrack.momentum().eta(), ChiSquaredProbability(track.chi2(), track.ndof()));
  hHitsPerTrackVsEta->Fill(simTrack.momentum().eta(), track.recHitsSize());
  hDofVsEta->Fill(simTrack.momentum().eta(), track.ndof());
}

void MuonTrackAnalyzer::fillPlots(TrajectoryStateOnSurface &recoTSOS,
                                  SimTrack &simTrack,
                                  HTrack *histo,
                                  MuonPatternRecoDumper &debug) {
  LogTrace("MuonTrackAnalyzer") << debug.dumpTSOS(recoTSOS) << endl;
  histo->Fill(recoTSOS);

  GlobalVector tsosVect = recoTSOS.globalMomentum();
  math::XYZVectorD reco(tsosVect.x(), tsosVect.y(), tsosVect.z());
  double deltaRVal = deltaR<double>(reco.eta(), reco.phi(), simTrack.momentum().eta(), simTrack.momentum().phi());
  histo->FillDeltaR(deltaRVal);

  histo->computeResolutionAndPull(recoTSOS, simTrack);
}

void MuonTrackAnalyzer::fillPlots(FreeTrajectoryState &recoFTS,
                                  SimTrack &simTrack,
                                  HTrack *histo,
                                  MuonPatternRecoDumper &debug) {
  LogTrace("MuonTrackAnalyzer") << debug.dumpFTS(recoFTS) << endl;
  histo->Fill(recoFTS);

  GlobalVector ftsVect = recoFTS.momentum();
  math::XYZVectorD reco(ftsVect.x(), ftsVect.y(), ftsVect.z());
  double deltaRVal = deltaR<double>(reco.eta(), reco.phi(), simTrack.momentum().eta(), simTrack.momentum().phi());
  histo->FillDeltaR(deltaRVal);

  histo->computeResolutionAndPull(recoFTS, simTrack);
}

pair<SimTrack, double> MuonTrackAnalyzer::getSimTrack(TrajectoryStateOnSurface &tsos,
                                                      Handle<SimTrackContainer> simTracks) {
  //   // Loop over the Sim tracks
  //   SimTrackContainer::const_iterator simTrack;

  //   SimTrack result;
  //   int mu=0;
  //   for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
  //     if (abs((*simTrack).type()) == 13) {
  //       result = *simTrack;
  //       ++mu;
  //     }

  //   if(mu != 1) LogTrace("MuonTrackAnalyzer") << "WARNING!! more than 1 simulated muon!!" <<endl;
  //   return result;

  // Loop over the Sim tracks
  SimTrackContainer::const_iterator simTrack;

  SimTrack result;

  double bestDeltaR = 10e5;
  for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack) {
    if (abs((*simTrack).type()) != 13)
      continue;

    //    double newDeltaR = tsos.globalMomentum().basicVector().deltaR(simTrack->momentum().vect());
    GlobalVector tsosVect = tsos.globalMomentum();
    math::XYZVectorD vect(tsosVect.x(), tsosVect.y(), tsosVect.z());
    double newDeltaR = deltaR<double>(vect.eta(), vect.phi(), simTrack->momentum().eta(), simTrack->momentum().phi());

    if (newDeltaR < bestDeltaR) {
      LogTrace("MuonTrackAnalyzer") << "Matching Track with DeltaR = " << newDeltaR << endl;
      bestDeltaR = newDeltaR;
      result = *simTrack;
    }
  }
  return pair<SimTrack, double>(result, bestDeltaR);
}

bool MuonTrackAnalyzer::isInTheAcceptance(double eta) {
  switch (theEtaRange) {
    case all:
      return (abs(eta) <= 2.4) ? true : false;
    case barrel:
      return (abs(eta) < 1.1) ? true : false;
    case endcap:
      return (abs(eta) >= 1.1 && abs(eta) <= 2.4) ? true : false;
    default: {
      LogTrace("MuonTrackAnalyzer") << "No correct Eta range selected!! " << endl;
      return false;
    }
  }
}

bool MuonTrackAnalyzer::checkMuonSimHitPresence(const Event &event, edm::Handle<edm::SimTrackContainer> simTracks) {
  // Get the SimHit collection from the event
  Handle<PSimHitContainer> dtSimHits;
  event.getByToken(theDTSimHitToken, dtSimHits);

  Handle<PSimHitContainer> cscSimHits;
  event.getByToken(theCSCSimHitToken, cscSimHits);

  Handle<PSimHitContainer> rpcSimHits;
  event.getByToken(theRPCSimHitToken, rpcSimHits);

  map<unsigned int, vector<const PSimHit *> > mapOfMuonSimHits;

  for (PSimHitContainer::const_iterator simhit = dtSimHits->begin(); simhit != dtSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13)
      continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }

  for (PSimHitContainer::const_iterator simhit = cscSimHits->begin(); simhit != cscSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13)
      continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }

  for (PSimHitContainer::const_iterator simhit = rpcSimHits->begin(); simhit != rpcSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13)
      continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }

  bool presence = false;

  for (SimTrackContainer::const_iterator simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack) {
    if (abs(simTrack->type()) != 13)
      continue;

    map<unsigned int, vector<const PSimHit *> >::const_iterator mapIterator =
        mapOfMuonSimHits.find(simTrack->trackId());

    if (mapIterator != mapOfMuonSimHits.end())
      presence = true;
  }

  return presence;
}

TrajectoryStateOnSurface MuonTrackAnalyzer::getSeedTSOS(const TrajectorySeed &seed) {
  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();

  // Transform it in a TrajectoryStateOnSurface

  DetId seedDetId(pTSOD.detId());

  const GeomDet *gdet = theService->trackingGeometry()->idToDet(seedDetId);

  TrajectoryStateOnSurface initialState =
      trajectoryStateTransform::transientState(pTSOD, &(gdet->surface()), &*theService->magneticField());

  // Get the layer on which the seed relies
  const DetLayer *initialLayer = theService->detLayerGeometry()->idToLayer(seedDetId);

  PropagationDirection detLayerOrder = oppositeToMomentum;

  // ask for compatible layers
  vector<const DetLayer *> detLayers;
  detLayers =
      theService->muonNavigationSchool()->compatibleLayers(*initialLayer, *initialState.freeState(), detLayerOrder);

  TrajectoryStateOnSurface result = initialState;
  if (!detLayers.empty()) {
    const DetLayer *finalLayer = detLayers.back();
    const TrajectoryStateOnSurface propagatedState =
        theService->propagator(theSeedPropagatorName)->propagate(initialState, finalLayer->surface());
    if (propagatedState.isValid())
      result = propagatedState;
  }

  return result;
}

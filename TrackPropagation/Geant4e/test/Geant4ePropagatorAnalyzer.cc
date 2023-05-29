#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"  //For define_fwk_module

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//- Timing
//#include "Utilities/Timing/interface/TimingReport.h"

//- Geometry
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//- Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"

//- Propagator
#include "TrackPropagation/Geant4e/interface/ConvertFromToCLHEP.h"
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

//- SimHits, Tracks and Vertices
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

//- Geant4
#include "G4TransportationManager.hh"

//- ROOT
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

using namespace std;

enum testMuChamberType { DT, RPC, CSC };

class Geant4ePropagatorAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit Geant4ePropagatorAnalyzer(const edm::ParameterSet &);
  ~Geant4ePropagatorAnalyzer() override = default;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  void beginJob() override;
  void iterateOverHits(edm::Handle<edm::PSimHitContainer> simHits,
                       testMuChamberType muonChamberType,
                       unsigned int trkIndex,
                       const FreeTrajectoryState &ftsTrack);

protected:
  // event setup tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken;

  int theRun;
  int theEvent;

  Propagator *thePropagator;
  // std::auto_ptr<sim::FieldBuilder> theFieldBuilder;

  // Geometry
  edm::ESHandle<DTGeometry> theDTGeomESH;    // DTs
  edm::ESHandle<RPCGeometry> theRPCGeomESH;  // RPC
  edm::ESHandle<CSCGeometry> theCSCGeomESH;  // CSC

  // Station that we want to study. -1 for all
  int fStudyStation;

  // Single muons Beam direction (phi) and interval to plot simhits
  float fBeamCenter;
  float fBeamInterval;

  // Histograms and ROOT stuff
  TFile *theRootFile;

  TH1F *fDistanceSHLayer;

  TH1F *fDistance;
  TH1F *fDistanceSt[4];

  TH1F *fHitR;
  TH1F *fHitRho;
  TH1F *fHitEta;
  TH1F *fHitPhi;
  TH1F *fHitPhi1L;
  TH2F *fHitRVsPhi;

  TH1F *fLayerR;
  TH1F *fLayerRho;
  TH1F *fLayerEta;
  TH1F *fLayerPhi;
  TH2F *fLayerRVsPhi;

  TH1F *fExtrapR;
  TH1F *fExtrapRho;
  TH1F *fExtrapEta;
  TH1F *fExtrapPhi;
  TH2F *fExtrapRVsPhi;

  TH1F *fDeltaRo;
  TH1F *fDeltaEta;
  TH1F *fDeltaPhi;

  // Studies on Phi distribution
  TH1F *fStationPosPhi;
  TH1F *fSectorPosPhi;
  TH1F *fSLayerPosPhi;
  TH1F *fLayerPosPhi;

  TH1F *fStationNegPhi;
  TH1F *fSectorNegPhi;
  TH1F *fSLayerNegPhi;
  TH1F *fLayerNegPhi;

  // event data tokens
  edm::InputTag G4TrkSrc_;
  edm::InputTag G4VtxSrc_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
  const edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken_;
  const edm::EDGetTokenT<edm::PSimHitContainer> simHitsDTToken_;
  const edm::EDGetTokenT<edm::PSimHitContainer> simHitsCSCToken_;
  const edm::EDGetTokenT<edm::PSimHitContainer> simHitsRPCToken_;
};

Geant4ePropagatorAnalyzer::Geant4ePropagatorAnalyzer(const edm::ParameterSet &iConfig)
    : magFieldToken(esConsumes()),
      dtGeomToken(esConsumes()),
      rpcGeomToken(esConsumes()),
      cscGeomToken(esConsumes()),
      theRun(-1),
      theEvent(-1),
      thePropagator(nullptr),
      G4TrkSrc_(iConfig.getParameter<edm::InputTag>("G4TrkSrc")),
      G4VtxSrc_(iConfig.getParameter<edm::InputTag>("G4VtxSrc")),
      simTrackToken_(consumes<edm::SimTrackContainer>(G4TrkSrc_)),
      simVertexToken_(consumes<edm::SimVertexContainer>(G4VtxSrc_)),
      simHitsDTToken_(consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits", "MuonDTHits"))),
      simHitsCSCToken_(consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits", "MuonCSCHits"))),
      simHitsRPCToken_(consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits", "MuonRPCHits"))) {
  // debug_ = iConfig.getParameter<bool>("debug");
  fStudyStation = iConfig.getParameter<int>("StudyStation");

  fBeamCenter = iConfig.getParameter<double>("BeamCenter");
  fBeamInterval = iConfig.getParameter<double>("BeamInterval");

  ///////////////////////////////////////////////////////////////////////////////////
  // Histograms
  //

  // Output ROOT file
  theRootFile = new TFile(iConfig.getParameter<std::string>("RootFile").c_str(), "recreate");

  // Distance between Sim Hit and associated Layer
  fDistanceSHLayer = new TH1F("fDistanceSHLayer", "Distance(sim hit - layer)", 200, -1, 1);

  // Distance between simhit and extrapolation
  fDistance = new TH1F("fDistance", "Distance(sim hit - extrap)", 150, 0, 300);
  // Distance between simhit and extrapolation
  fDistanceSt[0] = new TH1F("fDistance_St1", "Distance(sim hit - extrap) for station 1", 150, 0, 300);
  fDistanceSt[1] = new TH1F("fDistance_St2", "Distance(sim hit - extrap) for station 2", 150, 0, 300);
  fDistanceSt[2] = new TH1F("fDistance_St3", "Distance(sim hit - extrap) for station 3", 150, 0, 300);
  fDistanceSt[3] = new TH1F("fDistance_St4", "Distance(sim hit - extrap) for station 4", 150, 0, 300);

  // Simulated hits
  fHitR = new TH1F("fHitR", "R^{sim hit}", 300, 400, 1000);
  fHitRho = new TH1F("fHitRho", "#rho^{sim hit}", 300, 400, 1000);
  fHitEta = new TH1F("fHitEta", "#eta^{sim hit}", 100, -0.1, 0.1);
  fHitPhi = new TH1F("fHitPhi", "#varphi^{sim hit}", 160, fBeamCenter - fBeamInterval, fBeamCenter + fBeamInterval);
  fHitPhi1L = new TH1F(
      "fHitPhi1L", "#varphi^{sim hit} for 1st layer", 160, fBeamCenter - fBeamInterval, fBeamCenter + fBeamInterval);

  fHitRVsPhi = new TH2F("fHitRVsPhi",
                        "R vs. #varphi for sim hits",
                        60,
                        400,
                        1000,
                        80,
                        fBeamCenter - fBeamInterval,
                        fBeamCenter + fBeamInterval);

  // Position of layers
  fLayerR = new TH1F("fLayerR", "R^{layer}", 300, 400, 1000);
  fLayerRho = new TH1F("fLayerRho", "#rho^{layer}", 300, 400, 1000);
  fLayerEta = new TH1F("fLayerEta", "#eta^{layer}", 100, -0.1, 0.1);
  fLayerPhi = new TH1F("fLayerPhi", "#varphi^{layer}", 160, fBeamCenter - fBeamInterval, fBeamCenter + fBeamInterval);
  fLayerRVsPhi = new TH2F("fLayerRVsPhi",
                          "R vs. #varphi for layers",
                          60,
                          400,
                          1000,
                          80,
                          fBeamCenter - fBeamInterval,
                          fBeamCenter + fBeamInterval);

  // Extrapolated hits
  fExtrapR = new TH1F("fExtrapR", "R^{extrap. hit}", 300, 400, 1000);
  fExtrapRho = new TH1F("fExtrapRho", "#rho^{extrap. hit}", 300, 400, 1000);
  fExtrapEta = new TH1F("fExtrapEta", "#eta^{extrap. hit}", 100, -0.1, 0.1);
  fExtrapPhi =
      new TH1F("fExtrapPhi", "#varphi^{extrap. hit}", 160, fBeamCenter - fBeamInterval, fBeamCenter + fBeamInterval);

  fExtrapRVsPhi = new TH2F("fExtrapRVsPhi",
                           "R vs. #varphi for extrap. hits",
                           60,
                           400,
                           1000,
                           80,
                           fBeamCenter - fBeamInterval,
                           fBeamCenter + fBeamInterval);

  // Distances
  fDeltaRo = new TH1F("fDeltaRo", "#Delta(#rho^{sim}, #rho^{extrap})", 100, 0, 200);
  fDeltaEta = new TH1F("fDeltaEta", "#Delta(#eta^{sim}, #eta^{extrap})", 100, -1, 1);
  fDeltaPhi = new TH1F("fDeltaPhi", "#Delta(#varphi^{sim}, #varphi^{extrap})", 120, -30, 30);

  // Studies on Phi
  fStationPosPhi = new TH1F("fStationPosPhi", "Station with positive #varphi", 6, -0.5, 5.5);
  fSectorPosPhi = new TH1F("fSectorPosPhi", "Sector with positive #varphi", 6, -0.5, 5.5);
  fSLayerPosPhi = new TH1F("fSLayerPosPhi", "Superlayer with positive #varphi", 6, -0.5, 5.5);
  fLayerPosPhi = new TH1F("fLayerPosPhi", "Layer with positive #varphi", 6, -0.5, 5.5);
  fStationNegPhi = new TH1F("fStationNegPhi", "Station with negative #varphi", 6, -0.5, 5.5);
  fSectorNegPhi = new TH1F("fSectorNegPhi", "Sector with negative #varphi", 6, -0.5, 5.5);
  fSLayerNegPhi = new TH1F("fSLayerNegPhi", "Superlayer with negative #varphi", 6, -0.5, 5.5);
  fLayerNegPhi = new TH1F("fLayerNegPhi", "Layer with negative #varphi", 6, -0.5, 5.5);

  //
  ///////////////////////////////////////////////////////////////////////////////////
}

void Geant4ePropagatorAnalyzer::beginJob() { LogDebug("Geant4e") << "Nothing done in beginJob..."; }

void Geant4ePropagatorAnalyzer::endJob() {
  fDistanceSHLayer->Write();
  fDistance->Write();
  for (unsigned int i = 0; i < 4; i++)
    fDistanceSt[i]->Write();

  fHitR->Write();
  fHitRho->Write();
  fHitEta->Write();
  fHitPhi->Write();
  fHitPhi1L->Write();
  fHitRVsPhi->Write();

  fLayerR->Write();
  fLayerRho->Write();
  fLayerEta->Write();
  fLayerPhi->Write();
  fLayerRVsPhi->Write();

  fExtrapR->Write();
  fExtrapRho->Write();
  fExtrapEta->Write();
  fExtrapPhi->Write();
  fExtrapRVsPhi->Write();

  fDeltaRo->Write();
  fDeltaEta->Write();
  fDeltaPhi->Write();

  // Phi studies
  fStationPosPhi->Write();
  fSectorPosPhi->Write();
  fSLayerPosPhi->Write();
  fLayerPosPhi->Write();
  fStationNegPhi->Write();
  fSectorNegPhi->Write();
  fSLayerNegPhi->Write();
  fLayerNegPhi->Write();

  theRootFile->Close();
  //  TimingReport::current()->dump(std::cout);
}

void Geant4ePropagatorAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  LogDebug("Geant4e") << "Starting analyze...";

  ///////////////////////////////////////
  // Construct Magnetic Field
  const ESHandle<MagneticField> bField = iSetup.getHandle(magFieldToken);
  if (bField.isValid())
    LogDebug("Geant4e") << "G4e -- Magnetic field is valid. Value in (0,0,0): "
                        << bField->inTesla(GlobalPoint(0, 0, 0)).mag() << " Tesla";
  else
    LogError("Geant4e") << "G4e -- NO valid Magnetic field";

  ///////////////////////////////////////
  // Build geometry

  //- DT...
  theDTGeomESH = iSetup.getHandle(dtGeomToken);
  LogDebug("Geant4e") << "Got DTGeometry";

  //- CSC...
  theCSCGeomESH = iSetup.getHandle(cscGeomToken);
  LogDebug("Geant4e") << "Got CSCGeometry";

  //- RPC...
  theRPCGeomESH = iSetup.getHandle(rpcGeomToken);
  LogDebug("Geant4e") << "Got RPCGeometry";

  ///////////////////////////////////////
  // Run/Event information
  theRun = (int)iEvent.id().run();
  theEvent = (int)iEvent.id().event();
  LogDebug("Geant4e") << "G4e -- Begin for run/event ==" << theRun << "/" << theEvent
                      << " ---------------------------------";

  ///////////////////////////////////////
  // Initialise the propagator
  if (!thePropagator)
    thePropagator = new Geant4ePropagator(&*bField);

  if (thePropagator)
    LogDebug("Geant4e") << "Propagator built!";
  else
    LogError("Geant4e") << "Could not build propagator!";

  ///////////////////////////////////////
  // Get the sim tracks & vertices
  Handle<SimTrackContainer> simTracks = iEvent.getHandle(simTrackToken_);
  if (!simTracks.isValid()) {
    LogWarning("Geant4e") << "No tracks found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "G4e -- Got simTracks of size " << simTracks->size();

  Handle<SimVertexContainer> simVertices = iEvent.getHandle(simVertexToken_);
  if (!simVertices.isValid()) {
    LogWarning("Geant4e") << "No vertices found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got simVertices of size " << simVertices->size();

  ///////////////////////////////////////
  // Get the sim hits for the different muon parts
  Handle<PSimHitContainer> simHitsDT = iEvent.getHandle(simHitsDTToken_);
  if (!simHitsDT.isValid()) {
    LogWarning("Geant4e") << "No hits found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got MuonDTHits of size " << simHitsDT->size();

  Handle<PSimHitContainer> simHitsCSC = iEvent.getHandle(simHitsCSCToken_);
  if (!simHitsCSC.isValid()) {
    LogWarning("Geant4e") << "No hits found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got MuonCSCHits of size " << simHitsCSC->size();

  Handle<PSimHitContainer> simHitsRPC = iEvent.getHandle(simHitsRPCToken_);
  if (!simHitsRPC.isValid()) {
    LogWarning("Geant4e") << "No hits found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got MuonRPCHits of size " << simHitsRPC->size();

  ///////////////////////////////////////
  // Iterate over sim tracks to build the FreeTrajectoryState for
  // for the initial position.
  // DEBUG
  unsigned int counter = 0;
  // DEBUG
  for (SimTrackContainer::const_iterator simTracksIt = simTracks->begin(); simTracksIt != simTracks->end();
       simTracksIt++) {
    // DEBUG
    counter++;
    LogDebug("Geant4e") << "G4e -- Iterating over " << counter << "rd track. Number: " << simTracksIt->genpartIndex()
                        << "------------------";
    // DEBUG

    int simTrackPDG = simTracksIt->type();
    if (abs(simTrackPDG) != 13) {
      continue;
    }
    LogDebug("Geant4e") << "G4e -- Track PDG " << simTrackPDG;

    //- Timing
    //    TimeMe tProp("Geant4ePropagatorAnalyzer::analyze::propagate");

    //- Check if the track corresponds to a muon
    int trkPDG = simTracksIt->type();
    if (abs(trkPDG) != 13) {
      LogDebug("Geant4e") << "Track is not a muon: " << trkPDG;
      continue;
    } else
      LogDebug("Geant4e") << "Found a muon track " << trkPDG;

    //- Get momentum, but only use tracks with P > 2 GeV
    GlobalVector p3T = TrackPropagation::hep3VectorToGlobalVector(
        CLHEP::Hep3Vector(simTracksIt->momentum().x(), simTracksIt->momentum().y(), simTracksIt->momentum().z()));
    if (p3T.perp() < 2.) {
      LogDebug("Geant4e") << "Track PT is too low: " << p3T.perp();
      continue;
    } else {
      LogDebug("Geant4e") << "*** Phi (rad): " << p3T.phi() << " - Phi(deg)" << p3T.phi().degrees();
      LogDebug("Geant4e") << "Track PT is enough.";
      LogDebug("Geant4e") << "Track P.: " << p3T << "\nTrack P.: PT=" << p3T.perp() << "\tEta=" << p3T.eta()
                          << "\tPhi=" << p3T.phi().degrees() << "--> Rad: Phi=" << p3T.phi();
    }

    //- Vertex fixes the starting point
    int vtxInd = simTracksIt->vertIndex();
    GlobalPoint r3T(0., 0., 0.);
    if (vtxInd < 0)
      LogDebug("Geant4e") << "Track with no vertex, defaulting to (0,0,0)";
    else
      // seems to be stored in mm --> convert to cm
      r3T = TrackPropagation::hep3VectorToGlobalPoint(CLHEP::Hep3Vector((*simVertices)[vtxInd].position().x(),
                                                                        (*simVertices)[vtxInd].position().y(),
                                                                        (*simVertices)[vtxInd].position().z()));

    LogDebug("Geant4e") << "Init point: " << r3T << "\nInit point Ro=" << r3T.perp() << "\tEta=" << r3T.eta()
                        << "\tPhi=" << r3T.phi().degrees();

    //- Charge
    int charge = trkPDG > 0 ? -1 : 1;
    LogDebug("Geant4e") << "Track charge = " << charge;

    //- Initial covariance matrix is unity 10-6
    CurvilinearTrajectoryError covT;
    // covT *= 1E-6;

    //- Build FreeTrajectoryState
    GlobalTrajectoryParameters trackPars(r3T, p3T, charge, &*bField);
    FreeTrajectoryState ftsTrack(trackPars, covT);

    //- Get index of generated particle. Used further down
    unsigned int trkInd = simTracksIt->genpartIndex();

    ////////////////////////////////////////////////
    //- Iterate over Sim Hits in DT and check propagation
    iterateOverHits(simHitsDT, DT, trkInd, ftsTrack);
    ////////////////////////////////////////////////
    //- Iterate over Sim Hits in RPC and check propagation
    // iterateOverHits(simHitsRPC, RPC, trkInd, ftsTrack);
    ////////////////////////////////////////////////
    //- Iterate over Sim Hits in CSC and check propagation
    // iterateOverHits(simHitsCSC, CSC, trkInd, ftsTrack);

  }  // <-- for over sim tracks
}

void Geant4ePropagatorAnalyzer::iterateOverHits(edm::Handle<edm::PSimHitContainer> simHits,
                                                testMuChamberType muonChamberType,
                                                unsigned int trkIndex,
                                                const FreeTrajectoryState &ftsTrack) {
  using namespace edm;

  if (muonChamberType == DT)
    LogDebug("Geant4e") << "G4e -- Iterating over DT hits...";
  else if (muonChamberType == RPC)
    LogDebug("Geant4e") << "G4e -- Iterating over RPC hits...";
  else if (muonChamberType == CSC)
    LogDebug("Geant4e") << "G4e -- Iterating over CSC hits...";

  for (PSimHitContainer::const_iterator simHitIt = simHits->begin(); simHitIt != simHits->end(); simHitIt++) {
    ///////////////
    // Skip if this hit does not belong to the track
    if (simHitIt->trackId() != trkIndex) {
      LogDebug("Geant4e") << "Hit (in tr " << simHitIt->trackId() << ") does not belong to track " << trkIndex;
      continue;
    }

    LogDebug("Geant4e") << "G4e -- Hit belongs to track " << trkIndex;

    //////////////
    // Skip if it is not a muon (this is checked before also)
    int trkPDG = simHitIt->particleType();
    if (abs(trkPDG) != 13) {
      LogDebug("Geant4e") << "Associated track is not a muon: " << trkPDG;
      continue;
    }
    LogDebug("Geant4e") << "G4e -- Found a hit corresponding to a muon " << trkPDG;

    //////////////////////////////////////////////////////////
    // Build the surface. This is different for DT, RPC, CSC
    // const GeomDetUnit* layer = 0;
    const GeomDet *layer = nullptr;
    // * DT
    DTWireId *wIdDT = nullptr;  // For DT holds the information about the chamber
    if (muonChamberType == DT) {
      wIdDT = new DTWireId(simHitIt->detUnitId());
      int station = wIdDT->station();
      LogDebug("Geant4e") << "G4e -- DT Chamber. Station: " << station << ". DetId: " << wIdDT->layerId();

      if (fStudyStation != -1 && fStudyStation != station)
        continue;

      layer = theDTGeomESH->layer(*wIdDT);
      if (layer == nullptr) {
        LogDebug("Geant4e") << "Failed to get detector unit";
        continue;
      }
    }
    // * RPC
    else if (muonChamberType == RPC) {
      RPCDetId rpcId(simHitIt->detUnitId());
      layer = theRPCGeomESH->idToDet(rpcId);
      if (layer == nullptr) {
        LogDebug("Geant4e") << "Failed to get detector unit";
        continue;
      }
    }

    // * CSC
    else if (muonChamberType == CSC) {
      CSCDetId cscId(simHitIt->detUnitId());
      layer = theCSCGeomESH->idToDet(cscId);
      if (layer == nullptr) {
        LogDebug("Geant4e") << "Failed to get detector unit";
        continue;
      }
    }

    const Surface &surf = layer->surface();
    if (layer->geographicalId() != DTLayerId(simHitIt->detUnitId()))
      LogError("Geant4e") << "ERROR: wrong DetId";

    //==>DEBUG
    // const BoundPlane& bp = layer->surface();
    // const Bounds& bounds = bp.bounds();
    // LogDebug("Geant4e") << "Surface: length = " << bounds.length()
    //		  << ", thickness = " << bounds.thickness()
    //		<< ", width = " << bounds.width();
    //<==DEBUG

    ////////////
    // Discard hits with very low momentum ???
    GlobalVector p3Hit = surf.toGlobal(simHitIt->momentumAtEntry());
    if (p3Hit.perp() < 0.5)
      continue;
    GlobalPoint posHit = surf.toGlobal(simHitIt->localPosition());
    Point3DBase<float, GlobalTag> surfpos = surf.position();
    LogDebug("Geant4e") << "G4e -- Layer position: " << surfpos << " cm"
                        << "\nG4e --                   Ro=" << surfpos.perp() << "\tEta=" << surfpos.eta()
                        << "\tPhi=" << surfpos.phi().degrees() << "deg";
    LogDebug("Geant4e") << "G4e -- Sim Hit position: " << posHit << " cm"
                        << "\nG4e --                   Ro=" << posHit.perp() << "\tEta=" << posHit.eta()
                        << "\tPhi=" << posHit.phi().degrees() << "deg"
                        << "\t localpos=" << simHitIt->localPosition();

    const Plane *bp = dynamic_cast<const Plane *>(&surf);
    if (bp != nullptr) {
      Float_t distance = bp->localZ(posHit);
      LogDebug("Geant4e") << "\nG4e -- Distance from plane to sim hit: " << distance << "cm";
      fDistanceSHLayer->Fill(distance);
    } else {
      LogWarning("Geant4e") << "G4e -- Layer is not a Plane!!!";
      fDistanceSHLayer->Fill(-1);
    }

    LogDebug("Geant4e") << "Sim Hit Momentum PT=" << p3Hit.perp() << "\tEta=" << p3Hit.eta()
                        << "\tPhi=" << p3Hit.phi().degrees() << "deg";

    /////////////////////////////////////////
    // Propagate: Need to explicetely
    TrajectoryStateOnSurface tSOSDest = thePropagator->propagate(ftsTrack, surf);

    /////////////////////
    // Get hit position and extrapolation position to compare
    GlobalPoint posExtrap = tSOSDest.freeState()->position();
    //     GlobalPoint posExtrap(posExtrap_prov.theta(),
    // 			  posExtrap_prov.phi()*12,
    // 			  posExtrap_prov.mag());

    GlobalVector posDistance = posExtrap - posHit;
    float distance = posDistance.mag();
    float simhitphi = posHit.phi().degrees();
    float layerphi = surfpos.phi().degrees();
    float extrapphi = posExtrap.phi().degrees();
    LogDebug("Geant4e") << "G4e -- Difference between hit and final position: " << distance << " cm\n"
                        << "G4e -- Transversal difference between hit and final position: " << posDistance.perp()
                        << " cm.";
    LogDebug("Geant4e") << "G4e -- Extrapolated position:" << posExtrap << " cm\n"
                        << "G4e --       (Rho, eta, phi): (" << posExtrap.perp() << " cm, " << posExtrap.eta() << ", "
                        << extrapphi << " deg)";
    LogDebug("Geant4e") << "G4e --          Hit position: " << posHit << " cm\n"
                        << "G4e --       (Rho, eta, phi): (" << posHit.perp() << " cm, " << posHit.eta() << ", "
                        << simhitphi << " deg)";

    fDistance->Fill(distance);
    fDistanceSt[wIdDT->station() - 1]->Fill(distance);

    fHitR->Fill(posHit.mag());
    fHitRho->Fill(posHit.perp());
    fHitEta->Fill(posHit.eta());
    fHitPhi->Fill(simhitphi);
    if (posHit.perp() < 500)
      fHitPhi1L->Fill(simhitphi);
    fHitRVsPhi->Fill(posHit.mag(), simhitphi);

    fLayerR->Fill(surfpos.mag());
    fLayerRho->Fill(surfpos.perp());
    fLayerEta->Fill(surfpos.eta());
    fLayerPhi->Fill(layerphi);
    fLayerRVsPhi->Fill(surfpos.mag(), layerphi);

    fExtrapR->Fill(posExtrap.mag());
    fExtrapRho->Fill(posExtrap.perp());
    fExtrapEta->Fill(posExtrap.eta());
    fExtrapPhi->Fill(extrapphi);
    fExtrapRVsPhi->Fill(posExtrap.mag(), extrapphi);

    fDeltaRo->Fill(posExtrap.perp() - posHit.perp());
    fDeltaEta->Fill(posExtrap.eta() - posHit.eta());
    fDeltaPhi->Fill(extrapphi - simhitphi);

    if (wIdDT) {
      if (simhitphi > 0) {
        fStationPosPhi->Fill(wIdDT->station());
        fSectorPosPhi->Fill(wIdDT->sector());
        fSLayerPosPhi->Fill(wIdDT->superlayer());
        fLayerPosPhi->Fill(wIdDT->layer());
      } else {
        fStationNegPhi->Fill(wIdDT->station());
        fSectorNegPhi->Fill(wIdDT->sector());
        fSLayerNegPhi->Fill(wIdDT->superlayer());
        fLayerNegPhi->Fill(wIdDT->layer());
      }
      // Some cleaning...
      delete wIdDT;
    }

  }  //<== For over simhits
}

// define this as a plug-in
DEFINE_FWK_MODULE(Geant4ePropagatorAnalyzer);

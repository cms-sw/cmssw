#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h" //For define_fwk_module

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//- Timing
#include "Utilities/Timing/interface/TimingReport.h"

//- Geometry
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

//- Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"

//- Propagator
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"
#include "TrackPropagation/Geant4e/interface/ConvertFromToCLHEP.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


//- SimHits, Tracks and Vertices
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

//- Geant4
#include "G4TransportationManager.hh"

//- ROOT
#include "TMath.h"

//#include <iostream>

using namespace std;

enum testMuChamberType {DT, RPC, CSC};

class Geant4ePropagatorAnalyzer: public edm::EDAnalyzer {

public:
  explicit Geant4ePropagatorAnalyzer(const edm::ParameterSet&);
  virtual ~Geant4ePropagatorAnalyzer() {}

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void beginJob(edm::EventSetup const & iSetup);
  void iterateOverHits(edm::Handle<edm::PSimHitContainer> simHits,
		       testMuChamberType muonChamberType,
		       unsigned int trkIndex,
		       const FreeTrajectoryState& ftsTrack);


protected:

  int theRun;
  int theEvent;

  Propagator* thePropagator;
  //std::auto_ptr<sim::FieldBuilder> theFieldBuilder;

  //Magnetic field
  edm::ParameterSet theMagneticFieldPSet;

  //Geometry
  edm::ESHandle<DTGeometry> theDTGeomESH; //DTs
  edm::ESHandle<RPCGeometry> theRPCGeomESH; //RPC
  edm::ESHandle<CSCGeometry> theCSCGeomESH; //CSC
  


};


Geant4ePropagatorAnalyzer::Geant4ePropagatorAnalyzer(const edm::ParameterSet& p):
  theRun(-1),
  theEvent(-1),
  thePropagator(0) {

  //debug_ = iConfig.getParameter<bool>("debug");
  theMagneticFieldPSet = p.getParameter<edm::ParameterSet>("MagneticField");

}

void Geant4ePropagatorAnalyzer::beginJob(edm::EventSetup const & iSetup) {
  LogDebug("Geant4e") << "Nothing done in beginJob...";
}

void Geant4ePropagatorAnalyzer::endJob() {
  
  TimingReport::current()->dump(std::cout);
}

void Geant4ePropagatorAnalyzer::analyze(const edm::Event& iEvent, 
					const edm::EventSetup& iSetup) {

  using namespace edm;

  LogDebug("Geant4e") << "Starting analyze...";

  ///////////////////////////////////////
  //Construct Magnetic Field
  ESHandle<MagneticField> bField;
  iSetup.get<IdealMagneticFieldRecord>().get(bField);


  ///////////////////////////////////////
  //Build geometry

  //- DT...
  iSetup.get<MuonGeometryRecord>().get(theDTGeomESH);
  LogDebug("Geant4e") << "Got DTGeometry";

  //- CSC...
  iSetup.get<MuonGeometryRecord>().get(theCSCGeomESH);
  LogDebug("Geant4e") << "Got CSCGeometry";

  //- RPC...
  iSetup.get<MuonGeometryRecord>().get(theRPCGeomESH);
  LogDebug("Geant4e") << "Got RPCGeometry";


  ///////////////////////////////////////
  //Run/Event information
  theRun = (int)iEvent.id().run();
  theEvent = (int)iEvent.id().event();
  LogDebug("Geant4e") << "Begin for run:event ==" << theRun << ":" << theEvent;


  ///////////////////////////////////////
  //Initialise the propagator
  if (! thePropagator) 
    thePropagator = new Geant4ePropagator(&*bField);

  if (thePropagator)
    LogDebug("Geant4e") << "Propagator built!";
  else
    LogError("Geant4e") << "Could not build propagator!";





  ///////////////////////////////////////
  //Get the sim tracks & vertices 
  Handle<SimTrackContainer> simTracks;
  iEvent.getByType<SimTrackContainer>(simTracks);
  if (! simTracks.isValid() ){
    LogWarning("Geant4e") << "No tracks found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got simTracks of size " << simTracks->size();

  Handle<SimVertexContainer> simVertices;
  iEvent.getByType<SimVertexContainer>(simVertices);
  if (! simVertices.isValid() ){
    LogWarning("Geant4e") << "No tracks found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got simVertices of size " << simVertices->size();


  ///////////////////////////////////////
  //Get the sim hits for the different muon parts
  Handle<PSimHitContainer> simHitsDT;
  iEvent.getByLabel("g4SimHits", "MuonDTHits", simHitsDT);
  if (! simHitsDT.isValid() ){
    LogWarning("Geant4e") << "No hits found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got MuonDTHits of size " << simHitsDT->size();

  Handle<PSimHitContainer> simHitsCSC;
  iEvent.getByLabel("g4SimHits", "MuonCSCHits", simHitsCSC);
  if (! simHitsCSC.isValid() ){
    LogWarning("Geant4e") << "No hits found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got MuonCSCHits of size " << simHitsCSC->size();


  Handle<PSimHitContainer> simHitsRPC;
  iEvent.getByLabel("g4SimHits", "MuonRPCHits", simHitsRPC);
  if (! simHitsRPC.isValid() ){
    LogWarning("Geant4e") << "No hits found" << std::endl;
    return;
  }
  LogDebug("Geant4e") << "Got MuonRPCHits of size " << simHitsRPC->size();



  ///////////////////////////////////////
  // Iterate over sim tracks to build the FreeTrajectoryState for
  // for the initial position.
  //DEBUG
  unsigned int counter = 0;
  //DEBUG
  for(SimTrackContainer::const_iterator simTracksIt = simTracks->begin(); 
      simTracksIt != simTracks->end(); 
      simTracksIt++){

    //DEBUG
    counter++;
    LogDebug("Geant4e") << "G4e -- Iterating over " << counter 
			<< " track. Number: " 
			<< simTracksIt->genpartIndex();
    //DEBUG

    int simTrackPDG = simTracksIt->type();
    if (abs(simTrackPDG) != 13 ) {
      continue;
    }
    LogDebug("Geant4e") << "G4e -- Track PDG " << simTrackPDG;

    //- Timing
    TimeMe tProp("Geant4ePropagatorAnalyzer::analyze::propagate");
   
    //- Check if the track corresponds to a muon
    int trkPDG = simTracksIt->type();
    if (abs(trkPDG) != 13 ) {
      LogDebug("Geant4e") << "Track is not a muon: " << trkPDG;
      continue;
    }
    else
      LogDebug("Geant4e") << "Found a muon track " << trkPDG;
      
    
    //- Get momentum, but only use tracks with P > 2 GeV
    GlobalVector p3T = 
      TrackPropagation::hep3VectorToGlobalVector(simTracksIt->momentum().vect());
    if (p3T.mag() < 2.) {
      LogDebug("Geant4e") << "Track PT is too low: " << p3T.mag();
      continue;
    }
    else {
      LogDebug("Geant4e") << "Track PT is enough.";
      LogDebug("Geant4e") << "Track P.: " << p3T
			  << "\nTrack P.: PT=" << p3T.mag()
			  << "\tTheta=" << p3T.theta()*TMath::RadToDeg()  
			  << "\tPhi=" << p3T.phi()*TMath::RadToDeg()
			  << "--> Rad: Theta=" << p3T.theta() 
			  << ", Phi=" << p3T.phi();

    }
      

    //- Vertex fixes the starting point
    int vtxInd = simTracksIt->vertIndex();
    GlobalPoint r3T(0.,0.,0.);
    if (vtxInd < 0)
      LogDebug("Geant4e") << "Track with no vertex, defaulting to (0,0,0)";
    else
      //seems to be stored in mm --> convert to cm
      r3T = TrackPropagation::hep3VectorToGlobalPoint((*simVertices)[vtxInd].position().vect());

    LogDebug("Geant4e") << "Init point: " << r3T
			<< "\nInit point R=" << r3T.mag()
			<< "\tTheta=" << r3T.theta()*TMath::RadToDeg() 
			<< "\tPhi=" << r3T.phi()*TMath::RadToDeg() ;
    
    //- Charge
    int charge = trkPDG > 0 ? -1 : 1;
    LogDebug("Geant4e") << "Track charge = " << charge;

    //- Initial covariance matrix is unity 10-6
    CurvilinearTrajectoryError covT;
    covT *= 1E-6;

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
    iterateOverHits(simHitsRPC, RPC, trkInd, ftsTrack);
    ////////////////////////////////////////////////
    //- Iterate over Sim Hits in CSC and check propagation
    iterateOverHits(simHitsCSC, CSC, trkInd, ftsTrack);



  } // <-- for over sim tracks
}


void
Geant4ePropagatorAnalyzer::iterateOverHits(edm::Handle<edm::PSimHitContainer> simHits, 
					   testMuChamberType muonChamberType,
					   unsigned int trkIndex,
					   const FreeTrajectoryState& ftsTrack) {

  using namespace edm;

  if (muonChamberType == DT)
    LogDebug("Geant4e") << "G4e -- Iterating over DT hits";
  else if (muonChamberType == RPC)
    LogDebug("Geant4e") << "G4e -- Iterating over RPC hits";
  else if (muonChamberType == CSC)
    LogDebug("Geant4e") << "G4e -- Iterating over CSC hits";
  
  for (PSimHitContainer::const_iterator simHitIt = simHits->begin(); 
       simHitIt != simHits->end(); 
       simHitIt++){

    ///////////////
    // Skip if this hit does not belong to the track
    if (simHitIt->trackId() != trkIndex ) {
      LogDebug("Geant4e") << "Hit (in tr " << simHitIt->trackId()
			  << ") does not belong to track "<< trkIndex;
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
    //const GeomDetUnit* layer = 0;
    const GeomDet* layer = 0;
    // * DT
    if (muonChamberType == DT) {
      DTWireId wId(simHitIt->detUnitId());
      layer = theDTGeomESH->layer(wId);
      if (layer == 0){
	LogDebug("Geant4e") << "Failed to get detector unit";
	continue;
      }
    }
    // * RPC
    else if (muonChamberType == RPC) {
      RPCDetId wId(simHitIt->detUnitId());
      layer = theRPCGeomESH->idToDet(wId);
      if (layer == 0){
	LogDebug("Geant4e") << "Failed to get detector unit";
	continue;
      }
    }

    // * CSC
    else if (muonChamberType ==CSC) {
      CSCDetId wId(simHitIt->detUnitId());
      layer = theCSCGeomESH->idToDet(wId);
      if (layer == 0){
	LogDebug("Geant4e") << "Failed to get detector unit";
	continue;
      }

    }

    const Surface& surf = layer->surface();
    
    //==>DEBUG
    //const BoundPlane& bp = layer->surface();
    //const Bounds& bounds = bp.bounds();
    //LogDebug("Geant4e") << "Surface: length = " << bounds.length() 
    //		  << ", thickness = " << bounds.thickness() 
    //		<< ", width = " << bounds.width();
    //<==DEBUG
    
    ////////////
    // Discard hits with very low momentum ???
    GlobalVector p3Hit = surf.toGlobal(simHitIt->momentumAtEntry());
    if (p3Hit.mag() < 0.5 ) 
      continue;
    GlobalPoint posHit = surf.toGlobal(simHitIt->localPosition());
    Point3DBase< float, GlobalTag > surfpos = surf.position();
    LogDebug("Geant4e") << "Sim Hit position  R=" << posHit.mag()
			<< "\tTheta=" << posHit.theta()*TMath::RadToDeg() 
			<< "\tPhi=" << posHit.phi()*TMath::RadToDeg() ;
    LogDebug("Geant4e") << "Layer position    R=" << surfpos.mag()
			<< "\tTheta=" << surfpos.theta()*TMath::RadToDeg() 
			<< "\tPhi=" << surfpos.phi()*TMath::RadToDeg() ;
    LogDebug("Geant4e") << "Sim Hit Momentum PT=" << p3Hit.mag()
			<< "\tTheta=" << p3Hit.theta()*TMath::RadToDeg() 
			<< "\tPhi=" << p3Hit.phi()*TMath::RadToDeg() ;
    
    
    /////////////////////////////////////////
    // Propagate: Need to explicetely
    TrajectoryStateOnSurface tSOSDest = 
      thePropagator->propagate(ftsTrack, surf);
    
    /////////////////////
    // Get hit position and extrapolation position to compare
    GlobalPoint posExtrap = tSOSDest.freeState()->position();
    
    LogDebug("Geant4e") << "G4e -- Difference between hit and final position: " 
			<< (posExtrap - posHit).mag() << " cm.";
    LogDebug("Geant4e") << "G4e -- Extrapolated position:" << posExtrap 
			<< " cm\n"
			<< "G4e -- Hit position: " << posHit 
			<< " cm";
    
    
  } //<== For over simhits

}

//define this as a plug-in
DEFINE_FWK_MODULE(Geant4ePropagatorAnalyzer);

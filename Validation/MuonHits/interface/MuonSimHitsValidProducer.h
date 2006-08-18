#ifndef MuonSimHitsValidProducer_h
#define MuonSimHitsValidProducer_h

/// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "DataFormats/DetId/interface/DetId.h"

/// muon CSC, DT and RPC geometry info
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

/// muon CSC detector id
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

/// data in edm::event
#include "SimDataFormats/MuonValidation/interface/PMuonSimHit.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

/// helper files
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Units/SystemOfUnits.h>

#include <iostream>

#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"

class MuonSimHitsValidProducer : public edm::EDProducer
{
  
 public:

  typedef std::vector<float> FloatVector;
  typedef std::vector<int>   IntegerVector;

  explicit MuonSimHitsValidProducer(const edm::ParameterSet&);
  virtual ~MuonSimHitsValidProducer();

  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  void fillG4MC(edm::Event&);
  void storeG4MC(PMuonSimHit&);

  void fillCSC(edm::Event&, const edm::EventSetup&);
  void storeCSC(PMuonSimHit&);

  void fillDT(edm::Event&, const edm::EventSetup&);
  void storeDT(PMuonSimHit&);

  void fillRPC(edm::Event&, const edm::EventSetup&);
  void storeRPC(PMuonSimHit&);

  void clear();

 private:

  ///  parameter information
  std::string fName;
  int verbosity;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  /// G4MC info
  int nRawGenPart;
  FloatVector G4VtxX; 
  FloatVector G4VtxY; 
  FloatVector G4VtxZ;
 
  FloatVector G4TrkPt; 
  FloatVector G4TrkE;
  FloatVector G4TrkEta;
  FloatVector G4TrkPhi;

  /// CSC info

  IntegerVector CSCHitsId;
  FloatVector   CSCHitsDetUnId;
  FloatVector   CSCHitsTrkId; 
  FloatVector   CSCHitsProcType; 
  FloatVector   CSCHitsPartType; 
  FloatVector   CSCHitsPabs;
  FloatVector   CSCHitsGlobPosZ;
  FloatVector   CSCHitsGlobPosPhi;
  FloatVector   CSCHitsGlobPosEta;
  FloatVector   CSCHitsLocPosX; 
  FloatVector   CSCHitsLocPosY; 
  FloatVector   CSCHitsLocPosZ; 
  FloatVector   CSCHitsLocDirX; 
  FloatVector   CSCHitsLocDirY; 
  FloatVector   CSCHitsLocDirZ; 
  FloatVector   CSCHitsLocDirTheta; 
  FloatVector   CSCHitsLocDirPhi;
  FloatVector   CSCHitsExitPointX; 
  FloatVector   CSCHitsExitPointY; 
  FloatVector   CSCHitsExitPointZ;
  FloatVector   CSCHitsEntryPointX; 
  FloatVector   CSCHitsEntryPointY; 
  FloatVector   CSCHitsEntryPointZ;
  FloatVector   CSCHitsEnLoss; 
  FloatVector   CSCHitsTimeOfFlight;

  /// DT info

  FloatVector   DTHitsDetUnId;
  FloatVector   DTHitsTrkId; 
  FloatVector   DTHitsProcType; 
  FloatVector   DTHitsPartType; 
  FloatVector   DTHitsPabs;
  FloatVector   DTHitsGlobPosZ;
  FloatVector   DTHitsGlobPosPhi;
  FloatVector   DTHitsGlobPosEta;
  FloatVector   DTHitsLocPosX; 
  FloatVector   DTHitsLocPosY; 
  FloatVector   DTHitsLocPosZ; 
  FloatVector   DTHitsLocDirX; 
  FloatVector   DTHitsLocDirY; 
  FloatVector   DTHitsLocDirZ; 
  FloatVector   DTHitsLocDirTheta; 
  FloatVector   DTHitsLocDirPhi;
  FloatVector   DTHitsExitPointX; 
  FloatVector   DTHitsExitPointY; 
  FloatVector   DTHitsExitPointZ;
  FloatVector   DTHitsEntryPointX; 
  FloatVector   DTHitsEntryPointY; 
  FloatVector   DTHitsEntryPointZ;
  FloatVector   DTHitsEnLoss; 
  FloatVector   DTHitsTimeOfFlight;

  /// RPC info

  FloatVector   RPCHitsDetUnId;
  FloatVector   RPCHitsTrkId; 
  FloatVector   RPCHitsProcType; 
  FloatVector   RPCHitsPartType; 
  FloatVector   RPCHitsPabs;
  FloatVector   RPCHitsGlobPosZ;
  FloatVector   RPCHitsGlobPosPhi;
  FloatVector   RPCHitsGlobPosEta;
  FloatVector   RPCHitsLocPosX; 
  FloatVector   RPCHitsLocPosY; 
  FloatVector   RPCHitsLocPosZ; 
  FloatVector   RPCHitsLocDirX; 
  FloatVector   RPCHitsLocDirY; 
  FloatVector   RPCHitsLocDirZ; 
  FloatVector   RPCHitsLocDirTheta; 
  FloatVector   RPCHitsLocDirPhi;
  FloatVector   RPCHitsExitPointX; 
  FloatVector   RPCHitsExitPointY; 
  FloatVector   RPCHitsExitPointZ;
  FloatVector   RPCHitsEntryPointX; 
  FloatVector   RPCHitsEntryPointY; 
  FloatVector   RPCHitsEntryPointZ;
  FloatVector   RPCHitsEnLoss; 
  FloatVector   RPCHitsTimeOfFlight;

  /// Input tags

  edm::InputTag CSCHitsSrc_;
  edm::InputTag DTHitsSrc_;
  edm::InputTag RPCHitsSrc_;                                                
  

  /// private statistics information
  unsigned int count;

}; /// end class declaration

/// geometry mapping
 
static const int dMuon            = 2;

static const int sdMuonDT         = 1;
static const int sdMuonCSC        = 2;
static const int sdMuonRPC        = 3;

#endif

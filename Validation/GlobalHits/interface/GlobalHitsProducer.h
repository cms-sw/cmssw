#ifndef GlobalHitsProducer_h
#define GlobalHitsProducer_h

/** \class GlobalHitsProducer
 *  
 *  Class to fill PGlobalSimHit object to be inserted into data stream 
 *  containing information about various sub-systems in global coordinates 
 *  with full geometry
 *
 *  $Date: 2013/02/27 13:28:59 $
 *  $Revision: 1.16 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

// tracker info
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// muon info
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

// calorimeter info
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

// data in edm::event
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
//#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
//#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

// helper files
//#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"

class PGlobalSimHit;
  
class GlobalHitsProducer : public edm::EDProducer
{
  
 public:

  typedef std::vector<float> FloatVector;

  explicit GlobalHitsProducer(const edm::ParameterSet&);
  virtual ~GlobalHitsProducer();
  virtual void beginJob( void );
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
 private:

  //GlobalValidation(const GlobalValidation&);   
  //const GlobalValidation& operator=(const GlobalValidation&);

  // production related methods
  void fillG4MC(edm::Event&);
  void storeG4MC(PGlobalSimHit&);
  void fillTrk(edm::Event&, const edm::EventSetup&);
  void storeTrk(PGlobalSimHit&);
  void fillMuon(edm::Event&, const edm::EventSetup&);
  void storeMuon(PGlobalSimHit&);
  void fillECal(edm::Event&, const edm::EventSetup&);
  void storeECal(PGlobalSimHit&);
  void fillHCal(edm::Event&, const edm::EventSetup&);
  void storeHCal(PGlobalSimHit&);

  void clear();

 private:

  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  int vtxunit;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  // G4MC info
  int nRawGenPart;
  FloatVector G4VtxX; 
  FloatVector G4VtxY; 
  FloatVector G4VtxZ; 
  FloatVector G4TrkPt; 
  FloatVector G4TrkE;

  edm::InputTag G4VtxSrc_;
  edm::InputTag G4TrkSrc_;

  // Electromagnetic info
  // ECal info
  FloatVector ECalE; 
  FloatVector ECalToF; 
  FloatVector ECalPhi; 
  FloatVector ECalEta;
  edm::InputTag ECalEBSrc_;
  edm::InputTag ECalEESrc_;

  // Preshower info
  FloatVector PreShE; 
  FloatVector PreShToF; 
  FloatVector PreShPhi; 
  FloatVector PreShEta;
  edm::InputTag ECalESSrc_;

  // Hadronic info
  // HCal info
  FloatVector HCalE; 
  FloatVector HCalToF; 
  FloatVector HCalPhi; 
  FloatVector HCalEta;
  edm::InputTag HCalSrc_;

  // Tracker info
  // Pixel info
  FloatVector PxlBrlToF; 
  FloatVector PxlBrlR; 
  FloatVector PxlBrlPhi; 
  FloatVector PxlBrlEta; 
  FloatVector PxlFwdToF; 
  FloatVector PxlFwdZ;
  FloatVector PxlFwdPhi; 
  FloatVector PxlFwdEta;
  edm::InputTag PxlBrlLowSrc_;
  edm::InputTag PxlBrlHighSrc_;
  edm::InputTag PxlFwdLowSrc_;
  edm::InputTag PxlFwdHighSrc_;

  // Strip info
  FloatVector SiBrlToF; 
  FloatVector SiBrlR; 
  FloatVector SiBrlPhi; 
  FloatVector SiBrlEta;  
  FloatVector SiFwdToF; 
  FloatVector SiFwdZ;
  FloatVector SiFwdPhi; 
  FloatVector SiFwdEta;
  edm::InputTag SiTIBLowSrc_;
  edm::InputTag SiTIBHighSrc_;
  edm::InputTag SiTOBLowSrc_;
  edm::InputTag SiTOBHighSrc_;
  edm::InputTag SiTIDLowSrc_;
  edm::InputTag SiTIDHighSrc_;
  edm::InputTag SiTECLowSrc_;
  edm::InputTag SiTECHighSrc_;

  // Muon info
  // DT info
  FloatVector MuonDtToF; 
  FloatVector MuonDtR;
  FloatVector MuonDtPhi;
  FloatVector MuonDtEta;
  edm::InputTag MuonDtSrc_;
  // CSC info
  FloatVector MuonCscToF; 
  FloatVector MuonCscZ;
  FloatVector MuonCscPhi;
  FloatVector MuonCscEta;
  edm::InputTag MuonCscSrc_;
  // RPC info
  FloatVector MuonRpcBrlToF; 
  FloatVector MuonRpcBrlR;
  FloatVector MuonRpcBrlPhi;
  FloatVector MuonRpcBrlEta;
  FloatVector MuonRpcFwdToF; 
  FloatVector MuonRpcFwdZ;
  FloatVector MuonRpcFwdPhi;
  FloatVector MuonRpcFwdEta;
  edm::InputTag MuonRpcSrc_;

  // private statistics information
  unsigned int count;

}; // end class declaration
 
#endif
 
#ifndef GlobalHitMap
#define GlobalHitMap
// geometry mapping
static const int dTrk             = 1;
static const int sdPxlBrl         = 1;
static const int sdPxlFwd         = 2;
static const int sdSiTIB          = 3;
static const int sdSiTID          = 4;
static const int sdSiTOB          = 5;
static const int sdSiTEC          = 6;

static const int dMuon            = 2;
static const int sdMuonDT         = 1;
static const int sdMuonCSC        = 2;
static const int sdMuonRPC        = 3;
static const int sdMuonRPCRgnBrl  = 0;
static const int sdMuonRPCRgnFwdp = 1;
static const int sdMuonRPCRgnFwdn = -1;

static const int dEcal            = 3;
static const int sdEcalBrl        = 1;
static const int sdEcalFwd        = 2;
static const int sdEcalPS         = 3;
static const int sdEcalTT         = 4;
static const int sdEcalLPnD       = 5;

static const int dHcal            = 4;
static const int sdHcalEmpty      = 0;
static const int sdHcalBrl        = 1;
static const int sdHcalEC         = 2;
static const int sdHcalOut        = 3;
static const int sdHcalFwd        = 4;
static const int sdHcalTT         = 5;
static const int sdHcalCalib      = 6;
static const int sdHcalCompst     = 7;

#endif

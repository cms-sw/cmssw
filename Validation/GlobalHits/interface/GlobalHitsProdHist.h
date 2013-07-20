#ifndef GlobalHitsProdHist_h
#define GlobalHitsProdHist_h

/** \class GlobalHitsProdHist
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2013/05/17 22:02:50 $
 *  $Revision: 1.10 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
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

//DQM services
//#include "DQMServices/Core/interface/DQMStore.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"

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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

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
#include "TH1F.h"

class GlobalHitsProdHist : public edm::one::EDProducer<edm::EndRunProducer>
{
  
 public:

  //typedef std::vector<float> FloatVector;

  explicit GlobalHitsProdHist(const edm::ParameterSet&);
  virtual ~GlobalHitsProdHist();
  virtual void beginJob() override;
  virtual void endJob() override;  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endRunProduce(edm::Run&, const edm::EventSetup&) override;
  
 private:

  // production related methods
  void fillG4MC(edm::Event&);
  void fillTrk(edm::Event&, const edm::EventSetup&);
  void fillMuon(edm::Event&, const edm::EventSetup&);
  void fillECal(edm::Event&, const edm::EventSetup&);
  void fillHCal(edm::Event&, const edm::EventSetup&);

 private:

  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  int vtxunit;
  bool getAllProvenances;
  bool printProvenanceInfo;

  //DQMStore *dbe;
  //std::string outputfile;

  std::vector<std::string> histName_;
  std::map<std::string, TH1F*> histMap_;

  // G4MC info
  TH1F *hMCRGP[2];
  TH1F *hMCG4Vtx[2];
  TH1F *hGeantVtxX[2];
  TH1F *hGeantVtxY[2];  
  TH1F *hGeantVtxZ[2];  
  TH1F *hMCG4Trk[2];
  TH1F *hGeantTrkPt;
  TH1F *hGeantTrkE;
  int nRawGenPart;  

  edm::InputTag G4VtxSrc_;
  edm::InputTag G4TrkSrc_;

  // Electromagnetic info
  // ECal info
  TH1F *hCaloEcal[2];
  TH1F *hCaloEcalE[2];
  TH1F *hCaloEcalToF[2];
  TH1F *hCaloEcalPhi;
  TH1F *hCaloEcalEta;  
  edm::InputTag ECalEBSrc_;
  edm::InputTag ECalEESrc_;

  // Preshower info
  TH1F *hCaloPreSh[2];
  TH1F *hCaloPreShE[2];
  TH1F *hCaloPreShToF[2];
  TH1F *hCaloPreShPhi;
  TH1F *hCaloPreShEta;
  edm::InputTag ECalESSrc_;

  // Hadronic info
  // HCal info
  TH1F *hCaloHcal[2];
  TH1F *hCaloHcalE[2];
  TH1F *hCaloHcalToF[2];
  TH1F *hCaloHcalPhi;
  TH1F *hCaloHcalEta;  
  edm::InputTag HCalSrc_;

  // Tracker info
  // Pixel info
  int nPxlHits;
  TH1F *hTrackerPx[2];
  TH1F *hTrackerPxPhi;
  TH1F *hTrackerPxEta;
  TH1F *hTrackerPxBToF;
  TH1F *hTrackerPxBR;
  TH1F *hTrackerPxFToF;
  TH1F *hTrackerPxFZ;
  edm::InputTag PxlBrlLowSrc_;
  edm::InputTag PxlBrlHighSrc_;
  edm::InputTag PxlFwdLowSrc_;
  edm::InputTag PxlFwdHighSrc_;

  // Strip info
  int nSiHits;
  TH1F *hTrackerSi[2];
  TH1F *hTrackerSiPhi;
  TH1F *hTrackerSiEta;
  TH1F *hTrackerSiBToF;
  TH1F *hTrackerSiBR;
  TH1F *hTrackerSiFToF;
  TH1F *hTrackerSiFZ;
  edm::InputTag SiTIBLowSrc_;
  edm::InputTag SiTIBHighSrc_;
  edm::InputTag SiTOBLowSrc_;
  edm::InputTag SiTOBHighSrc_;
  edm::InputTag SiTIDLowSrc_;
  edm::InputTag SiTIDHighSrc_;
  edm::InputTag SiTECLowSrc_;
  edm::InputTag SiTECHighSrc_;

  // Muon info
  TH1F *hMuon[2];
  TH1F *hMuonPhi;
  TH1F *hMuonEta;
  int nMuonHits;

  // DT info
  TH1F *hMuonDtToF[2];
  TH1F *hMuonDtR;
  edm::InputTag MuonDtSrc_;
  // CSC info
  TH1F *hMuonCscToF[2];
  TH1F *hMuonCscZ;
  edm::InputTag MuonCscSrc_;
  // RPC info
  TH1F *hMuonRpcFToF[2];
  TH1F *hMuonRpcFZ;
  TH1F *hMuonRpcBToF[2];
  TH1F *hMuonRpcBR;
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

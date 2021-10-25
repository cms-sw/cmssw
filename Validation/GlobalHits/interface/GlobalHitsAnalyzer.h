#ifndef GlobalHitsAnalyzer_h
#define GlobalHitsAnalyzer_h

/** \class GlobalHitsAnalyzer
 *
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

// DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// tracker info
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// muon info
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// calorimeter info
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

// data in edm::event
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// helper files
//#include <CLHEP/Vector/LorentzVector.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "TString.h"

class HcalDDDRecConstants;
class CaloGeometryRecord;
class HcalRecNumberingRecord;
class GlobalHitsAnalyzer : public DQMEDAnalyzer {
public:
  // typedef std::vector<float> FloatVector;

  explicit GlobalHitsAnalyzer(const edm::ParameterSet &);
  ~GlobalHitsAnalyzer() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // production related methods
  void fillG4MC(const edm::Event &);
  void fillTrk(const edm::Event &, const edm::EventSetup &);
  void fillMuon(const edm::Event &, const edm::EventSetup &);
  void fillECal(const edm::Event &, const edm::EventSetup &);
  void fillHCal(const edm::Event &, const edm::EventSetup &);

private:
  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  int vtxunit;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;
  bool testNumber;

  bool validHepMCevt;
  bool validG4VtxContainer;
  bool validG4trkContainer;
  bool validPxlBrlLow;
  bool validPxlBrlHigh;
  bool validPxlFwdLow;
  bool validPxlFwdHigh;
  bool validSiTIBLow;
  bool validSiTIBHigh;
  bool validSiTOBLow;
  bool validSiTOBHigh;
  bool validSiTIDLow;
  bool validSiTIDHigh;
  bool validSiTECLow;
  bool validSiTECHigh;
  bool validMuonCSC;
  bool validMuonDt;
  bool validMuonRPC;
  bool validEB;
  bool validEE;
  bool validPresh;
  bool validHcal;

  // G4MC info
  MonitorElement *meMCRGP[2];
  MonitorElement *meMCG4Vtx[2];
  MonitorElement *meGeantVtxX[2];
  MonitorElement *meGeantVtxY[2];
  MonitorElement *meGeantVtxZ[2];
  MonitorElement *meMCG4Trk[2];
  MonitorElement *meGeantTrkPt;
  MonitorElement *meGeantTrkE;
  MonitorElement *meGeantVtxEta;
  MonitorElement *meGeantVtxPhi;
  MonitorElement *meGeantVtxRad[2];
  MonitorElement *meGeantVtxMulti;
  int nRawGenPart;

  edm::InputTag G4VtxSrc_;
  edm::InputTag G4TrkSrc_;
  edm::EDGetTokenT<edm::SimVertexContainer> G4VtxSrc_Token_;
  edm::EDGetTokenT<edm::SimTrackContainer> G4TrkSrc_Token_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> hcaldddRecToken_;

  // Electromagnetic info
  // ECal info
  MonitorElement *meCaloEcal[2];
  MonitorElement *meCaloEcalE[2];
  MonitorElement *meCaloEcalToF[2];
  MonitorElement *meCaloEcalPhi;
  MonitorElement *meCaloEcalEta;
  edm::InputTag ECalEBSrc_;
  edm::InputTag ECalEESrc_;
  edm::EDGetTokenT<edm::PCaloHitContainer> ECalEBSrc_Token_;
  edm::EDGetTokenT<edm::PCaloHitContainer> ECalEESrc_Token_;

  // Preshower info
  MonitorElement *meCaloPreSh[2];
  MonitorElement *meCaloPreShE[2];
  MonitorElement *meCaloPreShToF[2];
  MonitorElement *meCaloPreShPhi;
  MonitorElement *meCaloPreShEta;
  edm::InputTag ECalESSrc_;
  edm::EDGetTokenT<edm::PCaloHitContainer> ECalESSrc_Token_;

  // Hadronic info
  // HCal info
  MonitorElement *meCaloHcal[2];
  MonitorElement *meCaloHcalE[2];
  MonitorElement *meCaloHcalToF[2];
  MonitorElement *meCaloHcalPhi;
  MonitorElement *meCaloHcalEta;
  edm::InputTag HCalSrc_;
  edm::EDGetTokenT<edm::PCaloHitContainer> HCalSrc_Token_;

  // Tracker info
  // Pixel info
  int nPxlHits;
  MonitorElement *meTrackerPx[2];
  MonitorElement *meTrackerPxPhi;
  MonitorElement *meTrackerPxEta;
  MonitorElement *meTrackerPxBToF;
  MonitorElement *meTrackerPxBR;
  MonitorElement *meTrackerPxFToF;
  MonitorElement *meTrackerPxFZ;
  edm::InputTag PxlBrlLowSrc_;
  edm::InputTag PxlBrlHighSrc_;
  edm::InputTag PxlFwdLowSrc_;
  edm::InputTag PxlFwdHighSrc_;
  edm::EDGetTokenT<edm::PSimHitContainer> PxlBrlLowSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> PxlBrlHighSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> PxlFwdLowSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> PxlFwdHighSrc_Token_;

  // Strip info
  int nSiHits;
  MonitorElement *meTrackerSi[2];
  MonitorElement *meTrackerSiPhi;
  MonitorElement *meTrackerSiEta;
  MonitorElement *meTrackerSiBToF;
  MonitorElement *meTrackerSiBR;
  MonitorElement *meTrackerSiFToF;
  MonitorElement *meTrackerSiFZ;
  edm::InputTag SiTIBLowSrc_;
  edm::InputTag SiTIBHighSrc_;
  edm::InputTag SiTOBLowSrc_;
  edm::InputTag SiTOBHighSrc_;
  edm::InputTag SiTIDLowSrc_;
  edm::InputTag SiTIDHighSrc_;
  edm::InputTag SiTECLowSrc_;
  edm::InputTag SiTECHighSrc_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTIBLowSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTIBHighSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTOBLowSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTOBHighSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTIDLowSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTIDHighSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTECLowSrc_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> SiTECHighSrc_Token_;

  // Muon info
  MonitorElement *meMuon[2];
  MonitorElement *meMuonPhi;
  MonitorElement *meMuonEta;
  int nMuonHits;

  // DT info
  MonitorElement *meMuonDtToF[2];
  MonitorElement *meMuonDtR;
  edm::InputTag MuonDtSrc_;
  edm::EDGetTokenT<edm::PSimHitContainer> MuonDtSrc_Token_;
  // CSC info
  MonitorElement *meMuonCscToF[2];
  MonitorElement *meMuonCscZ;
  edm::InputTag MuonCscSrc_;
  edm::EDGetTokenT<edm::PSimHitContainer> MuonCscSrc_Token_;
  // RPC info
  MonitorElement *meMuonRpcFToF[2];
  MonitorElement *meMuonRpcFZ;
  MonitorElement *meMuonRpcBToF[2];
  MonitorElement *meMuonRpcBR;
  edm::InputTag MuonRpcSrc_;
  edm::EDGetTokenT<edm::PSimHitContainer> MuonRpcSrc_Token_;

  // private statistics information
  unsigned int count;

};  // end class declaration

#endif

#ifndef GlobalHitMap
#define GlobalHitMap
// geometry mapping
static const int dTrk = 1;
static const int sdPxlBrl = 1;
static const int sdPxlFwd = 2;
static const int sdSiTIB = 3;
static const int sdSiTID = 4;
static const int sdSiTOB = 5;
static const int sdSiTEC = 6;

static const int dMuon = 2;
static const int sdMuonDT = 1;
static const int sdMuonCSC = 2;
static const int sdMuonRPC = 3;
static const int sdMuonRPCRgnBrl = 0;
static const int sdMuonRPCRgnFwdp = 1;
static const int sdMuonRPCRgnFwdn = -1;

static const int dEcal = 3;
static const int sdEcalBrl = 1;
static const int sdEcalFwd = 2;
static const int sdEcalPS = 3;
static const int sdEcalTT = 4;
static const int sdEcalLPnD = 5;

static const int dHcal = 4;
static const int sdHcalEmpty = 0;
static const int sdHcalBrl = 1;
static const int sdHcalEC = 2;
static const int sdHcalOut = 3;
static const int sdHcalFwd = 4;
static const int sdHcalTT = 5;
static const int sdHcalCalib = 6;
static const int sdHcalCompst = 7;

#endif

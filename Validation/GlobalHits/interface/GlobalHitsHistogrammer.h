#ifndef GlobalHitsHistogrammer_h
#define GlobalHitsHistogrammer_h

/** \class GlobalHitsHistogrammer
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//#include "DataFormats/DetId/interface/DetId.h"

//DQM services
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// tracker info
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
//#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// muon info
//#include "Geometry/Records/interface/MuonGeometryRecord.h"
//#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
//#include "Geometry/DTGeometry/interface/DTGeometry.h"
//#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
//#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
//#include "DataFormats/MuonDetId/interface/RPCDetId.h"
//#include "DataFormats/MuonDetId/interface/DTWireId.h"

// calorimeter info
//#include "Geometry/Records/interface/IdealGeometryRecord.h"
//#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
//#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
//#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

// data in edm::event
//#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//#include "SimDataFormats/Track/interface/SimTrackContainer.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
//#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

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
#include "DQMServices/Core/interface/MonitorElement.h"

class GlobalHitsHistogrammer : public DQMEDAnalyzer {

 public:

  //typedef std::vector<float> FloatVector;

  explicit GlobalHitsHistogrammer(const edm::ParameterSet&);
  virtual ~GlobalHitsHistogrammer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &,
      edm::Run const &, edm::EventSetup const &) override;

 private:

  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  int vtxunit;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  std::string outputfile;
  bool doOutput;

  edm::InputTag GlobalHitSrc_;
  edm::EDGetTokenT<PGlobalSimHit> GlobalHitSrc_Token_;

  // G4MC info
  MonitorElement *meMCRGP[2];
  MonitorElement *meMCG4Vtx[2];
  MonitorElement *meGeantVtxX[2];
  MonitorElement *meGeantVtxY[2];  
  MonitorElement *meGeantVtxZ[2];  
  MonitorElement *meMCG4Trk[2];
  MonitorElement *meGeantTrkPt;
  MonitorElement *meGeantTrkE; 

  // Electromagnetic info
  // ECal info
  MonitorElement *meCaloEcal[2];
  MonitorElement *meCaloEcalE[2];
  MonitorElement *meCaloEcalToF[2];
  MonitorElement *meCaloEcalPhi;
  MonitorElement *meCaloEcalEta;  

  // Preshower info
  MonitorElement *meCaloPreSh[2];
  MonitorElement *meCaloPreShE[2];
  MonitorElement *meCaloPreShToF[2];
  MonitorElement *meCaloPreShPhi;
  MonitorElement *meCaloPreShEta;

  // Hadronic info
  // HCal info
  MonitorElement *meCaloHcal[2];
  MonitorElement *meCaloHcalE[2];
  MonitorElement *meCaloHcalToF[2];
  MonitorElement *meCaloHcalPhi;
  MonitorElement *meCaloHcalEta;  

  // Tracker info
  // Pixel info
  int nPxlBrlHits;
  int nPxlFwdHits;
  int nPxlHits;
  MonitorElement *meTrackerPx[2];
  MonitorElement *meTrackerPxPhi;
  MonitorElement *meTrackerPxEta;
  MonitorElement *meTrackerPxBToF;
  MonitorElement *meTrackerPxBR;
  MonitorElement *meTrackerPxFToF;
  MonitorElement *meTrackerPxFZ;

  // Strip info
  int nSiHits;
  int nSiBrlHits;
  int nSiFwdHits;
  MonitorElement *meTrackerSi[2];
  MonitorElement *meTrackerSiPhi;
  MonitorElement *meTrackerSiEta;
  MonitorElement *meTrackerSiBToF;
  MonitorElement *meTrackerSiBR;
  MonitorElement *meTrackerSiFToF;
  MonitorElement *meTrackerSiFZ;

  // Muon info
  MonitorElement *meMuon[2];
  MonitorElement *meMuonPhi;
  MonitorElement *meMuonEta;
  int nMuonHits;

  // DT info
  int nMuonDtHits;
  MonitorElement *meMuonDtToF[2];
  MonitorElement *meMuonDtR;

  // CSC info
  int nMuonCscHits;
  MonitorElement *meMuonCscToF[2];
  MonitorElement *meMuonCscZ;

  // RPC info
  int nMuonRpcBrlHits;
  int nMuonRpcFwdHits;
  MonitorElement *meMuonRpcFToF[2];
  MonitorElement *meMuonRpcFZ;
  MonitorElement *meMuonRpcBToF[2];
  MonitorElement *meMuonRpcBR;

  // private statistics information
  unsigned int count;

}; // end class declaration
  
#endif



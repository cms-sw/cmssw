#ifndef Validation_MuonHits_MuonSimHitsValidAnalyzer_h
#define Validation_MuonHits_MuonSimHitsValidAnalyzer_h

/// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

/// muon CSC, DT and RPC geometry info
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

/// data in edm::event
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"

class TH1F;
class TFile;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class MuonSimHitsValidAnalyzer : public DQMEDAnalyzer {
public:
  typedef std::vector<float> FloatVector;
  typedef std::vector<int> IntegerVector;
  typedef std::vector<long int> LongIntegerVector;
  typedef std::vector<unsigned int> UnsigIntegerVector;

  explicit MuonSimHitsValidAnalyzer(const edm::ParameterSet&);
  ~MuonSimHitsValidAnalyzer() override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void fillDT(const edm::Event&, const edm::EventSetup&);

private:
  ///  parameter information
  std::string fName;
  int verbosity;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  /// G4MC info
  int nRawGenPart;

  unsigned int iden;
  Int_t wheel, station, sector, superlayer, layer, wire;
  Int_t region, ring, subsector, roll;
  Int_t path, pathchamber;
  Int_t touch1, touch4, nummu_DT, nummu_RPC, nummu_CSC;
  Int_t touche1, touche4;
  Float_t pow6;
  Float_t mom1, mom4, mome1, mome4;
  Float_t costeta, radius, sinteta;
  Float_t globposx, globposy, globposz;
  Float_t globposphi, globposeta;

  /// Input tags
  edm::EDGetTokenT<edm::PSimHitContainer> DTHitsToken_;

  // Monitor elements
  // DT
  MonitorElement* meAllDTHits;
  MonitorElement* meMuDTHits;
  MonitorElement* meToF;
  MonitorElement* meEnergyLoss;
  MonitorElement* meMomentumMB1;
  MonitorElement* meMomentumMB4;
  MonitorElement* meLossMomIron;
  MonitorElement* meLocalXvsZ;
  MonitorElement* meLocalXvsY;
  MonitorElement* meGlobalXvsZ;
  MonitorElement* meGlobalXvsY;
  MonitorElement* meGlobalXvsZWm2;
  MonitorElement* meGlobalXvsZWm1;
  MonitorElement* meGlobalXvsZW0;
  MonitorElement* meGlobalXvsZWp1;
  MonitorElement* meGlobalXvsZWp2;
  MonitorElement* meGlobalXvsYWm2;
  MonitorElement* meGlobalXvsYWm1;
  MonitorElement* meGlobalXvsYW0;
  MonitorElement* meGlobalXvsYWp1;
  MonitorElement* meGlobalXvsYWp2;
  MonitorElement* meWheelOccup;
  MonitorElement* meStationOccup;
  MonitorElement* meSectorOccup;
  MonitorElement* meSuperLOccup;
  MonitorElement* meLayerOccup;
  MonitorElement* meWireOccup;
  MonitorElement* mePathMuon;
  MonitorElement* meChamberOccup;
  MonitorElement* meHitRadius;
  MonitorElement* meCosTheta;
  MonitorElement* meGlobalEta;
  MonitorElement* meGlobalPhi;

  /// private statistics information
  unsigned int count;

};  /// end class declaration

/// geometry mapping

static const int dMuon = 2;

static const int sdMuonDT = 1;
static const int sdMuonCSC = 2;
static const int sdMuonRPC = 3;

#endif

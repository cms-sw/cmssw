#ifndef MuonSimHitsValidAnalyzer_h
#define MuonSimHitsValidAnalyzer_h

/// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
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
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/// muon CSC, DT and RPC geometry info
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

/// muon CSC detector id
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

/// data in edm::event
//#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

/// helper files
#include <CLHEP/Vector/LorentzVector.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>

#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"

class TH1F;
class TFile;

namespace edm {
  class ParameterSet; class Event; class EventSetup;}

class MuonSimHitsValidAnalyzer : public edm::EDAnalyzer
{
  
 public:

  typedef std::vector<float> FloatVector;
  typedef std::vector<int>   IntegerVector;
  typedef std::vector<long int>   LongIntegerVector;
  typedef std::vector<unsigned int>   UnsigIntegerVector;

  explicit MuonSimHitsValidAnalyzer(const edm::ParameterSet&);
  virtual ~MuonSimHitsValidAnalyzer();

  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:

  void fillCSC(const edm::Event&, const edm::EventSetup&);
  void fillDT(const edm::Event&, const edm::EventSetup&);
  void fillRPC(const edm::Event&, const edm::EventSetup&);

  void bookHistos_DT();
  void bookHistos_CSC();
  void bookHistos_RPC();
  void saveHistos_DT();
  void saveHistos_CSC();
  void saveHistos_RPC();

 private:

  ///  parameter information
  std::string fName;
  int verbosity;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  std::string DToutputFile_;
  std::string CSCoutputFile_;
  std::string RPCoutputFile_;

  TFile *theDTFile;
  TFile *theCSCFile;
  TFile *theRPCFile;


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
  Float_t costeta, radius,sinteta;
  Float_t globposx, globposy, globposz;
  Float_t globposphi, globposeta;

  /// Input tags

  edm::InputTag CSCHitsSrc_;
  edm::InputTag DTHitsSrc_;
  edm::InputTag RPCHitsSrc_;                                                
  
  // DaqMonitor element
  DQMStore* dbeDT_;
  DQMStore* dbeCSC_; 
  DQMStore* dbeRPC_;


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

  //CSC
  MonitorElement* meAllCSCHits;
  MonitorElement* meMuCSCHits;
  MonitorElement* meEnergyLoss_111;
  MonitorElement* meToF_311;  
  MonitorElement* meEnergyLoss_112;
  MonitorElement* meToF_312;
  MonitorElement* meEnergyLoss_113;
  MonitorElement* meToF_313;
  MonitorElement* meEnergyLoss_114;
  MonitorElement* meToF_314;
  MonitorElement* meEnergyLoss_121;
  MonitorElement* meToF_321;
  MonitorElement* meEnergyLoss_122;
  MonitorElement* meToF_322;
  MonitorElement* meEnergyLoss_131;
  MonitorElement* meToF_331;
  MonitorElement* meEnergyLoss_132;
  MonitorElement* meToF_332;
  MonitorElement* meEnergyLoss_141;
  MonitorElement* meToF_341;
  MonitorElement* meEnergyLoss_211;
  MonitorElement* meToF_411;
  MonitorElement* meEnergyLoss_212;
  MonitorElement* meToF_412;
  MonitorElement* meEnergyLoss_213;
  MonitorElement* meToF_413;
  MonitorElement* meEnergyLoss_214;
  MonitorElement* meToF_414;
  MonitorElement* meEnergyLoss_221;
  MonitorElement* meToF_421;
  MonitorElement* meEnergyLoss_222;
  MonitorElement* meToF_422;
  MonitorElement* meEnergyLoss_231;
  MonitorElement* meToF_431;
  MonitorElement* meEnergyLoss_232;
  MonitorElement* meToF_432;
  MonitorElement* meEnergyLoss_241;
  MonitorElement* meToF_441;

  //RPC
  MonitorElement* meAllRPCHits;
  MonitorElement* meMuRPCHits;
  MonitorElement* meRegionOccup;
  MonitorElement* meRingOccBar;
  MonitorElement* meRingOccEndc;
  MonitorElement* meStatOccBar;
  MonitorElement* meStatOccEndc;
  MonitorElement* meSectorOccBar;
  MonitorElement* meSectorOccEndc;
  MonitorElement* meLayerOccBar;
  MonitorElement* meLayerOccEndc;
  MonitorElement* meSubSectOccBar;
  MonitorElement* meSubSectOccEndc;
  MonitorElement* meRollOccBar;
  MonitorElement* meRollOccEndc;
  MonitorElement* meElossBar;
  MonitorElement* meElossEndc;
  MonitorElement* mepathRPC;
  MonitorElement* meMomRB1;
  MonitorElement* meMomRB4;
  MonitorElement* meLossMomBar;
  MonitorElement* meMomRE1;
  MonitorElement* meMomRE4;
  MonitorElement* meLossMomEndc; 
  MonitorElement* meLocalXvsYBar;
  MonitorElement* meGlobalXvsZBar;
  MonitorElement* meGlobalXvsYBar;
  MonitorElement* meLocalXvsYEndc;
  MonitorElement* meGlobalXvsZEndc;
  MonitorElement* meGlobalXvsYEndc;
  MonitorElement* meHitRadiusBar;
  MonitorElement* meCosThetaBar;
  MonitorElement* meHitRadiusEndc;
  MonitorElement* meCosThetaEndc;


  /// private statistics information
  unsigned int count;

}; /// end class declaration

/// geometry mapping
 
static const int dMuon            = 2;

static const int sdMuonDT         = 1;
static const int sdMuonCSC        = 2;
static const int sdMuonRPC        = 3;

#endif

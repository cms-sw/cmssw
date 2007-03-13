#ifndef MuonSimHitsValidAnalyzer_h
#define MuonSimHitsValidAnalyzer_h

/// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
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
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
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

class TH1F;
class TFile;

class MuonSimHitsValidAnalyzer : public edm::EDAnalyzer
{
  
 public:

  typedef std::vector<float> FloatVector;
  typedef std::vector<int>   IntegerVector;
  typedef std::vector<long int>   LongIntegerVector;
  typedef std::vector<unsigned int>   UnsigIntegerVector;

  explicit MuonSimHitsValidAnalyzer(const edm::ParameterSet&);
  virtual ~MuonSimHitsValidAnalyzer();

  virtual void beginJob(const edm::EventSetup&);
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
  

  /// private statistics information
  unsigned int count;

}; /// end class declaration

/// geometry mapping
 
static const int dMuon            = 2;

static const int sdMuonDT         = 1;
static const int sdMuonCSC        = 2;
static const int sdMuonRPC        = 3;

#endif

#ifndef GlobalRecHitsProducer_h
#define GlobalRecHitsProducer_h

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Provenance.h"
//#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// ecal calorimeter info
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

// hcal calorimeter info
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
//#include "Geometry/Records/interface/IdealGeometryRecord.h"
//#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
//#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

// silicon strip info
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

// silicon pixel info
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

// muon DT info
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

// muon CSC info
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

// event info
#include "SimDataFormats/GlobalRecHitValidation/interface/PGlobalRecHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

// helper files
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Units/SystemOfUnits.h>

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>
#include <map>

#include "TString.h"

class PGlobalRecHit;

class GlobalRecHitsProducer : public edm::EDProducer
{

 public:

  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;
  typedef std::vector<int> IntVector;
  typedef std::map<uint32_t,float,std::less<uint32_t> > MapType;

  explicit GlobalRecHitsProducer(const edm::ParameterSet&);
  virtual ~GlobalRecHitsProducer();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  // production related methods
  void fillECal(edm::Event&, const edm::EventSetup&);
  void storeECal(PGlobalRecHit&);
  void fillHCal(edm::Event&, const edm::EventSetup&);
  void storeHCal(PGlobalRecHit&);
  void fillTrk(edm::Event&, const edm::EventSetup&);
  void storeTrk(PGlobalRecHit&);
  void fillMuon(edm::Event&, const edm::EventSetup&);
  void storeMuon(PGlobalRecHit&);  

  void clear();

 private:

  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  // Electromagnetic info
  // ECal info
 
  FloatVector EBRE; 
  FloatVector EBSHE;

  FloatVector EERE; 
  FloatVector EESHE;

  FloatVector ESRE; 
  FloatVector ESSHE;

  edm::InputTag ECalEBSrc_;
  edm::InputTag ECalUncalEBSrc_;
  edm::InputTag ECalEESrc_;
  edm::InputTag ECalUncalEESrc_;
  edm::InputTag ECalESSrc_;

  // HCal info

  FloatVector HBCalREC;
  FloatVector HBCalSHE;

  FloatVector HECalREC;
  FloatVector HECalSHE;

  FloatVector HOCalREC;
  FloatVector HOCalSHE;

  FloatVector HFCalREC;
  FloatVector HFCalSHE;

  edm::InputTag HCalSrc_;

  // Tracker info
  // SiStrip
  
  FloatVector TIBL1RX, TIBL2RX, TIBL3RX, TIBL4RX;
  FloatVector TIBL1RY, TIBL2RY, TIBL3RY, TIBL4RY;
  FloatVector TIBL1SX, TIBL2SX, TIBL3SX, TIBL4SX;
  FloatVector TIBL1SY, TIBL2SY, TIBL3SY, TIBL4SY;

  FloatVector TOBL1RX, TOBL2RX, TOBL3RX, TOBL4RX;
  FloatVector TOBL1RY, TOBL2RY, TOBL3RY, TOBL4RY;
  FloatVector TOBL1SX, TOBL2SX, TOBL3SX, TOBL4SX;
  FloatVector TOBL1SY, TOBL2SY, TOBL3SY, TOBL4SY;

  FloatVector TIDW1RX, TIDW2RX, TIDW3RX;
  FloatVector TIDW1RY, TIDW2RY, TIDW3RY;
  FloatVector TIDW1SX, TIDW2SX, TIDW3SX;
  FloatVector TIDW1SY, TIDW2SY, TIDW3SY;

  FloatVector TECW1RX, TECW2RX, TECW3RX, TECW4RX, TECW5RX, TECW6RX, TECW7RX,
    TECW8RX;
  FloatVector TECW1RY, TECW2RY, TECW3RY, TECW4RY, TECW5RY, TECW6RY, TECW7RY,
    TECW8RY;
  FloatVector TECW1SX, TECW2SX, TECW3SX, TECW4SX, TECW5SX, TECW6SX, TECW7SX,
    TECW8SX;
  FloatVector TECW1SY, TECW2SY, TECW3SY, TECW4SY, TECW5SY, TECW6SY, TECW7SY,
    TECW8SY;

  edm::InputTag SiStripSrc_;

  // SiPxl

  FloatVector BRL1RX, BRL2RX, BRL3RX;
  FloatVector BRL1RY, BRL2RY, BRL3RY;
  FloatVector BRL1SX, BRL2SX, BRL3SX;
  FloatVector BRL1SY, BRL2SY, BRL3SY;

  FloatVector FWD1pRX, FWD1nRX, FWD2pRX, FWD2nRX;
  FloatVector FWD1pRY, FWD1nRY, FWD2pRY, FWD2nRY;
  FloatVector FWD1pSX, FWD1nSX, FWD2pSX, FWD2nSX;
  FloatVector FWD1pSY, FWD1nSY, FWD2pSY, FWD2nSY;

  edm::InputTag SiPxlSrc_;

  // Muon info
  // DT

  FloatVector DTRHD;
  FloatVector DTSHD;

  edm::InputTag MuDTSrc_;

  // CSC

  FloatVector CSCRHPHI;
  FloatVector CSCRHPERP;
  FloatVector CSCSHPHI;

  edm::InputTag MuCSCSrc_;

  // RPC

  FloatVector RPCRHX;
  FloatVector RPCSHX;

  edm::InputTag MuRPCSrc_;

  // private statistics information
  unsigned int count;

}; // end class declaration

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

#endif //PGlobalRecHitsProducer_h

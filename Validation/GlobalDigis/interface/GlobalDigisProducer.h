#ifndef GlobalDigisProducer_h
#define GlobalDigisProducer_h

/** \class GlobalDigiProducer
 *  
 *  Class to fill PGlobalDigi object to be inserted into data stream 
 *  containing information about various sub-systems in global coordinates 
 *  with full geometry
 *
 *  $Date: 2012/12/26 22:47:50 $
 *  $Revision: 1.18 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// ecal calorimeter info
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"

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

// silicon pixel info
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// muon DT info
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

// muon CSC Strip info
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

// muon CSC Wire info
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

// event info
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

// helper files
//#include <CLHEP/Vector/LorentzVector.h>
//#include <CLHEP/Units/SystemOfUnits.h>

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>
#include <map>

#include "TString.h"

class PGlobalDigi;

class GlobalDigisProducer : public edm::EDProducer
{

 public:

  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;
  typedef std::vector<int> IntVector;
  typedef std::map<uint32_t,float,std::less<uint32_t> > MapType;

  explicit GlobalDigisProducer(const edm::ParameterSet&);
  virtual ~GlobalDigisProducer();
  virtual void beginJob( void );
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  // production related methods
  void fillECal(edm::Event&, const edm::EventSetup&);
  void storeECal(PGlobalDigi&);
  void fillHCal(edm::Event&, const edm::EventSetup&);
  void storeHCal(PGlobalDigi&);
  void fillTrk(edm::Event&, const edm::EventSetup&);
  void storeTrk(PGlobalDigi&);
  void fillMuon(edm::Event&, const edm::EventSetup&);
  void storeMuon(PGlobalDigi&);  

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
 
  IntVector EBCalmaxPos; 
  DoubleVector EBCalAEE; 
  FloatVector EBCalSHE;

  IntVector EECalmaxPos; 
  DoubleVector EECalAEE; 
  FloatVector EECalSHE;

  FloatVector ESCalADC0, ESCalADC1, ESCalADC2;
  FloatVector ESCalSHE;

  edm::InputTag ECalEBSrc_;
  edm::InputTag ECalEESrc_;
  edm::InputTag ECalESSrc_;

  std::map<int, double, std::less<int> > ECalgainConv_;
  double ECalbarrelADCtoGeV_;
  double ECalendcapADCtoGeV_;

  // HCal info

  FloatVector HBCalAEE;
  FloatVector HBCalSHE;

  FloatVector HECalAEE;
  FloatVector HECalSHE;

  FloatVector HOCalAEE;
  FloatVector HOCalSHE;

  FloatVector HFCalAEE;
  FloatVector HFCalSHE;

  edm::InputTag HCalSrc_;
  edm::InputTag HCalDigi_;

  // Tracker info
  // SiStrip
  
  FloatVector TIBL1ADC, TIBL2ADC, TIBL3ADC, TIBL4ADC;
  IntVector TIBL1Strip, TIBL2Strip, TIBL3Strip, TIBL4Strip;

  FloatVector TOBL1ADC, TOBL2ADC, TOBL3ADC, TOBL4ADC;
  IntVector TOBL1Strip, TOBL2Strip, TOBL3Strip, TOBL4Strip;

  FloatVector TIDW1ADC, TIDW2ADC, TIDW3ADC;
  IntVector TIDW1Strip, TIDW2Strip, TIDW3Strip;

  FloatVector TECW1ADC, TECW2ADC, TECW3ADC, TECW4ADC, TECW5ADC, TECW6ADC, 
    TECW7ADC, TECW8ADC;
  IntVector TECW1Strip, TECW2Strip, TECW3Strip, TECW4Strip, TECW5Strip, 
    TECW6Strip, TECW7Strip, TECW8Strip;

  edm::InputTag SiStripSrc_;

  // SiPxl

  FloatVector BRL1ADC, BRL2ADC, BRL3ADC;
  IntVector BRL1Row, BRL2Row, BRL3Row;
  IntVector BRL1Col, BRL2Col, BRL3Col;

  FloatVector FWD1pADC, FWD1nADC, FWD2pADC, FWD2nADC;
  IntVector FWD1pRow, FWD1nRow, FWD2pRow, FWD2nRow;
  IntVector FWD1pCol, FWD1nCol, FWD2pCol, FWD2nCol;

  edm::InputTag SiPxlSrc_;

  // Muon info
  // DT

  IntVector MB1SLayer, MB2SLayer, MB3SLayer, MB4SLayer;
  FloatVector MB1Time, MB2Time, MB3Time, MB4Time;
  IntVector MB1Layer, MB2Layer, MB3Layer, MB4Layer;

  edm::InputTag MuDTSrc_;

  // CSC Strip

  float theCSCStripPedestalSum;
  int theCSCStripPedestalCount;

  FloatVector CSCStripADC;

  edm::InputTag MuCSCStripSrc_;

  // CSC Wire

  FloatVector CSCWireTime;

  edm::InputTag MuCSCWireSrc_;

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

#endif //PGlobalDigisProducer_h

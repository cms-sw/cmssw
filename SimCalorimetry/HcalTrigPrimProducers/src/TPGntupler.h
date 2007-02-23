//system include files
#include <memory>
#include <map>
#include <iostream>
#include <math.h>
#include <utility>
#include <algorithm>

//Framework includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//ROOT includes
#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"

//Geometry includes
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//DataFormats includes
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"


//SimDataFormats includes
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

//CalibFormats
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"


using namespace std;
using namespace edm;

class CaloGeometry;

class TPGntupler : public edm::EDAnalyzer {
 public:
  explicit TPGntupler(const edm::ParameterSet&);
  typedef multimap<HcalTrigTowerDetId, PCaloHit> IdtoHit;
  typedef map<HcalDetId, double> Cell_Map;
  typedef map<HcalTrigTowerDetId, double> IdtoEnergy;
  ~TPGntupler();


 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  HcalTrigTowerGeometry theTrigTowerGeometry;
  TFile file;
  TTree tree;
  IdtoHit hit_map;
  IdtoEnergy Hit_towers;
  IdtoEnergy TP_towers;
  Cell_Map Hit_cells;
  int run_num;
  int event_num;
  int ieta;
  int iphi;
  float tpg_energy;
  float hit_energy;
};

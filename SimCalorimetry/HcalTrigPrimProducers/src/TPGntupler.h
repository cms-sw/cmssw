//system include files
#include <memory>
#include <map>
#include <iostream>
#include <math.h>

//Framework includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
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
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"

//SimDataFormats includes
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

//CalibFormats
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"


class CaloGeometry;

class TPGntupler : public edm::EDAnalyzer {
 public:
  explicit TPGntupler(const edm::ParameterSet&);
  typedef std::map<HcalTrigTowerDetId, double> IdtoEnergy;
  ~TPGntupler();


 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  HcalTrigTowerGeometry theTrigTowerGeometry;
  TFile file;
  TTree tree;
  IdtoEnergy Rec_towers;
  int run_num;
  int event_num;
  int ieta[4176];
  int iphi[4176];
  float tpg_energy[4176];
  float rec_energy[4176];
  float tpg_uncompressed[4176];
  int index[4176];
  edm::ESHandle<CaloTPGTranscoder> transcoder_;

};

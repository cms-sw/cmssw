//#ifndef HCALDIGITESTER_H
//#define HCALDIGITESTER_H

#include <memory>
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"


#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>


//
// class declaration
//

using namespace cms;
using namespace edm;
using namespace std;


class HcalDigiTester : public edm::EDAnalyzer {
public:
  explicit HcalDigiTester(const edm::ParameterSet&);
  ~HcalDigiTester();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  //  virtual void fill(const HBHEDataFrame& digi) const;
  //  virtual void fill(const HODataFrame& digi) const;
  // virtual void fill(const HFDataFrame& digi) const;
 template<class Digi>  void reco(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 

 private:

  DaqMonitorBEInterface* dbe_;
  
  string outputFile_;
  string hcalselector_;
  bool subpedvalue_;

  //  HE Monitor Elements
  MonitorElement* meEtaHE;
  MonitorElement* mePhiHE;
  MonitorElement* meDigiSimhitHE;
  MonitorElement* meRatioDigiSimhitHE;
  MonitorElement* meDigiSimhitHEprofile;
  MonitorElement* menDigisHE;
  MonitorElement* meSumDigisHE;
  MonitorElement* meSumDigis_noise_HE;

 //  HE Monitor Elements
  MonitorElement* meEtaHB;
  MonitorElement* mePhiHB;
  MonitorElement* meDigiSimhitHB;
  MonitorElement* meRatioDigiSimhitHB;
  MonitorElement* meDigiSimhitHBprofile;
  MonitorElement* menDigisHB;
  MonitorElement* meSumDigisHB;
  MonitorElement* meSumDigis_noise_HB;

  //   HF Monitor Elements
  MonitorElement* meEtaHF;
  MonitorElement* mePhiHF;
  MonitorElement* meDigiSimhitHF;
  MonitorElement* meRatioDigiSimhitHF;
  MonitorElement* meDigiSimhitHFprofile;
  MonitorElement* menDigisHF;
  MonitorElement* meSumDigisHF;
  MonitorElement* meSumDigis_noise_HF;
 //   HO Monitor Elements
  MonitorElement* meEtaHO;
  MonitorElement* mePhiHO;
  MonitorElement* meDigiSimhitHO;
  MonitorElement* meRatioDigiSimhitHO;
  MonitorElement* meDigiSimhitHOprofile; 
  MonitorElement* menDigisHO;
  MonitorElement* meSumDigisHO;
  MonitorElement* meSumDigis_noise_HO;
ESHandle<CaloGeometry> geometry ;
ESHandle<HcalDbService> conditions;
 float pedvalue;
};

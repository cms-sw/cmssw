#ifndef ValidationSimHitsValidationHcal_H
#define ValidationSimHitsValidationHcal_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class SimHitsValidationHcal: public edm::EDAnalyzer{
public:

  SimHitsValidationHcal(const edm::ParameterSet& ps);
  ~SimHitsValidationHcal();

protected:

  void beginJob ();
  void endJob   ();
  void analyze  (const edm::Event& e, const edm::EventSetup& c);

  void analyzeHits  (std::vector<PCaloHit> &);

private:
  
  std::string    g4Label, hcalHits;
  bool           verbose_;
  DQMStore       *dbe_;

  struct energysum {
    double e25, e50, e100, e250;
    energysum() {e25=e50=e100=e250=0.0;}
  };

  static const int nType = 25;

  MonitorElement *meHcalHitEta_[nType];
  MonitorElement *meHcalHitTimeEta_[nType];
  MonitorElement *meHcalEnergyl25_[nType];
  MonitorElement *meHcalEnergyl50_[nType];
  MonitorElement *meHcalEnergyl100_[nType];
  MonitorElement *meHcalEnergyl250_[nType];
  
  MonitorElement *meEnergy_HB;
  MonitorElement *meEnergy_HE;
  MonitorElement *meEnergy_HO;
  MonitorElement *meEnergy_HF;


  MonitorElement *metime_HB;
  MonitorElement *metime_HE;
  MonitorElement *metime_HO;
  MonitorElement *metime_HF;

  MonitorElement *metime_enweighted_HB;
  MonitorElement *metime_enweighted_HE;
  MonitorElement *metime_enweighted_HO;
  MonitorElement *metime_enweighted_HF;
  
};

#endif

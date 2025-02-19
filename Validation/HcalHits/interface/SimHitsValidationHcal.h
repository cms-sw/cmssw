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

  MonitorElement *meHcalHitEta_[25];

};

#endif

#ifndef ValidationSimHitsValidationHcal_H
#define ValidationSimHitsValidationHcal_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

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

  void beginRun(const edm::Run& r,  const edm::EventSetup& c);
  void beginJob () {}
  void endJob   () {}
  void analyze  (const edm::Event& e, const edm::EventSetup& c);

private:

  struct energysum {
    energysum() {e25=e50=e100=e250=0.0;}
    double e25, e50, e100, e250;
  };

  struct idType {
    idType() {subdet=HcalEmpty; z=depth1=depth2=0;}
    idType(HcalSubdetector det, int iz, int d1, int d2) {subdet=det; z=iz; depth1=d1; depth2=d2;}
    HcalSubdetector subdet;
    int             z, depth1, depth2;
  };

  struct etaRange {
    etaRange() {bins=0; low=high=0;}
    etaRange(int bin, double min, double max) {bins=bin; low=min; high=max;}
    int             bins;
    double          low, high;
  };

  std::vector<std::pair<std::string,std::string> > getHistogramTypes();
  void analyzeHits  (std::vector<PCaloHit> &);
  etaRange getLimits (idType);
  std::pair<int,int> histId(int subdet, int eta, int depth, unsigned int dep);

  bool                initialized;
  std::string         g4Label_, hcalHits_;
  const HcalDDDRecConstants *hcons;
  std::vector<idType> types;
  bool                verbose_, testNumber_;
  int                 maxDepthHB_, maxDepthHE_, maxDepthHO_, maxDepthHF_; 
  DQMStore            *dbe_;


  std::vector<MonitorElement*> meHcalHitEta_, meHcalHitTimeEta_;
  std::vector<MonitorElement*> meHcalEnergyl25_, meHcalEnergyl50_;
  std::vector<MonitorElement*> meHcalEnergyl100_, meHcalEnergyl250_;
  MonitorElement *meEnergy_HB, *metime_HB, *metime_enweighted_HB;
  MonitorElement *meEnergy_HE, *metime_HE, *metime_enweighted_HE;
  MonitorElement *meEnergy_HO, *metime_HO, *metime_enweighted_HO;
  MonitorElement *meEnergy_HF, *metime_HF, *metime_enweighted_HF;
};

#endif

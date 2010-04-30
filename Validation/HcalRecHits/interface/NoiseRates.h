#ifndef _ANOMALOUSNOISE_ANALYZERS_NOISERATES_H_
#define _ANOMALOUSNOISE_ANALYZERS_NOISERATES_H_


//
// NoiseRates.h
//
//    description: Makes plots to calculate the anomalous noise rates
//
//    author: J.P. Chou, Brown
//
//

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"


//
// class declaration
//

class NoiseRates : public edm::EDAnalyzer {
 public:
  explicit NoiseRates(const edm::ParameterSet&);
  ~NoiseRates();
  
  
 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  DQMStore* dbe_;
  std::string outputFile_;

  // parameters
  edm::InputTag rbxCollName_;          // label for the rbx collection
  double minRBXEnergy_;                // RBX energy threshold
  double minHitEnergy_;                // RecHit energy threshold
  bool   useAllHistos_;

  MonitorElement* hLumiBlockCount_;
  MonitorElement* hRBXEnergy_;
  MonitorElement* hRBXEnergyType1_;
  MonitorElement* hRBXEnergyType2_;
  MonitorElement* hRBXEnergyType3_;
  MonitorElement* hRBXNHits_;

  // count lumi segments
  std::map<int, int> lumiCountMap_;

};


#endif

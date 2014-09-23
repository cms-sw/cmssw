#ifndef ValidationHGCalDigiClient_H
#define ValidationHGCalDigiClient_H

// -*- C++ -*-
/*
 Description: This is a SImHit CLient code
*/
//
// Originally create by: Kalyanmoy Chatterjee
//       and Raman Khurana
//                        

#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include <iostream>
#include <fstream>
#include <vector>

class DQMStore;
class MonitorElement;

class HGCalDigiClient : public edm::EDAnalyzer {
 
private:
  DQMStore* dbe_;
  std::string outputFile_;
  // edm::ParameterSet conf_;
  std::string nameDetector_;
  int verbosity_;
  HGCalDDDConstants *hgcons_;

public:
  explicit HGCalDigiClient(const edm::ParameterSet& );
  virtual ~HGCalDigiClient();
  
  virtual void beginJob(void);
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  void runClient_();   
  int          DigisEndjob(const std::vector<MonitorElement*> &hcalMEs);
};

#endif

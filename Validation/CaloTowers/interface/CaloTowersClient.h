#ifndef HCALVALIDATION_CALOTOWERS_CALOTOWERSCLIENT
#define HCALVALIDATION_CALOTOWERS_CALOTOWERSCLIENT

// -*- C++ -*-
//
// 
/*
 Description: This is a CaloTowers client meant to plot calotowers quantities 
*/

//
// Originally create by: Hongxuan Liu 
//                        May 2010
//

#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <iostream>
#include <fstream>
#include <vector>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class DQMStore;
class MonitorElement;

class CaloTowersClient : public DQMEDHarvester {
 
 private:
  std::string outputFile_;

  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;

  std::string dirName_;
  std::string dirNameJet_;
  std::string dirNameMET_;

 public:
  explicit CaloTowersClient(const edm::ParameterSet& );
  virtual ~CaloTowersClient();
  
  virtual void beginJob(void);
//  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
//  virtual void analyze(const edm::Event&, const edm::EventSetup&);
//  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);

  int CaloTowersEndjob(const std::vector<MonitorElement*> &hcalMEs);

};
 
#endif

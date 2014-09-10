#ifndef HCALVALIDATION_CALOTOWERS_HCALRECHITSCLIENT
#define HCALVALIDATION_CALOTOWERS_HCALRECHITSCLIENT

// -*- C++ -*-
//
// 
/*
 Description: This is a RecHits client meant to plot rechits quantities 
*/

//
// Originally create by: Hongxuan Liu 
//                        May 2010
//

#include <memory>
#include <unistd.h>
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <iostream>
#include <fstream>
#include <vector>

class MonitorElement;

class HcalRecHitsClient : public DQMEDHarvester {
 
 private:
  std::string outputFile_;

  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;

  std::string dirName_;
  std::string dirNameJet_;
  std::string dirNameMET_;

 public:
  explicit HcalRecHitsClient(const edm::ParameterSet& );
  virtual ~HcalRecHitsClient();

  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);
  
  virtual void runClient_(DQMStore::IBooker &, DQMStore::IGetter &);   

  int HcalRecHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);

};
 
#endif

#ifndef ValidationHGCalRecHitsClient_H
#define ValidationHGCalRecHitsClient_H

// -*- C++ -*-
/*
 Description: This is a HGCRecHit CLient code
*/
//
// Originally create by: Kalyanmoy Chatterjee
//       and Raman Khurana
//                        

#include <memory>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

class DQMStore;
class MonitorElement;

class HGCalRecHitsClient : public DQMEDHarvester {
 
private:
  //member data
  int          verbosity_;
  std::string  nameDetector_;
  unsigned int layers_;

public:
  explicit HGCalRecHitsClient(const edm::ParameterSet& );
  virtual ~HGCalRecHitsClient();
  
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig);
  virtual void runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig);   

  int recHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);
};

#endif

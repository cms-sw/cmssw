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


#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <unistd.h>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

class DQMStore;
class MonitorElement;

class HGCalDigiClient : public DQMEDHarvester {
 
private:
  std::string nameDetector_;
  int         verbosity_;
  int         layers_;

public:
  explicit HGCalDigiClient(const edm::ParameterSet& );
  virtual ~HGCalDigiClient();
  
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig);
  void         runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig);   
  int          digisEndjob(const std::vector<MonitorElement*> &hcalMEs);
};

#endif

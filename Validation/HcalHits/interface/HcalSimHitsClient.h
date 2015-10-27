#ifndef ValidationHcalSimHitsClient_H
#define ValidationHcalSimHitsClient_H

// -*- C++ -*-
//
// 
/*
 Description: This is a SImHit CLient code
*/

//
// Originally create by: Bhawna Gomber
//                        
//

#include <memory>
#include <unistd.h>
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include <iostream>
#include <fstream>
#include <vector>

class MonitorElement;

class HcalSimHitsClient : public DQMEDHarvester {
 
private:
  int SimHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);
  std::vector<std::string> getHistogramTypes();

  std::string                dirName_;
  bool                       verbose_;
  static const int           nTime = 4;
  static const int           nType1 = 4;
  const HcalDDDRecConstants *hcons;
  int                        maxDepthHB_, maxDepthHE_, maxDepthHO_, maxDepthHF_;

public:
  explicit HcalSimHitsClient(const edm::ParameterSet& );
  virtual ~HcalSimHitsClient();
  
  virtual void beginRun(edm::Run const& run, edm::EventSetup const& c);
  virtual void runClient_(DQMStore::IBooker &, DQMStore::IGetter &);   
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

};

#endif

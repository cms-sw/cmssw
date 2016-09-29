#ifndef ValidationHGCalSimHitsClient_H
#define ValidationHGCalSimHitsClient_H

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

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include <iostream>
#include <fstream>
#include <vector>

class MonitorElement;

class HGCalSimHitsClient : public DQMEDHarvester {
 
private:

  //member data
  std::string  nameDetector_;
  int          nTimes_, verbosity_;
  unsigned int layers_;

public:
  explicit HGCalSimHitsClient(const edm::ParameterSet& );
  virtual ~HGCalSimHitsClient();
  
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig);
  virtual void runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig);   
  int simHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);
};

#endif

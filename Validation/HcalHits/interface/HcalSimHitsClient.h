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

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include <memory>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

class HcalSimHitsClient : public DQMEDHarvester {
private:
  int SimHitsEndjob(const std::vector<MonitorElement *> &hcalMEs);
  std::vector<std::string> getHistogramTypes();

  std::string dirName_;
  bool verbose_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;
  static const int nTime = 4;
  static const int nType1 = 4;
  const HcalDDDRecConstants *hcons;
  int maxDepthHB_, maxDepthHE_, maxDepthHO_, maxDepthHF_;

public:
  explicit HcalSimHitsClient(const edm::ParameterSet &);
  ~HcalSimHitsClient() override;

  void beginRun(edm::Run const &run, edm::EventSetup const &c) override;
  virtual void runClient_(DQMStore::IBooker &, DQMStore::IGetter &);
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
};

#endif

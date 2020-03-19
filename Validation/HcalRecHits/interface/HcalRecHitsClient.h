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

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <memory>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

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
  explicit HcalRecHitsClient(const edm::ParameterSet &);
  ~HcalRecHitsClient() override;

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  int HcalRecHitsEndjob(const std::vector<MonitorElement *> &hcalMEs);
};

#endif

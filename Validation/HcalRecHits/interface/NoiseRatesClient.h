#ifndef HCALVALIDATION_CALOTOWERS_NOISERATESCLIENT
#define HCALVALIDATION_CALOTOWERS_NOISERATESCLIENT

// -*- C++ -*-
//
// 
/*
 Description: This is a NoiseRates client meant to plot noiserates quantities 
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

class NoiseRatesClient : public DQMEDHarvester {
 
 private:
  std::string outputFile_;

  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;

  std::string dirName_;
  std::string dirNameJet_;
  std::string dirNameMET_;

 public:
  explicit NoiseRatesClient(const edm::ParameterSet& );
  virtual ~NoiseRatesClient();
  
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);
  virtual void runClient_(DQMStore::IBooker &, DQMStore::IGetter &);   

  int NoiseRatesEndjob(const std::vector<MonitorElement*> &hcalMEs);

};
 
#endif

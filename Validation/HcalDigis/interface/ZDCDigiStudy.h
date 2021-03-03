////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Package:    ZDCDigiStudy
// Class:      ZDCDigiStudy
//
/*
 Description: 
              This code has been developed to be a check for the ZDC sim. In 2009, it was found that the ZDC Simulation was unrealistic and needed repair. The aim of this code is to show the user the input and output of a ZDC MinBias simulation.

 Implementation:
      First a MinBias simulation should be run, it could be pythia,hijin,or hydjet. This will output a .root file which should have information about recoGenParticles, hcalunsuppresseddigis. Use this .root file as the input into the cfg.py which is found in the main directory of this package. This output will be another .root file which is meant to be viewed in a TBrowser.

*/
//
// Original Author: Jaime Gomez (U. of Maryland) with SIGNIFICANT assistance of Dr. Jefferey Temple (U. of Maryland)
//
//
//         Created:  Summer 2012
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SimG4CMS_ZDCDigiStudy_H
#define SimG4CMS_ZDCDigiStudy_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <memory>

class ZDCDigiStudy : public DQMOneEDAnalyzer<> {
public:
  ZDCDigiStudy(const edm::ParameterSet& ps);
  ~ZDCDigiStudy() override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  void dqmEndRun(const edm::Run& run, const edm::EventSetup& c) override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  int FillHitValHist(int side, int section, int channel, double energy, double time);

private:
  /////////////////////////////////////////
  //#   Below all the monitoring elements #
  //# are simply the plots "code names"  #
  //# they will be filled in the .cc file #
  /////////////////////////////////////////

  std::string zdcHits, outFile_;
  bool verbose_, checkHit_;

  edm::EDGetTokenT<ZDCDigiCollection> tok_zdc_;

  /////////////////

  /////////ZDC Digi Plots////////
  MonitorElement* meZdcfCPHAD;
  MonitorElement* meZdcfCPTOT;
  MonitorElement* meZdcfCNHAD;
  MonitorElement* meZdcfCNTOT;
  MonitorElement* meZdcfCPEMvHAD;
  MonitorElement* meZdcfCNEMvHAD;
  MonitorElement* meZdcPEM1fCvsTS;
  MonitorElement* meZdcPEM2fCvsTS;
  MonitorElement* meZdcPEM3fCvsTS;
  MonitorElement* meZdcPEM4fCvsTS;
  MonitorElement* meZdcPEM5fCvsTS;
  MonitorElement* meZdcPHAD1fCvsTS;
  MonitorElement* meZdcPHAD2fCvsTS;
  MonitorElement* meZdcPHAD3fCvsTS;
  MonitorElement* meZdcPHAD4fCvsTS;
  MonitorElement* meZdcNEM1fCvsTS;
  MonitorElement* meZdcNEM2fCvsTS;
  MonitorElement* meZdcNEM3fCvsTS;
  MonitorElement* meZdcNEM4fCvsTS;
  MonitorElement* meZdcNEM5fCvsTS;
  MonitorElement* meZdcNHAD1fCvsTS;
  MonitorElement* meZdcNHAD2fCvsTS;
  MonitorElement* meZdcNHAD3fCvsTS;
  MonitorElement* meZdcNHAD4fCvsTS;
  ////////////////////////////////////////
};

#endif

// -*- C++ -*-
//
// Class:      GlobalTest
// 
/**\class GlobalTest

 Description: test suite for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
// $Id: GlobalTest.h,v 1.6 2012/10/10 14:39:02 wdd Exp $
//
//


// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class TFile;
class MonitorElement;

//
// class declaration
//

class GlobalTest : public edm::EDAnalyzer {
public:
  explicit GlobalTest(const edm::ParameterSet&);
  ~GlobalTest();

  void beginJob();
  void endJob();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  std::string filename_;
  int minbunch_;
  int maxbunch_;
  TFile *histfile_;

  const static int nMaxH=10;
  MonitorElement * nrPileupsH_[nMaxH];
  MonitorElement * nrVerticesH_[nMaxH];
  MonitorElement * nrTracksH_[nMaxH];
  MonitorElement * trackPartIdH_[nMaxH];
  MonitorElement * caloEnergyEBH_[nMaxH];
  MonitorElement * caloEnergyEEH_[nMaxH];

  DQMStore* dbe_;
  
  const static int nrHistos=6;
  char * labels[nrHistos];

  edm::InputTag cfTrackTag_;
  edm::InputTag cfVertexTag_;
};


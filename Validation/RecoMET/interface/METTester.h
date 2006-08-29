#ifndef METTESTER_H
#define METTESTER_H

// author: Mike Schmitt (The University of Florida)
// date: 8/24/2006

#include <memory>
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

//
// class declaration
//

using namespace cms;
using namespace edm;
using namespace std;

class METTester : public edm::EDAnalyzer {
public:

  explicit METTester(const edm::ParameterSet&);
  ~METTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 
 private:

  DaqMonitorBEInterface* dbe_;
  string outputFile_;

  // CaloMET Quantities
  MonitorElement* meCaloMEx;
  MonitorElement* meCaloMEy;
  MonitorElement* meCaloMET;
  MonitorElement* meCaloEz;
  MonitorElement* meCaloMETSig;
  MonitorElement* meCaloMETPhi;
  MonitorElement* meCaloSumET;
  MonitorElement* meCaloMaxEtInEmTowers;
  MonitorElement* meCaloMaxEtInHadTowers;
  MonitorElement* meCaloEtFractionHadronic;
  MonitorElement* meCaloEmEtFraction;
  MonitorElement* meCaloHadEtInHB;
  MonitorElement* meCaloHadEtInHO;
  MonitorElement* meCaloHadEtInHE;
  MonitorElement* meCaloHadEtInHF;
  MonitorElement* meCaloEmEtInEB;
  MonitorElement* meCaloEmEtInEE;
  MonitorElement* meCaloEmEtInHF;

  // GenMET Quantities
  MonitorElement* meGenMEx;
  MonitorElement* meGenMEy;
  MonitorElement* meGenMET;
  MonitorElement* meGenMETPhi;
  MonitorElement* meGenEz;
  MonitorElement* meGenMETSig;
  MonitorElement* meGenSumET;
  MonitorElement* meGenEmEnergy;
  MonitorElement* meGenHadEnergy;
  MonitorElement* meGenInvisibleEnergy;
  MonitorElement* meGenAuxiliaryEnergy;

  // Sigma := fabs(CaloMET - GenMET)
  MonitorElement* meMETSigmaVsGenSumET;

};

#endif

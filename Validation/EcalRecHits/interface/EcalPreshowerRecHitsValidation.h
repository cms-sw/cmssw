#ifndef EcalPreshowerRecHitsValidation_H
#define EcalPreshowerRecHitsValidation_H

/*
 * \file EcalPreshowerRecHitsValidation.h
 *
 * $Date: 2006/05/22 $
 * \author C. Rovelli
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace cms;
using namespace edm;
using namespace std;

class EcalPreshowerRecHitsValidation: public EDAnalyzer{

public:

/// Constructor
EcalPreshowerRecHitsValidation(const ParameterSet& ps);

/// Destructor
~EcalPreshowerRecHitsValidation();

protected:

/// Analyze
void analyze(const Event& e, const EventSetup& c);

// BeginJob
void beginJob(const EventSetup& c);

// EndJob
void endJob(void);

private:

 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;

 string ESrecHitProducer_;
 string ESrechitCollection_;

 MonitorElement* meESRecHitsEnergy_;    
 MonitorElement* meESRecHitsStripOccupancy_[2][2][32];
};

#endif

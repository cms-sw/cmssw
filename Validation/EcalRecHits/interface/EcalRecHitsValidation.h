#ifndef EcalRecHitsValidation_H
#define EcalRecHitsValidation_H

/*
 * \file EcalRecHitsValidation.h
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

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace cms;
using namespace edm;
using namespace std;

class EcalRecHitsValidation: public EDAnalyzer{

  typedef map<uint32_t,float,less<uint32_t> >  MapType;

public:

/// Constructor
EcalRecHitsValidation(const ParameterSet& ps);

/// Destructor
~EcalRecHitsValidation();

protected:

/// Analyze
void analyze(const Event& e, const EventSetup& c);

// BeginJob
void beginJob(const EventSetup& c);

// EndJob
void endJob(void);

private:

 string HepMCLabel;
 
 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;
 
 string outputFile_;

 string recHitProducer_;
 string ESrecHitProducer_;
 string EBrechitCollection_;
 string EErechitCollection_;
 string ESrechitCollection_;
 string uncalibrecHitProducer_;
 string EBuncalibrechitCollection_;
 string EEuncalibrechitCollection_;
 
 MonitorElement* meGunEnergy_;
 MonitorElement* meGunEta_;
 MonitorElement* meGunPhi_;   
 MonitorElement* meEBRecHitSimHitRatio_;
 MonitorElement* meEERecHitSimHitRatio_;
 MonitorElement* meESRecHitSimHitRatio_;
};

#endif

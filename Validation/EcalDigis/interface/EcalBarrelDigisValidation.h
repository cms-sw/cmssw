#ifndef EcalBarrelDigisValidation_H
#define EcalBarrelDigisValidation_H

/*
 * \file EcalBarrelDigisValidation.h
 *
 * $Date: 2006/07/26 14:55:26 $
 * $Revision: 1.2 $
 * \author F. Cossutti
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

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace cms;
using namespace edm;
using namespace std;

class EcalBarrelDigisValidation: public EDAnalyzer{

    typedef map<uint32_t,float,less<uint32_t> >  MapType;

public:

/// Constructor
EcalBarrelDigisValidation(const ParameterSet& ps);

/// Destructor
~EcalBarrelDigisValidation();

protected:

/// Analyze
void analyze(const Event& e, const EventSetup& c);

// BeginJob
void beginJob(const EventSetup& c);

// EndJob
void endJob(void);

void checkCalibrations(const edm::EventSetup & c);

private:

 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;
 
 string outputFile_;

 edm::InputTag EBdigiCollection_;
 
 map<int, double, less<int> > gainConv_;

 double barrelADCtoGeV_;
 double endcapADCtoGeV_;
 
 MonitorElement* meEBDigiOccupancy_;

 MonitorElement* meEBDigiMultiplicity_;

 MonitorElement* meEBDigiADCGlobal_;

 MonitorElement* meEBDigiADCAnalog_[10];

 MonitorElement* meEBDigiADCg1_[10];
 MonitorElement* meEBDigiADCg6_[10];
 MonitorElement* meEBDigiADCg12_[10];

 MonitorElement* meEBDigiGain_[10];

 MonitorElement* meEBPedestal_;

 MonitorElement* meEBMaximumgt100ADC_;

 MonitorElement* meEBMaximumgt10ADC_;

 MonitorElement* meEBnADCafterSwitch_;

};

#endif

#ifndef EcalEndcapDigisValidation_H
#define EcalEndcapDigisValidation_H

/*
 * \file EcalEndcapDigisValidation.h
 *
 * $Date: 2006/10/05 13:19:02 $
 * $Revision: 1.3 $
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

#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace cms;
using namespace edm;
using namespace std;

class EcalEndcapDigisValidation: public EDAnalyzer{

    typedef map<uint32_t,float,less<uint32_t> >  MapType;

public:

/// Constructor
EcalEndcapDigisValidation(const ParameterSet& ps);

/// Destructor
~EcalEndcapDigisValidation();

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
 
 edm::InputTag EEdigiCollection_;

 map<int, double, less<int> > gainConv_;

 double barrelADCtoGeV_;
 double endcapADCtoGeV_;
 
 MonitorElement* meEEDigiOccupancyzp_;
 MonitorElement* meEEDigiOccupancyzm_;
 
 MonitorElement* meEEDigiMultiplicityzp_;
 MonitorElement* meEEDigiMultiplicityzm_;

 MonitorElement* meEEDigiADCGlobal_;

 MonitorElement* meEEDigiADCAnalog_[10];

 MonitorElement* meEEDigiADCg1_[10];
 MonitorElement* meEEDigiADCg6_[10];
 MonitorElement* meEEDigiADCg12_[10];

 MonitorElement* meEEDigiGain_[10];

 MonitorElement* meEEPedestal_;

 MonitorElement* meEEMaximumgt100ADC_; 

 MonitorElement* meEEMaximumgt20ADC_; 

 MonitorElement* meEEnADCafterSwitch_;

};

#endif

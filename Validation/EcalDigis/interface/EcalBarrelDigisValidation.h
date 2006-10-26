#ifndef EcalBarrelDigisValidation_H
#define EcalBarrelDigisValidation_H

/*
 * \file EcalBarrelDigisValidation.h
 *
 * $Date: 2006/10/13 13:13:14 $
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

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class EcalBarrelDigisValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalBarrelDigisValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalBarrelDigisValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

void checkCalibrations(const edm::EventSetup & c);

private:

 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;
 
 std::string outputFile_;

 edm::InputTag EBdigiCollection_;
 
 std::map<int, double, std::less<int> > gainConv_;

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

#ifndef EcalEndcapDigisValidation_H
#define EcalEndcapDigisValidation_H

/*
 * \file EcalEndcapDigisValidation.h
 *
 * $Date: 2010/01/04 15:10:59 $
 * $Revision: 1.9 $
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

class EcalEndcapDigisValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalEndcapDigisValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalEndcapDigisValidation();

protected:

/// Analyze
void analyze(edm::Event const & e, edm::EventSetup const & c);

// BeginRun
void beginRun(edm::Run const &, edm::EventSetup const & c);

// EndJob
void endJob(void);


void checkCalibrations(edm::EventSetup const & c);

private:

 bool verbose_;
 
 DQMStore* dbe_;
 
 std::string outputFile_;
 
 edm::InputTag EEdigiCollection_;

 std::map<int, double, std::less<int> > gainConv_;

 double barrelADCtoGeV_;
 double endcapADCtoGeV_;
 
 MonitorElement* meEEDigiOccupancyzp_;
 MonitorElement* meEEDigiOccupancyzm_;
 
 MonitorElement* meEEDigiMultiplicityzp_;
 MonitorElement* meEEDigiMultiplicityzm_;

 MonitorElement* meEEDigiADCGlobal_;

 MonitorElement* meEEDigiADCAnalog_[10];

 MonitorElement* meEEDigiADCgS_[10];
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

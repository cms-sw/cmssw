#ifndef EcalShashlikDigisValidation_H
#define EcalShashlikDigisValidation_H

/*
 * \file EcalShashlikDigisValidation.h
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

#include "DataFormats/EcalDigi/interface/EKDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

class EcalShashlikDigisValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalShashlikDigisValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalShashlikDigisValidation();

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
 
 edm::InputTag EKdigiCollection_;

 std::map<int, double, std::less<int> > gainConv_;

 double barrelADCtoGeV_;
 double shashlikADCtoGeV_;
 
 MonitorElement* meEKDigiOccupancyzp_;
 MonitorElement* meEKDigiOccupancyzm_;
 
 MonitorElement* meEKDigiMultiplicityzp_;
 MonitorElement* meEKDigiMultiplicityzm_;

 MonitorElement* meEKDigiADCGlobal_;

 MonitorElement* meEKDigiADCAnalog_[10];

 MonitorElement* meEKDigiADCgS_[10];
 MonitorElement* meEKDigiADCg1_[10];
 MonitorElement* meEKDigiADCg6_[10];
 MonitorElement* meEKDigiADCg12_[10];

 MonitorElement* meEKDigiGain_[10];

 MonitorElement* meEKPedestal_;

 MonitorElement* meEKMaximumgt100ADC_; 

 MonitorElement* meEKMaximumgt20ADC_; 

 MonitorElement* meEKnADCafterSwitch_;

};

#endif

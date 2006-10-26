#ifndef EcalBarrelRecHitsValidation_H
#define EcalBarrelRecHitsValidation_H

/*
 * \file EcalBarrelRecHitsValidation.h
 *
 * $Date: 2006/10/17 09:56:12 $
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

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class EcalBarrelRecHitsValidation: public edm::EDAnalyzer{

public:

/// Constructor
EcalBarrelRecHitsValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalBarrelRecHitsValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;

 edm::InputTag EBdigiCollection_;
 edm::InputTag EBuncalibrechitCollection_;

 MonitorElement* meEBUncalibRecHitsOccupancy_;     
 MonitorElement* meEBUncalibRecHitsAmplitude_;    
 MonitorElement* meEBUncalibRecHitsPedestal_;      
 MonitorElement* meEBUncalibRecHitsJitter_;        
 MonitorElement* meEBUncalibRecHitsChi2_;          
 MonitorElement* meEBUncalibRecHitMaxSampleRatio_;
 MonitorElement* meEBUncalibRecHitsOccupancyGt100adc_;     
 MonitorElement* meEBUncalibRecHitsAmplitudeGt100adc_;    
 MonitorElement* meEBUncalibRecHitsPedestalGt100adc_;      
 MonitorElement* meEBUncalibRecHitsJitterGt100adc_;        
 MonitorElement* meEBUncalibRecHitsChi2Gt100adc_;          
 MonitorElement* meEBUncalibRecHitMaxSampleRatioGt100adc_;
 MonitorElement* meEBUncalibRecHitsAmpFullMap_;
 MonitorElement* meEBUncalibRecHitsPedFullMap_;
 MonitorElement* meEBUncalibRecHitAmplMap_[36];
 MonitorElement* meEBUncalibRecHitPedMap_[36];
};

#endif

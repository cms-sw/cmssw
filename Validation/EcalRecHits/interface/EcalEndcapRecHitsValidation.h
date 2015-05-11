#ifndef EcalEndcapRecHitsValidation_H
#define EcalEndcapRecHitsValidation_H

/*
 * \file EcalEndcapRecHitsValidation.h
 *
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class EcalEndcapRecHitsValidation: public DQMEDAnalyzer{

public:

/// Constructor
EcalEndcapRecHitsValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalEndcapRecHitsValidation();

protected:

void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

 bool verbose_;

 // fix for consumes
 edm::EDGetTokenT< EEDigiCollection > EEdigiCollection_token_;
 edm::EDGetTokenT< EEUncalibratedRecHitCollection > EEuncalibrechitCollection_token_;

 MonitorElement* meEEUncalibRecHitsOccupancyPlus_;     
 MonitorElement* meEEUncalibRecHitsOccupancyMinus_;     
 MonitorElement* meEEUncalibRecHitsAmplitude_;    
 MonitorElement* meEEUncalibRecHitsPedestal_;      
 MonitorElement* meEEUncalibRecHitsJitter_;        
 MonitorElement* meEEUncalibRecHitsChi2_;          
 MonitorElement* meEEUncalibRecHitMaxSampleRatio_; 
 MonitorElement* meEEUncalibRecHitsOccupancyPlusGt60adc_;     
 MonitorElement* meEEUncalibRecHitsOccupancyMinusGt60adc_;     
 MonitorElement* meEEUncalibRecHitsAmplitudeGt60adc_;    
 MonitorElement* meEEUncalibRecHitsPedestalGt60adc_;      
 MonitorElement* meEEUncalibRecHitsJitterGt60adc_;        
 MonitorElement* meEEUncalibRecHitsChi2Gt60adc_;          
 MonitorElement* meEEUncalibRecHitMaxSampleRatioGt60adc_; 
 MonitorElement* meEEUncalibRecHitsAmpFullMap_;
 MonitorElement* meEEUncalibRecHitsPedFullMap_;
};

#endif

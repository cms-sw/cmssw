#ifndef EcalBarrelDigisValidation_H
#define EcalBarrelDigisValidation_H

/*
 * \file EcalBarrelDigisValidation.h
 *
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

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


class EcalBarrelDigisValidation: public DQMEDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalBarrelDigisValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalBarrelDigisValidation();

void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;

protected:

/// Analyze
void analyze(edm::Event const & e, edm::EventSetup const & c) override;
void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
void checkCalibrations(edm::EventSetup const & c);


private:

 bool verbose_;
 
 std::string outputFile_;

 edm::EDGetTokenT<EBDigiCollection> EBdigiCollection_;
 
 std::map<int, double, std::less<int> > gainConv_;

 double barrelADCtoGeV_;
 double endcapADCtoGeV_;
 
 MonitorElement* meEBDigiOccupancy_;

 MonitorElement* meEBDigiMultiplicity_;

 MonitorElement* meEBDigiADCGlobal_;

 MonitorElement* meEBDigiADCAnalog_[10];

 MonitorElement* meEBDigiADCgS_[10];
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

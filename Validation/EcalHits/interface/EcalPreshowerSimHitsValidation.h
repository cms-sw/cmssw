#ifndef EcalPreshowerSimHitsValidation_H
#define EcalPreshowerSimHitsValidation_H

/*
 * \file EcalPreshowerSimHitsValidation.h
 *
 * \author C.Rovelli
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
#include "SimDataFormats/EcalValidation/interface/PEcalValidInfo.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>


class EcalPreshowerSimHitsValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalPreshowerSimHitsValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalPreshowerSimHitsValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

 std::string HepMCLabel; 
 std::string g4InfoLabel; 
 std::string EEHitsCollection;
 std::string ESHitsCollection;
 
 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;
 
 std::string outputFile_;

 MonitorElement* menESHits1zp_;
 MonitorElement* menESHits2zp_;

 MonitorElement* menESHits1zm_;
 MonitorElement* menESHits2zm_;

 MonitorElement* meESEnergyHits1zp_;
 MonitorElement* meESEnergyHits2zp_;

 MonitorElement* meESEnergyHits1zm_;
 MonitorElement* meESEnergyHits2zm_;

 MonitorElement* meE1alphaE2zp_;
 MonitorElement* meE1alphaE2zm_;

 MonitorElement* meEEoverESzp_;
 MonitorElement* meEEoverESzm_;

 MonitorElement* me2eszpOver1eszp_; 
 MonitorElement* me2eszmOver1eszm_; 

};

#endif

#ifndef EcalPreshowerDigisValidation_H
#define EcalPreshowerDigisValidation_H

/*
 * \file EcalPreshowerDigisValidation.h
 *
 * $Date: 2006/10/13 13:13:14 $
 * $Revision: 1.4 $
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

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class EcalPreshowerDigisValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalPreshowerDigisValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalPreshowerDigisValidation();

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
 
 std::string outputFile_;

 edm::InputTag ESdigiCollection_;

 MonitorElement* meESDigiMultiplicity_;
 
 MonitorElement* meESDigiADC_[3];

};

#endif

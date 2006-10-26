#ifndef EcalDigisValidation_H
#define EcalDigisValidation_H

/*
 * \file EcalDigisValidation.h
 *
 * $Date: 2006/10/16 13:13:53 $
 * $Revision: 1.8 $
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

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class EcalDigisValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalDigisValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalDigisValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

 void checkCalibrations(const edm::EventSetup & c);
 
 std::string HepMCLabel;
 std::string g4InfoLabel;
 
 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;
 
 std::string outputFile_;

 edm::InputTag EBdigiCollection_;
 edm::InputTag EEdigiCollection_;
 edm::InputTag ESdigiCollection_;
 
 std::map<int, double, std::less<int> > gainConv_;

 double barrelADCtoGeV_;
 double endcapADCtoGeV_;
 
 MonitorElement* meGunEnergy_;
 MonitorElement* meGunEta_;
 MonitorElement* meGunPhi_;   

 MonitorElement* meEBDigiSimRatio_;
 MonitorElement* meEEDigiSimRatio_;

 MonitorElement* meEBDigiSimRatiogt10ADC_;
 MonitorElement* meEEDigiSimRatiogt20ADC_;

 MonitorElement* meEBDigiSimRatiogt100ADC_;
 MonitorElement* meEEDigiSimRatiogt100ADC_;

};

#endif

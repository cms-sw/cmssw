#ifndef EcalMixingModuleValidation_H
#define EcalMixingModuleValidation_H

/*
 * \file EcalMixingModuleValidation.h
 *
 * $Date: 2006/10/18 15:04:00 $
 * $Revision: 1.2 $
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

#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace cms;
using namespace edm;
using namespace std;

class EcalMixingModuleValidation: public EDAnalyzer{

    typedef map<uint32_t,float,less<uint32_t> >  MapType;

public:

/// Constructor
EcalMixingModuleValidation(const ParameterSet& ps);

/// Destructor
~EcalMixingModuleValidation();

protected:

/// Analyze
void analyze(const Event& e, const EventSetup& c);

// BeginJob
void beginJob(const EventSetup& c);

// EndJob
void endJob(void);

private:

 void checkPedestals(const edm::EventSetup & c);

 void findPedestal(const DetId & detId, int gainId, double & ped) const;

 void checkCalibrations(const edm::EventSetup & c);
 
 string HepMCLabel;
 
 bool verbose_;

 DaqMonitorBEInterface* dbe_;
 
 string outputFile_;

 edm::InputTag EBdigiCollection_;
 edm::InputTag EEdigiCollection_;
 edm::InputTag ESdigiCollection_;
 
 map<int, double, less<int> > gainConv_;

 double barrelADCtoGeV_;
 double endcapADCtoGeV_;
 
 MonitorElement* meEBDigiMixRatiogt100ADC_;
 MonitorElement* meEEDigiMixRatiogt100ADC_;

 MonitorElement* meEBDigiMixRatioOriggt50pc_;
 MonitorElement* meEEDigiMixRatioOriggt40pc_;

 MonitorElement* meEBbunchCrossing_;
 MonitorElement* meEEbunchCrossing_;
 MonitorElement* meESbunchCrossing_;

 static const int nBunch = 21;

 MonitorElement* meEBBunchShape_[nBunch];
 MonitorElement* meEEBunchShape_[nBunch];
 MonitorElement* meESBunchShape_[nBunch];

 MonitorElement* meEBShape_;
 MonitorElement* meEEShape_;
 MonitorElement* meESShape_;

 MonitorElement* meEBShapeRatio_;
 MonitorElement* meEEShapeRatio_;
 MonitorElement* meESShapeRatio_;

 const EcalSimParameterMap * theParameterMap;
 const CaloVShape * theEcalShape;
 const ESShape * theESShape;

 CaloHitResponse * theEcalResponse;
 CaloHitResponse * theESResponse;
 
 void computeSDBunchDigi(const edm::EventSetup & eventSetup, MixCollection<PCaloHit> & theHits, MapType & ebSignalSimMap, const EcalSubdetector & thisDet, const double & theSimThreshold);

 void bunchSumTest(std::vector<MonitorElement *> & theBunches, MonitorElement* & theTotal, MonitorElement* & theRatio, int nSample);

 double esBaseline_;
 double esADCtokeV_;
 double esThreshold_;

 int theMinBunch;
 int theMaxBunch;

 const CaloGeometry * theGeometry;
 
 // the pedestals
 const EcalPedestals * thePedestals;

};

#endif

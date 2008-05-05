#ifndef EcalMixingModuleValidation_H
#define EcalMixingModuleValidation_H

/*
 * \file EcalMixingModuleValidation.h
 *
 * $Date: 2008/02/29 20:48:25 $
 * $Revision: 1.6 $
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
#include "DQMServices/Core/interface/MonitorElement.h"

class EcalMixingModuleValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalMixingModuleValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalMixingModuleValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

 void checkPedestals(const edm::EventSetup & c);

 void findPedestal(const DetId & detId, int gainId, double & ped) const;

 void checkCalibrations(const edm::EventSetup & c);
 
 std::string HepMCLabel;
 std::string hitsProducer_;

 bool verbose_;

 DQMStore* dbe_;
 
 std::string outputFile_;

 edm::InputTag EBdigiCollection_;
 edm::InputTag EEdigiCollection_;
 edm::InputTag ESdigiCollection_;
 
 std::map<int, double, std::less<int> > gainConv_;

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

#ifndef EcalRecHitsValidation_H
#define EcalRecHitsValidation_H

/*
 * \file EcalRecHitsValidation.h
 *
 * $Date: 2008/02/15 16:14:28 $
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
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class EcalRecHitsValidation: public edm::EDAnalyzer{

  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalRecHitsValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalRecHitsValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

 uint32_t getUnitWithMaxEnergy(MapType& themap);
 void     findBarrelMatrix(int nCellInEta, int nCellInPhi,
                           int CentralEta, int CentralPhi,int CentralZ,
                           MapType& themap); 
 void     findEndcapMatrix(int nCellInX, int nCellInY,
                           int CentralX, int CentralY,int CentralZ,
                           MapType&  themap);

private:

 std::string HepMCLabel;
 
 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;
 
 std::string outputFile_;

 edm::InputTag EBrechitCollection_;
 edm::InputTag EErechitCollection_;
 edm::InputTag ESrechitCollection_;
 edm::InputTag EBuncalibrechitCollection_;
 edm::InputTag EEuncalibrechitCollection_;
 
 MonitorElement* meGunEnergy_;
 MonitorElement* meGunEta_;
 MonitorElement* meGunPhi_;   
 MonitorElement* meEBRecHitSimHitRatio_;
 MonitorElement* meEERecHitSimHitRatio_;
 MonitorElement* meESRecHitSimHitRatio_;
 MonitorElement* meEBRecHitSimHitRatioGt35_;
 MonitorElement* meEERecHitSimHitRatioGt35_;
 MonitorElement* meEBUnRecHitSimHitRatio_;
 MonitorElement* meEEUnRecHitSimHitRatio_;
 MonitorElement* meEBUnRecHitSimHitRatioGt35_;
 MonitorElement* meEEUnRecHitSimHitRatioGt35_;
 MonitorElement* meEBe5x5_;
 MonitorElement* meEBe5x5OverSimHits_;
 MonitorElement* meEBe5x5OverGun_;
 MonitorElement* meEEe5x5_;
 MonitorElement* meEEe5x5OverSimHits_;
 MonitorElement* meEEe5x5OverGun_;

 std::vector<uint32_t> crystalMatrix;

};

#endif

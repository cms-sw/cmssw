#ifndef EcalEndcapSimHitsValidation_H
#define EcalEndcapSimHitsValidation_H

/*
 * \file EcalEndcapSimHitsValidation.h
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

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/EcalValidation/interface/PEcalValidInfo.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>


class EcalEndcapSimHitsValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalEndcapSimHitsValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalEndcapSimHitsValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:

 uint32_t getUnitWithMaxEnergy(MapType& themap);

 virtual float energyInMatrixEE(int nCellInX, int nCellInY, 
                                int centralX, int centralY, int centralZ,
                                MapType& themap); 
 
 bool  fillEEMatrix(int nCellInX, int nCellInY,
                    int CentralX, int CentralY,int CentralZ,
                    MapType& fillmap, MapType&  themap);

 float eCluster2x2( MapType& themap);
 float eCluster4x4(float e33,MapType& themap);

 std::string g4InfoLabel;
 std::string EEHitsCollection;
 std::string ValidationCollection;
 
 bool verbose_;
 
 DaqMonitorBEInterface* dbe_;
 
 std::string outputFile_;

 int myEntries;
 float eRLength[26];

 MonitorElement* meEEzpHits_;
 MonitorElement* meEEzmHits_;

 MonitorElement* meEEzpCrystals_;
 MonitorElement* meEEzmCrystals_;

 MonitorElement* meEEzpOccupancy_;
 MonitorElement* meEEzmOccupancy_;

 MonitorElement* meEELongitudinalShower_;

 MonitorElement* meEEzpHitEnergy_;
 MonitorElement* meEEzmHitEnergy_;

 MonitorElement* meEEe1_; 
 MonitorElement* meEEe4_; 
 MonitorElement* meEEe9_; 
 MonitorElement* meEEe16_; 
 MonitorElement* meEEe25_; 

 MonitorElement* meEEe1oe4_;
 MonitorElement* meEEe4oe9_;
 MonitorElement* meEEe9oe16_;
 MonitorElement* meEEe1oe25_;
 MonitorElement* meEEe9oe25_;
 MonitorElement* meEEe16oe25_;
};

#endif

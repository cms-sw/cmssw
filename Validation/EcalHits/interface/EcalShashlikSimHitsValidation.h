#ifndef EcalShashlikSimHitsValidation_H
#define EcalShashlikSimHitsValidation_H

/*
 * \file EcalShashlikSimHitsValidation.h
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

#include "Geometry/Records/interface/ShashlikNumberingRecord.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"


class EcalShashlikSimHitsValidation: public edm::EDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalShashlikSimHitsValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalShashlikSimHitsValidation();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob();

// EndJob
void endJob(void);

private:

 uint32_t getUnitWithMaxEnergy(MapType& themap);

 virtual float energyInMatrixEK(int nCellInX, int nCellInY, 
                                int centralX, int centralY, int centralZ,
                                MapType& themap); 
 
 std::vector<uint32_t> getIdsAroundMax(int nCellInX, int nCellInY, 
                                int centralX, int centralY, int centralZ,
                                MapType& themap); 

 bool  fillEKMatrix(int nCellInX, int nCellInY,
                    int CentralX, int CentralY,int CentralZ,
                    MapType& fillmap, MapType&  themap);

 float eCluster2x2( MapType& themap);
 float eCluster4x4(float e33,MapType& themap);

 std::string g4InfoLabel;
 std::string EKHitsCollection;
 std::string ValidationCollection;
 
 bool verbose_;
 
 DQMStore* dbe_;
 
 std::string outputFile_;

 int myEntries;
 float eRLength[26];

 MonitorElement* meEKzpHits_;
 MonitorElement* meEKzmHits_;

 MonitorElement* meEKzpCrystals_;
 MonitorElement* meEKzmCrystals_;

 MonitorElement* meEKzpOccupancy_;
 MonitorElement* meEKzmOccupancy_;

 MonitorElement* meEKLongitudinalShower_;

 MonitorElement* meEKHitEnergy_;

 MonitorElement* meEKhitLog10Energy_;

 MonitorElement* meEKhitLog10EnergyNorm_;

 MonitorElement* meEKhitLog10Energy25Norm_;


 MonitorElement* meEKHitEnergy2_;

 MonitorElement* meEKcrystalEnergy_;
 MonitorElement* meEKcrystalEnergy2_;

 MonitorElement* meEKe1_; 
 MonitorElement* meEKe4_; 
 MonitorElement* meEKe9_; 
 MonitorElement* meEKe16_; 
 MonitorElement* meEKe25_; 

 MonitorElement* meEKe1oe4_;
 MonitorElement* meEKe1oe9_;
 MonitorElement* meEKe4oe9_;
 MonitorElement* meEKe9oe16_;
 MonitorElement* meEKe1oe25_;
 MonitorElement* meEKe9oe25_;
 MonitorElement* meEKe16oe25_;
 const ShashlikTopology *_topology;
};

#endif

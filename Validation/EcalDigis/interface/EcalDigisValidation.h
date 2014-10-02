#ifndef EcalDigisValidation_H
#define EcalDigisValidation_H

/*
 * \file EcalDigisValidation.h
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

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
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

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class EcalDigisValidation: public DQMEDAnalyzer{

    typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

/// Constructor
EcalDigisValidation(const edm::ParameterSet& ps);

/// Destructor
~EcalDigisValidation();

void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;

protected:

/// Analyze
void analyze(edm::Event const & e, edm::EventSetup const & c);
void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override; 

private:

 void checkCalibrations(edm::EventSetup const & c);
  
 bool verbose_;
 
 std::string outputFile_;

 edm::EDGetTokenT<edm::HepMCProduct> HepMCToken_;
 edm::EDGetTokenT<edm::SimTrackContainer> g4TkInfoToken_;
 edm::EDGetTokenT<edm::SimVertexContainer> g4VtxInfoToken_;

 edm::EDGetTokenT<EBDigiCollection> EBdigiCollectionToken_;
 edm::EDGetTokenT<EEDigiCollection> EEdigiCollectionToken_;
 edm::EDGetTokenT<ESDigiCollection> ESdigiCollectionToken_;
 
 edm::EDGetTokenT< CrossingFrame<PCaloHit> > crossingFramePCaloHitEBToken_, crossingFramePCaloHitEEToken_, crossingFramePCaloHitESToken_;
 
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

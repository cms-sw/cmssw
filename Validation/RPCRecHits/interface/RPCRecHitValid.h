#ifndef RPCRecHitValid_h
#define RPCRecHitValid_h

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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include <iostream>
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"

class  RPCRecHitValid: public edm::EDAnalyzer {

 public:

  RPCRecHitValid(const edm::ParameterSet& ps);
    ~RPCRecHitValid();

 protected:
     void analyze(const edm::Event& e, const edm::EventSetup& c);
     void beginJob();
     void endJob(void);

 private:

     MonitorElement* Res;
     MonitorElement* ResWmin2;
     MonitorElement* ResWmin1;
     MonitorElement* ResWzer0;
     MonitorElement* ResWplu1;
     MonitorElement* ResWplu2;
     MonitorElement* ResS1;
     MonitorElement* ResS3;

     MonitorElement* Rechisto;
     MonitorElement* Simhisto;
     MonitorElement* Pulls;
     MonitorElement* ClSize;
     
     MonitorElement* res1cl;
     
     MonitorElement* occRB1IN;
     MonitorElement* occRB1OUT;
     
     DQMStore* dbe_;
     std::string rootFileName;

};




#endif

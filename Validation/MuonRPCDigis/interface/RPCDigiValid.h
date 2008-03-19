#ifndef RPCDigiValid_h
#define RPCDigiValid_h

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
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <iostream>
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"

class  RPCDigiValid: public edm::EDAnalyzer {

 public:

  RPCDigiValid(const edm::ParameterSet& ps);
    ~RPCDigiValid();

 protected:
     void analyze(const edm::Event& e, const edm::EventSetup& c);
     void beginJob(const edm::EventSetup& c);
     void endJob(void);

 private:

    MonitorElement* xyview;
    MonitorElement* rzview;
    MonitorElement* Res;
    MonitorElement* ResWmin2;
    MonitorElement* ResWmin1;
    MonitorElement* ResWzer0;
    MonitorElement* ResWplu1;
    MonitorElement* ResWplu2;
    MonitorElement* BxDist;
    MonitorElement* StripProf;

    DQMStore* dbe_;
    std::string outputFile_;
    std::string digiLabel;
};




#endif


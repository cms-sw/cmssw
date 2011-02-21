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
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"

class  RPCDigiValid: public edm::EDAnalyzer {

 public:

  RPCDigiValid(const edm::ParameterSet& ps);
    ~RPCDigiValid();

 protected:
     void analyze(const edm::Event& e, const edm::EventSetup& c);
     void beginJob();
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

//members for EndCap's disks:
    MonitorElement* ResDmin1;
    MonitorElement* ResDmin2;
    MonitorElement* ResDmin3;
    MonitorElement* ResDplu1;
    MonitorElement* ResDplu2;
    MonitorElement* ResDplu3;

//members for X_vs_Y_view:
    MonitorElement* xyvWmin2;
    MonitorElement* xyvWmin1;
    MonitorElement* xyvWzer0;
    MonitorElement* xyvWplu1;
    MonitorElement* xyvWplu2;

    MonitorElement* xyvDplu1;
    MonitorElement* xyvDplu2;
    MonitorElement* xyvDplu3;

    MonitorElement* xyvDmin1;
    MonitorElement* xyvDmin2;
    MonitorElement* xyvDmin3;

    DQMStore* dbe_;
    std::string outputFile_;
    std::string digiLabel;
};




#endif


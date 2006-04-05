#ifndef SiStripDigiValid_h
#define SiStripDigiValid_h

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

#include <string>

using namespace std;
using namespace edm;


class  SiStripDigiValid: public EDAnalyzer {

 public:

    SiStripDigiValid(const ParameterSet& ps);
    ~SiStripDigiValid();

 protected:
     void analyze(const Event& e, const EventSetup& c);
     void beginJob(const EventSetup& c);
     void endJob(void);

 private:

 
    //TIB  ADC
    MonitorElement* meAdcTIBLayer1_;
    MonitorElement* meAdcTIBLayer2_;
    MonitorElement* meAdcTIBLayer3_;
    MonitorElement* meAdcTIBLayer4_;
    //TIB Strip
    MonitorElement* meStripTIBLayer1_;
    MonitorElement* meStripTIBLayer2_;
    MonitorElement* meStripTIBLayer3_;
    MonitorElement* meStripTIBLayer4_;
    //TIB Mulitiplicity
    MonitorElement* meNDigiTIBLayer_;
    MonitorElement* meNDigiTOBLayer_;
    MonitorElement* meNDigiTIDWheel_;
    MonitorElement* meNDigiTECWheel_;
 
    //TOB ADC
    MonitorElement* meAdcTOBLayer1_;
    MonitorElement* meAdcTOBLayer2_;
    MonitorElement* meAdcTOBLayer3_;
    MonitorElement* meAdcTOBLayer4_;
    MonitorElement* meAdcTOBLayer5_;
    MonitorElement* meAdcTOBLayer6_; 
    //TOB Strip
    MonitorElement* meStripTOBLayer1_;
    MonitorElement* meStripTOBLayer2_;
    MonitorElement* meStripTOBLayer3_;
    MonitorElement* meStripTOBLayer4_;
    MonitorElement* meStripTOBLayer5_;
    MonitorElement* meStripTOBLayer6_;
    //TID  ADC
    MonitorElement* meAdcTIDWheel1_;
    MonitorElement* meAdcTIDWheel2_;
    MonitorElement* meAdcTIDWheel3_;
    //TID Strip
    MonitorElement* meStripTIDWheel1_;
    MonitorElement* meStripTIDWheel2_;
    MonitorElement* meStripTIDWheel3_;
    //TEC ADC
    MonitorElement* meAdcTECWheel1_;
    MonitorElement* meAdcTECWheel2_;
    MonitorElement* meAdcTECWheel3_;
    MonitorElement* meAdcTECWheel4_;
    MonitorElement* meAdcTECWheel5_;
    MonitorElement* meAdcTECWheel6_;
    MonitorElement* meAdcTECWheel7_;
    MonitorElement* meAdcTECWheel8_;
    MonitorElement* meAdcTECWheel9_;
    //TEC Strip
    MonitorElement* meStripTECWheel1_;
    MonitorElement* meStripTECWheel2_;
    MonitorElement* meStripTECWheel3_;
    MonitorElement* meStripTECWheel4_;
    MonitorElement* meStripTECWheel5_;
    MonitorElement* meStripTECWheel6_;
    MonitorElement* meStripTECWheel7_;
    MonitorElement* meStripTECWheel8_;
    MonitorElement* meStripTECWheel9_;



    //Back-End Interface
    DaqMonitorBEInterface* dbe_;
    string outputFile_;

};




#endif


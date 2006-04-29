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
    MonitorElement* meAdcTIBLayer1zp_;
    MonitorElement* meAdcTIBLayer2zp_;
    MonitorElement* meAdcTIBLayer3zp_;
    MonitorElement* meAdcTIBLayer4zp_;

    MonitorElement* meAdcTIBLayer1zm_;
    MonitorElement* meAdcTIBLayer2zm_;
    MonitorElement* meAdcTIBLayer3zm_;
    MonitorElement* meAdcTIBLayer4zm_;

    //TIB Strip
    MonitorElement* meStripTIBLayer1zp_;
    MonitorElement* meStripTIBLayer2zp_;
    MonitorElement* meStripTIBLayer3zp_;
    MonitorElement* meStripTIBLayer4zp_;

    MonitorElement* meStripTIBLayer1zm_;
    MonitorElement* meStripTIBLayer2zm_;
    MonitorElement* meStripTIBLayer3zm_;
    MonitorElement* meStripTIBLayer4zm_;

    //TOB ADC
    MonitorElement* meAdcTOBLayer1zp_;
    MonitorElement* meAdcTOBLayer2zp_;
    MonitorElement* meAdcTOBLayer3zp_;
    MonitorElement* meAdcTOBLayer4zp_;
    MonitorElement* meAdcTOBLayer5zp_;
    MonitorElement* meAdcTOBLayer6zp_; 

    MonitorElement* meAdcTOBLayer1zm_;
    MonitorElement* meAdcTOBLayer2zm_;
    MonitorElement* meAdcTOBLayer3zm_;
    MonitorElement* meAdcTOBLayer4zm_;
    MonitorElement* meAdcTOBLayer5zm_;
    MonitorElement* meAdcTOBLayer6zm_;

    //TOB Strip
    MonitorElement* meStripTOBLayer1zp_;
    MonitorElement* meStripTOBLayer2zp_;
    MonitorElement* meStripTOBLayer3zp_;
    MonitorElement* meStripTOBLayer4zp_;
    MonitorElement* meStripTOBLayer5zp_;
    MonitorElement* meStripTOBLayer6zp_;

    MonitorElement* meStripTOBLayer1zm_;
    MonitorElement* meStripTOBLayer2zm_;
    MonitorElement* meStripTOBLayer3zm_;
    MonitorElement* meStripTOBLayer4zm_;
    MonitorElement* meStripTOBLayer5zm_;
    MonitorElement* meStripTOBLayer6zm_;


    //TID  ADC
    MonitorElement* meAdcTIDWheel1zp_;
    MonitorElement* meAdcTIDWheel2zp_;
    MonitorElement* meAdcTIDWheel3zp_;

    MonitorElement* meAdcTIDWheel1zm_;
    MonitorElement* meAdcTIDWheel2zm_;
    MonitorElement* meAdcTIDWheel3zm_;

    //TID Strip
    MonitorElement* meStripTIDWheel1zp_;
    MonitorElement* meStripTIDWheel2zp_;
    MonitorElement* meStripTIDWheel3zp_;

    MonitorElement* meStripTIDWheel1zm_;
    MonitorElement* meStripTIDWheel2zm_;
    MonitorElement* meStripTIDWheel3zm_;

    //TEC ADC
    MonitorElement* meAdcTECWheel1zp_;
    MonitorElement* meAdcTECWheel2zp_;
    MonitorElement* meAdcTECWheel3zp_;
    MonitorElement* meAdcTECWheel4zp_;
    MonitorElement* meAdcTECWheel5zp_;
    MonitorElement* meAdcTECWheel6zp_;
    MonitorElement* meAdcTECWheel7zp_;
    MonitorElement* meAdcTECWheel8zp_;
    MonitorElement* meAdcTECWheel9zp_;

    MonitorElement* meAdcTECWheel1zm_;
    MonitorElement* meAdcTECWheel2zm_;
    MonitorElement* meAdcTECWheel3zm_;
    MonitorElement* meAdcTECWheel4zm_;
    MonitorElement* meAdcTECWheel5zm_;
    MonitorElement* meAdcTECWheel6zm_;
    MonitorElement* meAdcTECWheel7zm_;
    MonitorElement* meAdcTECWheel8zm_;
    MonitorElement* meAdcTECWheel9zm_;

    //TEC Strip
    MonitorElement* meStripTECWheel1zp_;
    MonitorElement* meStripTECWheel2zp_;
    MonitorElement* meStripTECWheel3zp_;
    MonitorElement* meStripTECWheel4zp_;
    MonitorElement* meStripTECWheel5zp_;
    MonitorElement* meStripTECWheel6zp_;
    MonitorElement* meStripTECWheel7zp_;
    MonitorElement* meStripTECWheel8zp_;
    MonitorElement* meStripTECWheel9zp_;

    MonitorElement* meStripTECWheel1zm_;
    MonitorElement* meStripTECWheel2zm_;
    MonitorElement* meStripTECWheel3zm_;
    MonitorElement* meStripTECWheel4zm_;
    MonitorElement* meStripTECWheel5zm_;
    MonitorElement* meStripTECWheel6zm_;
    MonitorElement* meStripTECWheel7zm_;
    MonitorElement* meStripTECWheel8zm_;
    MonitorElement* meStripTECWheel9zm_;

    MonitorElement* meNDigiTIBLayerzm_[4];
    MonitorElement* meNDigiTOBLayerzm_[6];
    MonitorElement* meNDigiTIDWheelzm_[3];
    MonitorElement* meNDigiTECWheelzm_[9];

    MonitorElement* meNDigiTIBLayerzp_[4];
    MonitorElement* meNDigiTOBLayerzp_[6];
    MonitorElement* meNDigiTIDWheelzp_[3];
    MonitorElement* meNDigiTECWheelzp_[9];


    //Back-End Interface
    DaqMonitorBEInterface* dbe_;
    string outputFile_;

};




#endif


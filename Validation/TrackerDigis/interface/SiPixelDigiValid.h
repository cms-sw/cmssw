#ifndef SiPixelDigiValid_h
#define SiPixelDigiValid_h

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
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>

using namespace std;
using namespace edm;


class  SiPixelDigiValid: public EDAnalyzer {

 public:
    
    SiPixelDigiValid(const ParameterSet& ps);
    ~SiPixelDigiValid();

 protected:
     void analyze(const Event& e, const EventSetup& c);
     void beginJob(const EventSetup& c);
     void endJob(void);

 private:

  string outputFile_;
  //Blade Number
  MonitorElement* meNDigiBlade1Zp_;
  MonitorElement* meNDigiBlade2Zp_; 
  MonitorElement* meNDigiBlade1Zm_;
  MonitorElement* meNDigiBlade2Zm_;
  //ADC Count
  MonitorElement* meAdcDisk1Panel1Zp_;
  MonitorElement* meAdcDisk1Panel2Zp_;
  MonitorElement* meAdcDisk2Panel1Zp_;
  MonitorElement* meAdcDisk2Panel2Zp_;

  MonitorElement* meAdcDisk1Panel1Zm_;
  MonitorElement* meAdcDisk1Panel2Zm_;
  MonitorElement* meAdcDisk2Panel1Zm_;
  MonitorElement* meAdcDisk2Panel2Zm_;
  //Col Number
  MonitorElement* meColDisk1Panel1Zp_;
  MonitorElement* meColDisk1Panel2Zp_;
  MonitorElement* meColDisk2Panel1Zp_;
  MonitorElement* meColDisk2Panel2Zp_;

  MonitorElement* meColDisk1Panel1Zm_;
  MonitorElement* meColDisk1Panel2Zm_;
  MonitorElement* meColDisk2Panel1Zm_;
  MonitorElement* meColDisk2Panel2Zm_;
  
  // ROW Number
  MonitorElement* meRowDisk1Panel1Zp_;
  MonitorElement* meRowDisk1Panel2Zp_;
  MonitorElement* meRowDisk2Panel1Zp_;
  MonitorElement* meRowDisk2Panel2Zp_; 

  MonitorElement* meRowDisk1Panel1Zm_;
  MonitorElement* meRowDisk1Panel2Zm_;
  MonitorElement* meRowDisk2Panel1Zm_;
  MonitorElement* meRowDisk2Panel2Zm_;
  //////Barrel Pixel
  MonitorElement* meAdcLayer1Ladder1_;
  MonitorElement* meAdcLayer1Ladder2_;
  MonitorElement* meAdcLayer1Ladder3_;
  MonitorElement* meAdcLayer1Ladder4_;
  MonitorElement* meAdcLayer1Ladder5_;
  MonitorElement* meAdcLayer1Ladder6_;
  MonitorElement* meAdcLayer1Ladder7_;
  MonitorElement* meAdcLayer1Ladder8_;
 
  MonitorElement* meAdcLayer2Ladder1_;
  MonitorElement* meAdcLayer2Ladder2_;
  MonitorElement* meAdcLayer2Ladder3_;
  MonitorElement* meAdcLayer2Ladder4_;
  MonitorElement* meAdcLayer2Ladder5_;
  MonitorElement* meAdcLayer2Ladder6_;
  MonitorElement* meAdcLayer2Ladder7_;
  MonitorElement* meAdcLayer2Ladder8_;

  MonitorElement* meAdcLayer3Ladder1_;
  MonitorElement* meAdcLayer3Ladder2_;
  MonitorElement* meAdcLayer3Ladder3_;
  MonitorElement* meAdcLayer3Ladder4_;
  MonitorElement* meAdcLayer3Ladder5_;
  MonitorElement* meAdcLayer3Ladder6_;
  MonitorElement* meAdcLayer3Ladder7_;
  MonitorElement* meAdcLayer3Ladder8_;


 

  DaqMonitorBEInterface* dbe_;

};
#endif


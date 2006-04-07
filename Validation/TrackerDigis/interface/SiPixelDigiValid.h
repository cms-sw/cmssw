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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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

  //////Barrel Pixel
  /* 1st Layer */
  MonitorElement* meAdcLayer1Ladder1_;
  MonitorElement* meAdcLayer1Ladder2_;
  MonitorElement* meAdcLayer1Ladder3_;
  MonitorElement* meAdcLayer1Ladder4_;
  MonitorElement* meAdcLayer1Ladder5_;
  MonitorElement* meAdcLayer1Ladder6_;
  MonitorElement* meAdcLayer1Ladder7_;
  MonitorElement* meAdcLayer1Ladder8_;

  MonitorElement* meRowLayer1Ladder1_;
  MonitorElement* meRowLayer1Ladder2_;
  MonitorElement* meRowLayer1Ladder3_;
  MonitorElement* meRowLayer1Ladder4_;
  MonitorElement* meRowLayer1Ladder5_;
  MonitorElement* meRowLayer1Ladder6_;
  MonitorElement* meRowLayer1Ladder7_;
  MonitorElement* meRowLayer1Ladder8_;

  MonitorElement* meColLayer1Ladder1_;
  MonitorElement* meColLayer1Ladder2_;
  MonitorElement* meColLayer1Ladder3_;
  MonitorElement* meColLayer1Ladder4_;
  MonitorElement* meColLayer1Ladder5_;
  MonitorElement* meColLayer1Ladder6_;
  MonitorElement* meColLayer1Ladder7_;
  MonitorElement* meColLayer1Ladder8_;
  MonitorElement* meNdigiPerLadderL1_;

  /* 2nd Layer */
  MonitorElement* meAdcLayer2Ladder1_;
  MonitorElement* meAdcLayer2Ladder2_;
  MonitorElement* meAdcLayer2Ladder3_;
  MonitorElement* meAdcLayer2Ladder4_;
  MonitorElement* meAdcLayer2Ladder5_;
  MonitorElement* meAdcLayer2Ladder6_;
  MonitorElement* meAdcLayer2Ladder7_;
  MonitorElement* meAdcLayer2Ladder8_;

  MonitorElement* meRowLayer2Ladder1_;
  MonitorElement* meRowLayer2Ladder2_;
  MonitorElement* meRowLayer2Ladder3_;
  MonitorElement* meRowLayer2Ladder4_;
  MonitorElement* meRowLayer2Ladder5_;
  MonitorElement* meRowLayer2Ladder6_;
  MonitorElement* meRowLayer2Ladder7_;
  MonitorElement* meRowLayer2Ladder8_;

  MonitorElement* meColLayer2Ladder1_;
  MonitorElement* meColLayer2Ladder2_;
  MonitorElement* meColLayer2Ladder3_;
  MonitorElement* meColLayer2Ladder4_;
  MonitorElement* meColLayer2Ladder5_;
  MonitorElement* meColLayer2Ladder6_;
  MonitorElement* meColLayer2Ladder7_;
  MonitorElement* meColLayer2Ladder8_;
  MonitorElement* meNdigiPerLadderL2_;

  /* 3rd Layer */

  MonitorElement* meAdcLayer3Ladder1_;
  MonitorElement* meAdcLayer3Ladder2_;
  MonitorElement* meAdcLayer3Ladder3_;
  MonitorElement* meAdcLayer3Ladder4_;
  MonitorElement* meAdcLayer3Ladder5_;
  MonitorElement* meAdcLayer3Ladder6_;
  MonitorElement* meAdcLayer3Ladder7_;
  MonitorElement* meAdcLayer3Ladder8_;

  MonitorElement* meRowLayer3Ladder1_;
  MonitorElement* meRowLayer3Ladder2_;
  MonitorElement* meRowLayer3Ladder3_;
  MonitorElement* meRowLayer3Ladder4_;
  MonitorElement* meRowLayer3Ladder5_;
  MonitorElement* meRowLayer3Ladder6_;
  MonitorElement* meRowLayer3Ladder7_;
  MonitorElement* meRowLayer3Ladder8_;

  MonitorElement* meColLayer3Ladder1_;
  MonitorElement* meColLayer3Ladder2_;
  MonitorElement* meColLayer3Ladder3_;
  MonitorElement* meColLayer3Ladder4_;
  MonitorElement* meColLayer3Ladder5_;
  MonitorElement* meColLayer3Ladder6_;
  MonitorElement* meColLayer3Ladder7_;
  MonitorElement* meColLayer3Ladder8_;
  MonitorElement* meNdigiPerLadderL3_;

///Forwar Pixel
  /* 1st Disk in ZPlus Side */
  MonitorElement*  meAdcZpDisk1Panel1Plaq1_;
  MonitorElement*  meAdcZpDisk1Panel1Plaq2_;
  MonitorElement*  meAdcZpDisk1Panel1Plaq3_;
  MonitorElement*  meAdcZpDisk1Panel1Plaq4_;
  MonitorElement*  meAdcZpDisk1Panel2Plaq1_;
  MonitorElement*  meAdcZpDisk1Panel2Plaq2_;
  MonitorElement*  meAdcZpDisk1Panel2Plaq3_;

  MonitorElement*  meRowZpDisk1Panel1Plaq1_;
  MonitorElement*  meRowZpDisk1Panel1Plaq2_;
  MonitorElement*  meRowZpDisk1Panel1Plaq3_;
  MonitorElement*  meRowZpDisk1Panel1Plaq4_;
  MonitorElement*  meRowZpDisk1Panel2Plaq1_;
  MonitorElement*  meRowZpDisk1Panel2Plaq2_;
  MonitorElement*  meRowZpDisk1Panel2Plaq3_;

  MonitorElement*  meColZpDisk1Panel1Plaq1_;
  MonitorElement*  meColZpDisk1Panel1Plaq2_;
  MonitorElement*  meColZpDisk1Panel1Plaq3_;
  MonitorElement*  meColZpDisk1Panel1Plaq4_;
  MonitorElement*  meColZpDisk1Panel2Plaq1_;
  MonitorElement*  meColZpDisk1Panel2Plaq2_;
  MonitorElement*  meColZpDisk1Panel2Plaq3_;
  MonitorElement*  meNdigiZpDisk1PerPanel1_;
  MonitorElement*  meNdigiZpDisk1PerPanel2_;
  

  /* 2nd Disk in ZPlus Side */
  MonitorElement*  meAdcZpDisk2Panel1Plaq1_;
  MonitorElement*  meAdcZpDisk2Panel1Plaq2_;
  MonitorElement*  meAdcZpDisk2Panel1Plaq3_;
  MonitorElement*  meAdcZpDisk2Panel1Plaq4_;
  MonitorElement*  meAdcZpDisk2Panel2Plaq1_;
  MonitorElement*  meAdcZpDisk2Panel2Plaq2_;
  MonitorElement*  meAdcZpDisk2Panel2Plaq3_;

  MonitorElement*  meRowZpDisk2Panel1Plaq1_;
  MonitorElement*  meRowZpDisk2Panel1Plaq2_;
  MonitorElement*  meRowZpDisk2Panel1Plaq3_;
  MonitorElement*  meRowZpDisk2Panel1Plaq4_;
  MonitorElement*  meRowZpDisk2Panel2Plaq1_;
  MonitorElement*  meRowZpDisk2Panel2Plaq2_;
  MonitorElement*  meRowZpDisk2Panel2Plaq3_;

  MonitorElement*  meColZpDisk2Panel1Plaq1_;
  MonitorElement*  meColZpDisk2Panel1Plaq2_;
  MonitorElement*  meColZpDisk2Panel1Plaq3_;
  MonitorElement*  meColZpDisk2Panel1Plaq4_;
  MonitorElement*  meColZpDisk2Panel2Plaq1_;
  MonitorElement*  meColZpDisk2Panel2Plaq2_;
  MonitorElement*  meColZpDisk2Panel2Plaq3_;
  MonitorElement*  meNdigiZpDisk2PerPanel1_;
  MonitorElement*  meNdigiZpDisk2PerPanel2_;

  /* 1st Disk in ZMinus Side */
  MonitorElement*  meAdcZmDisk1Panel1Plaq1_;
  MonitorElement*  meAdcZmDisk1Panel1Plaq2_;
  MonitorElement*  meAdcZmDisk1Panel1Plaq3_;
  MonitorElement*  meAdcZmDisk1Panel1Plaq4_;
  MonitorElement*  meAdcZmDisk1Panel2Plaq1_;
  MonitorElement*  meAdcZmDisk1Panel2Plaq2_;
  MonitorElement*  meAdcZmDisk1Panel2Plaq3_;

  MonitorElement*  meRowZmDisk1Panel1Plaq1_;
  MonitorElement*  meRowZmDisk1Panel1Plaq2_;
  MonitorElement*  meRowZmDisk1Panel1Plaq3_;
  MonitorElement*  meRowZmDisk1Panel1Plaq4_;
  MonitorElement*  meRowZmDisk1Panel2Plaq1_;
  MonitorElement*  meRowZmDisk1Panel2Plaq2_;
  MonitorElement*  meRowZmDisk1Panel2Plaq3_;

  MonitorElement*  meColZmDisk1Panel1Plaq1_;
  MonitorElement*  meColZmDisk1Panel1Plaq2_;
  MonitorElement*  meColZmDisk1Panel1Plaq3_;
  MonitorElement*  meColZmDisk1Panel1Plaq4_;
  MonitorElement*  meColZmDisk1Panel2Plaq1_;
  MonitorElement*  meColZmDisk1Panel2Plaq2_;
  MonitorElement*  meColZmDisk1Panel2Plaq3_;
  MonitorElement*  meNdigiZmDisk1PerPanel1_;
  MonitorElement*  meNdigiZmDisk1PerPanel2_;

  /* 2nd Disk in ZMius Side */
  MonitorElement*  meAdcZmDisk2Panel1Plaq1_;
  MonitorElement*  meAdcZmDisk2Panel1Plaq2_;
  MonitorElement*  meAdcZmDisk2Panel1Plaq3_;
  MonitorElement*  meAdcZmDisk2Panel1Plaq4_;
  MonitorElement*  meAdcZmDisk2Panel2Plaq1_;
  MonitorElement*  meAdcZmDisk2Panel2Plaq2_;
  MonitorElement*  meAdcZmDisk2Panel2Plaq3_;

  MonitorElement*  meRowZmDisk2Panel1Plaq1_;
  MonitorElement*  meRowZmDisk2Panel1Plaq2_;
  MonitorElement*  meRowZmDisk2Panel1Plaq3_;
  MonitorElement*  meRowZmDisk2Panel1Plaq4_;
  MonitorElement*  meRowZmDisk2Panel2Plaq1_;
  MonitorElement*  meRowZmDisk2Panel2Plaq2_;
  MonitorElement*  meRowZmDisk2Panel2Plaq3_;

  MonitorElement*  meColZmDisk2Panel1Plaq1_;
  MonitorElement*  meColZmDisk2Panel1Plaq2_;
  MonitorElement*  meColZmDisk2Panel1Plaq3_;
  MonitorElement*  meColZmDisk2Panel1Plaq4_;
  MonitorElement*  meColZmDisk2Panel2Plaq1_;
  MonitorElement*  meColZmDisk2Panel2Plaq2_;
  MonitorElement*  meColZmDisk2Panel2Plaq3_;
  MonitorElement*  meNdigiZmDisk2PerPanel1_;
  MonitorElement*  meNdigiZmDisk2PerPanel2_;
   
 
  DaqMonitorBEInterface* dbe_;

};
#endif


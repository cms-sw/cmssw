#ifndef SiPixelDigiValid_h
#define SiPixelDigiValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <string>

namespace edm {
  template< class T > class DetSetVector;
}
class PixelDigi;
class DQMStore;
class MonitorElement;

class  SiPixelDigiValid: public DQMEDAnalyzer {

 public:
    
    SiPixelDigiValid(const edm::ParameterSet& ps);
    ~SiPixelDigiValid();

 protected:
     void analyze(const edm::Event& e, const edm::EventSetup& c);
     void beginJob();
     void bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es);
     void endJob(void);

 private:

  std::string outputFile_;
  bool runStandalone;

  //////Barrel Pixel
  /* 1st Layer */
  MonitorElement* meAdcLayer1Ring1_;
  MonitorElement* meAdcLayer1Ring2_;
  MonitorElement* meAdcLayer1Ring3_;
  MonitorElement* meAdcLayer1Ring4_;
  MonitorElement* meAdcLayer1Ring5_;
  MonitorElement* meAdcLayer1Ring6_;
  MonitorElement* meAdcLayer1Ring7_;
  MonitorElement* meAdcLayer1Ring8_;

  MonitorElement* meRowLayer1Ring1_;
  MonitorElement* meRowLayer1Ring2_;
  MonitorElement* meRowLayer1Ring3_;
  MonitorElement* meRowLayer1Ring4_;
  MonitorElement* meRowLayer1Ring5_;
  MonitorElement* meRowLayer1Ring6_;
  MonitorElement* meRowLayer1Ring7_;
  MonitorElement* meRowLayer1Ring8_;

  MonitorElement* meColLayer1Ring1_;
  MonitorElement* meColLayer1Ring2_;
  MonitorElement* meColLayer1Ring3_;
  MonitorElement* meColLayer1Ring4_;
  MonitorElement* meColLayer1Ring5_;
  MonitorElement* meColLayer1Ring6_;
  MonitorElement* meColLayer1Ring7_;
  MonitorElement* meColLayer1Ring8_;

  MonitorElement* meDigiMultiLayer1Ring1_;
  MonitorElement* meDigiMultiLayer1Ring2_;
  MonitorElement* meDigiMultiLayer1Ring3_;
  MonitorElement* meDigiMultiLayer1Ring4_;
  MonitorElement* meDigiMultiLayer1Ring5_;
  MonitorElement* meDigiMultiLayer1Ring6_;
  MonitorElement* meDigiMultiLayer1Ring7_;
  MonitorElement* meDigiMultiLayer1Ring8_;


  /* 2nd Layer */
  MonitorElement* meAdcLayer2Ring1_;
  MonitorElement* meAdcLayer2Ring2_;
  MonitorElement* meAdcLayer2Ring3_;
  MonitorElement* meAdcLayer2Ring4_;
  MonitorElement* meAdcLayer2Ring5_;
  MonitorElement* meAdcLayer2Ring6_;
  MonitorElement* meAdcLayer2Ring7_;
  MonitorElement* meAdcLayer2Ring8_;

  MonitorElement* meRowLayer2Ring1_;
  MonitorElement* meRowLayer2Ring2_;
  MonitorElement* meRowLayer2Ring3_;
  MonitorElement* meRowLayer2Ring4_;
  MonitorElement* meRowLayer2Ring5_;
  MonitorElement* meRowLayer2Ring6_;
  MonitorElement* meRowLayer2Ring7_;
  MonitorElement* meRowLayer2Ring8_;

  MonitorElement* meColLayer2Ring1_;
  MonitorElement* meColLayer2Ring2_;
  MonitorElement* meColLayer2Ring3_;
  MonitorElement* meColLayer2Ring4_;
  MonitorElement* meColLayer2Ring5_;
  MonitorElement* meColLayer2Ring6_;
  MonitorElement* meColLayer2Ring7_;
  MonitorElement* meColLayer2Ring8_;

  MonitorElement* meDigiMultiLayer2Ring1_;
  MonitorElement* meDigiMultiLayer2Ring2_;
  MonitorElement* meDigiMultiLayer2Ring3_;
  MonitorElement* meDigiMultiLayer2Ring4_;
  MonitorElement* meDigiMultiLayer2Ring5_;
  MonitorElement* meDigiMultiLayer2Ring6_;
  MonitorElement* meDigiMultiLayer2Ring7_;
  MonitorElement* meDigiMultiLayer2Ring8_;


  /* 3rd Layer */

  MonitorElement* meAdcLayer3Ring1_;
  MonitorElement* meAdcLayer3Ring2_;
  MonitorElement* meAdcLayer3Ring3_;
  MonitorElement* meAdcLayer3Ring4_;
  MonitorElement* meAdcLayer3Ring5_;
  MonitorElement* meAdcLayer3Ring6_;
  MonitorElement* meAdcLayer3Ring7_;
  MonitorElement* meAdcLayer3Ring8_;

  MonitorElement* meRowLayer3Ring1_;
  MonitorElement* meRowLayer3Ring2_;
  MonitorElement* meRowLayer3Ring3_;
  MonitorElement* meRowLayer3Ring4_;
  MonitorElement* meRowLayer3Ring5_;
  MonitorElement* meRowLayer3Ring6_;
  MonitorElement* meRowLayer3Ring7_;
  MonitorElement* meRowLayer3Ring8_;

  MonitorElement* meColLayer3Ring1_;
  MonitorElement* meColLayer3Ring2_;
  MonitorElement* meColLayer3Ring3_;
  MonitorElement* meColLayer3Ring4_;
  MonitorElement* meColLayer3Ring5_;
  MonitorElement* meColLayer3Ring6_;
  MonitorElement* meColLayer3Ring7_;
  MonitorElement* meColLayer3Ring8_;

  MonitorElement* meDigiMultiLayer3Ring1_;
  MonitorElement* meDigiMultiLayer3Ring2_;
  MonitorElement* meDigiMultiLayer3Ring3_;
  MonitorElement* meDigiMultiLayer3Ring4_;
  MonitorElement* meDigiMultiLayer3Ring5_;
  MonitorElement* meDigiMultiLayer3Ring6_;
  MonitorElement* meDigiMultiLayer3Ring7_;
  MonitorElement* meDigiMultiLayer3Ring8_;
 
  /*Number of digi versus ladder number */
  MonitorElement* meDigiMultiLayer1Ladders_;
  MonitorElement* meDigiMultiLayer2Ladders_;
  MonitorElement* meDigiMultiLayer3Ladders_;


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
   
 
  DQMStore* dbe_;
  edm::EDGetTokenT< edm::DetSetVector<PixelDigi> > edmDetSetVector_PixelDigi_Token_;

 
};
#endif


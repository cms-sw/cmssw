#ifndef SiStripDigiValid_h
#define SiStripDigiValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <string>

namespace edm {
  template<class T> class DetSetVector;
}
class SiStripDigi;
class DQMStore;
class MonitorElement;

class  SiStripDigiValid: public DQMEDAnalyzer {

 public:

    SiStripDigiValid(const edm::ParameterSet& ps);
    ~SiStripDigiValid();

 protected:
     void analyze(const edm::Event& e, const edm::EventSetup& c);
     void beginJob();
     void bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es);
     void endJob(void);

 private:

 
    //TIB  ADC
    MonitorElement* meAdcTIBLayer1Extzp_[3];
    MonitorElement* meAdcTIBLayer2Extzp_[3];
    MonitorElement* meAdcTIBLayer3Extzp_[3];
    MonitorElement* meAdcTIBLayer4Extzp_[3];

    MonitorElement* meStripTIBLayer1Extzp_[3];
    MonitorElement* meStripTIBLayer2Extzp_[3];
    MonitorElement* meStripTIBLayer3Extzp_[3];
    MonitorElement* meStripTIBLayer4Extzp_[3];

    MonitorElement* meAdcTIBLayer1Intzp_[3];
    MonitorElement* meAdcTIBLayer2Intzp_[3];
    MonitorElement* meAdcTIBLayer3Intzp_[3];
    MonitorElement* meAdcTIBLayer4Intzp_[3];

    MonitorElement* meStripTIBLayer1Intzp_[3];
    MonitorElement* meStripTIBLayer2Intzp_[3];
    MonitorElement* meStripTIBLayer3Intzp_[3];
    MonitorElement* meStripTIBLayer4Intzp_[3];

    MonitorElement* meAdcTIBLayer1Extzm_[3];
    MonitorElement* meAdcTIBLayer2Extzm_[3];
    MonitorElement* meAdcTIBLayer3Extzm_[3];
    MonitorElement* meAdcTIBLayer4Extzm_[3];

    MonitorElement* meStripTIBLayer1Extzm_[3];
    MonitorElement* meStripTIBLayer2Extzm_[3];
    MonitorElement* meStripTIBLayer3Extzm_[3];
    MonitorElement* meStripTIBLayer4Extzm_[3];

    MonitorElement* meAdcTIBLayer1Intzm_[3];
    MonitorElement* meAdcTIBLayer2Intzm_[3];
    MonitorElement* meAdcTIBLayer3Intzm_[3];
    MonitorElement* meAdcTIBLayer4Intzm_[3];

    MonitorElement* meStripTIBLayer1Intzm_[3];
    MonitorElement* meStripTIBLayer2Intzm_[3];
    MonitorElement* meStripTIBLayer3Intzm_[3];
    MonitorElement* meStripTIBLayer4Intzm_[3];

    //TOB ADC
    MonitorElement* meAdcTOBLayer1zp_[6];
    MonitorElement* meAdcTOBLayer2zp_[6];
    MonitorElement* meAdcTOBLayer3zp_[6];
    MonitorElement* meAdcTOBLayer4zp_[6];
    MonitorElement* meAdcTOBLayer5zp_[6];
    MonitorElement* meAdcTOBLayer6zp_[6]; 

    MonitorElement* meAdcTOBLayer1zm_[6];
    MonitorElement* meAdcTOBLayer2zm_[6];
    MonitorElement* meAdcTOBLayer3zm_[6];
    MonitorElement* meAdcTOBLayer4zm_[6];
    MonitorElement* meAdcTOBLayer5zm_[6];
    MonitorElement* meAdcTOBLayer6zm_[6];

    //TOB Strip
    MonitorElement* meStripTOBLayer1zp_[6];
    MonitorElement* meStripTOBLayer2zp_[6];
    MonitorElement* meStripTOBLayer3zp_[6];
    MonitorElement* meStripTOBLayer4zp_[6];
    MonitorElement* meStripTOBLayer5zp_[6];
    MonitorElement* meStripTOBLayer6zp_[6];

    MonitorElement* meStripTOBLayer1zm_[6];
    MonitorElement* meStripTOBLayer2zm_[6];
    MonitorElement* meStripTOBLayer3zm_[6];
    MonitorElement* meStripTOBLayer4zm_[6];
    MonitorElement* meStripTOBLayer5zm_[6];
    MonitorElement* meStripTOBLayer6zm_[6];


    //TID  ADC
    MonitorElement* meAdcTIDWheel1zp_[3];
    MonitorElement* meAdcTIDWheel2zp_[3];
    MonitorElement* meAdcTIDWheel3zp_[3];

    MonitorElement* meAdcTIDWheel1zm_[3];
    MonitorElement* meAdcTIDWheel2zm_[3];
    MonitorElement* meAdcTIDWheel3zm_[3];

    //TID Strip
    MonitorElement* meStripTIDWheel1zp_[3];
    MonitorElement* meStripTIDWheel2zp_[3];
    MonitorElement* meStripTIDWheel3zp_[3];

    MonitorElement* meStripTIDWheel1zm_[3];
    MonitorElement* meStripTIDWheel2zm_[3];
    MonitorElement* meStripTIDWheel3zm_[3];

    //TEC ADC
    MonitorElement* meAdcTECWheel1zp_[7];
    MonitorElement* meAdcTECWheel2zp_[7];
    MonitorElement* meAdcTECWheel3zp_[7];
    MonitorElement* meAdcTECWheel4zp_[6];
    MonitorElement* meAdcTECWheel5zp_[6];
    MonitorElement* meAdcTECWheel6zp_[6];
    MonitorElement* meAdcTECWheel7zp_[5];
    MonitorElement* meAdcTECWheel8zp_[5];
    MonitorElement* meAdcTECWheel9zp_[4];

    MonitorElement* meAdcTECWheel1zm_[7];
    MonitorElement* meAdcTECWheel2zm_[7];
    MonitorElement* meAdcTECWheel3zm_[7];
    MonitorElement* meAdcTECWheel4zm_[6];
    MonitorElement* meAdcTECWheel5zm_[6];
    MonitorElement* meAdcTECWheel6zm_[6];
    MonitorElement* meAdcTECWheel7zm_[5];
    MonitorElement* meAdcTECWheel8zm_[5];
    MonitorElement* meAdcTECWheel9zm_[4];

    //TEC Strip
    MonitorElement* meStripTECWheel1zp_[7];
    MonitorElement* meStripTECWheel2zp_[7];
    MonitorElement* meStripTECWheel3zp_[7];
    MonitorElement* meStripTECWheel4zp_[6];
    MonitorElement* meStripTECWheel5zp_[6];
    MonitorElement* meStripTECWheel6zp_[6];
    MonitorElement* meStripTECWheel7zp_[5];
    MonitorElement* meStripTECWheel8zp_[5];
    MonitorElement* meStripTECWheel9zp_[4];

    MonitorElement* meStripTECWheel1zm_[7];
    MonitorElement* meStripTECWheel2zm_[7];
    MonitorElement* meStripTECWheel3zm_[7];
    MonitorElement* meStripTECWheel4zm_[6];
    MonitorElement* meStripTECWheel5zm_[6];
    MonitorElement* meStripTECWheel6zm_[6];
    MonitorElement* meStripTECWheel7zm_[5];
    MonitorElement* meStripTECWheel8zm_[5];
    MonitorElement* meStripTECWheel9zm_[4];

    MonitorElement* meNDigiTIBLayerzm_[4];
    MonitorElement* meNDigiTOBLayerzm_[6];
    MonitorElement* meNDigiTIDWheelzm_[3];
    MonitorElement* meNDigiTECWheelzm_[9];

    MonitorElement* meNDigiTIBLayerzp_[4];
    MonitorElement* meNDigiTOBLayerzp_[6];
    MonitorElement* meNDigiTIDWheelzp_[3];
    MonitorElement* meNDigiTECWheelzp_[9];


    //Back-End Interface
    DQMStore* dbe_;
    bool runStandalone;
    std::string outputFile_;
    edm::EDGetTokenT< edm::DetSetVector<SiStripDigi> > edmDetSetVector_SiStripDigi_Token_;
};




#endif


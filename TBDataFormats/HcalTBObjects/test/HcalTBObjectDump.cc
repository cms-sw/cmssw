#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include <iostream>

using namespace std;

namespace cms {

  /** \class HcalTBObjectDump
      
  $Date: 2012/09/20 20:16:10 $
  $Revision: 1.9 $
  \author J. Mans - Minnesota
  */
  class HcalTBObjectDump : public edm::EDAnalyzer {
  public:
    explicit HcalTBObjectDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  private:
    edm::InputTag hcalTBTriggerDataTag_;
    edm::InputTag hcalTBRunDataTag_;
    edm::InputTag hcalTBEventPositionTag_;
    edm::InputTag hcalTBTimingTag_;
  };


  HcalTBObjectDump::HcalTBObjectDump(edm::ParameterSet const& conf) :
    hcalTBTriggerDataTag_(conf.getParameter<edm::InputTag>("hcalTBTriggerDataTag")),
    hcalTBRunDataTag_(conf.getParameter<edm::InputTag>("hcalTBRunDataTag")),
    hcalTBEventPositionTag_(conf.getParameter<edm::InputTag>("hcalTBEventPositionTag")),
    hcalTBTimingTag_(conf.getParameter<edm::InputTag>("hcalTBTimingTag"))
 {
  }
  
  void HcalTBObjectDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    
    try {
      edm::Handle<HcalTBTriggerData> td;

      e.getByLabel(hcalTBTriggerDataTag_, td);
      const HcalTBTriggerData& info=*(td);

      cout << "TRIGGER DATA: ";
      cout << info;    
    } catch (...) {
      cout << "No HcalTBTriggerData." << endl;
    }

    try {
      edm::Handle<HcalTBRunData> td;

      e.getByLabel(hcalTBRunDataTag_, td);
      const HcalTBRunData& info=*(td);

      cout << "RUN DATA: ";
      cout << info;      
    } catch (...) {
      cout << "No HcalTBRunData." << endl;
    }

    try {
      edm::Handle<HcalTBEventPosition> td;

      e.getByLabel(hcalTBEventPositionTag_, td);
      const HcalTBEventPosition& info=*td;

      cout << "Event position info: ";
      cout << info;      
    } catch (...) {
      cout << "No HcalTBEventPosition." << endl;
    }

    try {
      
      edm::Handle<HcalTBTiming>td;

      e.getByLabel(hcalTBTimingTag_, td);
      const HcalTBTiming& info=*(td);

      cout << "Timing: ";
      cout << info;
      
    } catch (...) {
      cout << "No HcalTBTiming." << endl;
    }

    cout << endl;    
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;


DEFINE_FWK_MODULE(HcalTBObjectDump);


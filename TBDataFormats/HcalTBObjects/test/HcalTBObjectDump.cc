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
      
  $Date: 2012/07/20 20:59:13 $
  $Revision: 1.8 $
  \author J. Mans - Minnesota
  */
  class HcalTBObjectDump : public edm::EDAnalyzer {
  public:
    explicit HcalTBObjectDump(edm::ParameterSet const& conf);
    void analyze(edm::Event const& e, edm::EventSetup const& c) override;

  private:
    edm::EDGetTokenT<HcalTBTriggerData> tok_tb_;
    edm::EDGetTokenT<HcalTBRunData> tok_run_;
    edm::EDGetTokenT<HcalTBEventPosition> tok_pos_;
    edm::EDGetTokenT<HcalTBTiming> tok_timing_;
  };

  HcalTBObjectDump::HcalTBObjectDump(edm::ParameterSet const& conf) {
    tok_tb_ = consumes<HcalTBTriggerData>(conf.getParameter<edm::InputTag>("hcalTBTriggerDataTag"));
    tok_run_ = consumes<HcalTBRunData>(conf.getParameter<edm::InputTag>("hcalTBRunDataTag"));
    tok_pos_ = consumes<HcalTBEventPosition>(conf.getParameter<edm::InputTag>("hcalTBEventPositionTag"));
    tok_timing_ = consumes<HcalTBTiming>(conf.getParameter<edm::InputTag>("hcalTBTimingTag"));
  }

  void HcalTBObjectDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    try {
      edm::Handle<HcalTBTriggerData> td;

      e.getByToken(tok_tb_, td);
      const HcalTBTriggerData& info = *(td);

      cout << "TRIGGER DATA: ";
      cout << info;
    } catch (...) {
      cout << "No HcalTBTriggerData." << endl;
    }

    try {
      edm::Handle<HcalTBRunData> td;

      e.getByToken(tok_run_, td);
      const HcalTBRunData& info = *(td);

      cout << "RUN DATA: ";
      cout << info;
    } catch (...) {
      cout << "No HcalTBRunData." << endl;
    }

    try {
      edm::Handle<HcalTBEventPosition> td;

      e.getByToken(tok_pos_, td);
      const HcalTBEventPosition& info = *td;

      cout << "Event position info: ";
      cout << info;
    } catch (...) {
      cout << "No HcalTBEventPosition." << endl;
    }

    try {
      edm::Handle<HcalTBTiming> td;

      e.getByToken(tok_timing_, td);
      const HcalTBTiming& info = *(td);

      cout << "Timing: ";
      cout << info;

    } catch (...) {
      cout << "No HcalTBTiming." << endl;
    }

    cout << endl;
  }
}  // namespace cms

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;

DEFINE_FWK_MODULE(HcalTBObjectDump);

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>

using namespace std;

namespace cms {

  /** \class HcalTBObjectDump
      
  $Date: 2005/08/23 01:07:18 $
  $Revision: 1.1 $
  \author J. Mans - Minnesota
  */
  class HcalTBObjectDump : public edm::EDAnalyzer {
  public:
    explicit HcalTBObjectDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  };



 class HTBOHappySelector : public edm::Selector {
  public:
    HTBOHappySelector() { }
  private:
    virtual bool doMatch(const edm::Provenance& p) const {
      //      cout << p << endl;
      return true;
    }
 };


  HcalTBObjectDump::HcalTBObjectDump(edm::ParameterSet const& conf) {
  }
  
  void HcalTBObjectDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    HTBOHappySelector s;
    
    try {
      std::vector<edm::Handle<HcalTBTriggerData> > td;

      e.getMany(s,td);
      std::vector<edm::Handle<HcalTBTriggerData> >::iterator i;
      for (i=td.begin(); i!=td.end(); i++) {
	const HcalTBTriggerData& info=*(*i);

	cout << "TRIGGER DATA: ";
	cout << info;
      }
    } catch (...) {
      cout << "No HcalTBTriggerData." << endl;
    }

    try {
      std::vector<edm::Handle<HcalTBRunData> > td;

      e.getMany(s,td);
      std::vector<edm::Handle<HcalTBRunData> >::iterator i;
      for (i=td.begin(); i!=td.end(); i++) {
	const HcalTBRunData& info=*(*i);

	cout << "RUN DATA: ";
	cout << info;
      }
    } catch (...) {
      cout << "No HcalTBRunData." << endl;
    }

    try {
      std::vector<edm::Handle<HcalTBEventPosition> > td;

      e.getMany(s,td);
      std::vector<edm::Handle<HcalTBEventPosition> >::iterator i;
      for (i=td.begin(); i!=td.end(); i++) {
	const HcalTBEventPosition& info=*(*i);

	cout << "Event position info: ";
	cout << info;
      }
    } catch (...) {
      cout << "No HcalTBEventPosition." << endl;
    }

    try {
      std::vector<edm::Handle<HcalTBTiming> > td;

      e.getMany(s,td);
      std::vector<edm::Handle<HcalTBTiming> >::iterator i;
      for (i=td.begin(); i!=td.end(); i++) {
	const HcalTBTiming& info=*(*i);

	cout << "Timing: ";
	cout << info;
      }
    } catch (...) {
      cout << "No HcalTBTiming." << endl;
    }

    cout << endl;    
  }
}

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalTBObjectDump)


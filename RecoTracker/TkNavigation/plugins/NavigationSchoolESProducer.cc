#include <FWCore/Utilities/interface/ESInputTag.h>

#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include "NavigationSchoolFactory.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

//
// class decleration
//

class dso_hidden NavigationSchoolESProducer final : public edm::ESProducer {
public:
  NavigationSchoolESProducer(const edm::ParameterSet&);

  typedef std::unique_ptr<NavigationSchool> ReturnType;

  virtual ReturnType produce(const NavigationSchoolRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> geometricSearchTrackerToken_;
  const std::string navigationSchoolName_;
  std::string navigationSchoolPluginName_;
};

//
//
// constructors and destructor
//
NavigationSchoolESProducer::NavigationSchoolESProducer(const edm::ParameterSet& iConfig)
    : navigationSchoolName_(iConfig.getParameter<std::string>("ComponentName")),
      navigationSchoolPluginName_(iConfig.getParameter<std::string>("PluginName")) {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this, navigationSchoolName_);
  magFieldToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("SimpleMagneticField")));
  geometricSearchTrackerToken_ = cc.consumes();

  //now do what ever other initialization is needed
  if (navigationSchoolPluginName_.empty())
    navigationSchoolPluginName_ = navigationSchoolName_;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
NavigationSchoolESProducer::ReturnType NavigationSchoolESProducer::produce(const NavigationSchoolRecord& iRecord) {
  using namespace edm::es;

  //get the geometricsearch tracker geometry
  return ReturnType(NavigationSchoolFactory::get()->create(
      navigationSchoolPluginName_, &iRecord.get(geometricSearchTrackerToken_), &iRecord.get(magFieldToken_)));
}

void NavigationSchoolESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName");
  desc.add<std::string>("PluginName", "");
  desc.add<std::string>("SimpleMagneticField", "");
  descriptions.addDefault(desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(NavigationSchoolESProducer);

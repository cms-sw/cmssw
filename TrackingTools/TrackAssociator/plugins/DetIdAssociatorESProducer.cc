// -*- C++ -*-
//
// Package:    DetIdAssociatorESProducer
// Class:      DetIdAssociatorESProducer
//
/**\class DetIdAssociatorESProducer TrackingTools/TrackAssociator/plugins/DetIdAssociatorESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Thu Oct  4 02:28:48 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "DetIdAssociatorFactory.h"
#include "DetIdAssociatorMaker.h"

#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"

//
// class decleration
//

class DetIdAssociatorESProducer : public edm::ESProducer {
public:
  DetIdAssociatorESProducer(const edm::ParameterSet&);
  ~DetIdAssociatorESProducer() override;

  typedef std::unique_ptr<DetIdAssociator> ReturnType;

  ReturnType produce(const DetIdAssociatorRecord&);

private:
  const std::string cName;
  std::unique_ptr<const DetIdAssociatorMaker> maker_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DetIdAssociatorESProducer::DetIdAssociatorESProducer(const edm::ParameterSet& iConfig)
    : cName{iConfig.getParameter<std::string>("ComponentName")},
      maker_{DetIdAssociatorFactory::get()->create(cName, iConfig, setWhatProduced(this, cName))} {}

DetIdAssociatorESProducer::~DetIdAssociatorESProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
DetIdAssociatorESProducer::ReturnType DetIdAssociatorESProducer::produce(const DetIdAssociatorRecord& iRecord) {
  using namespace edm::es;
  LogTrace("TrackAssociator") << "Making DetIdAssociatorRecord with label: " << cName;
  ReturnType dia = maker_->make(iRecord);
  dia->buildMap();
  LogTrace("TrackAssociator") << "Map id built for DetIdAssociatorRecord with label: " << cName;
  return dia;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DetIdAssociatorESProducer);

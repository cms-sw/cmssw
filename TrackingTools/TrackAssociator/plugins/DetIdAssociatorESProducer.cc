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

#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"


//
// class decleration
//

class DetIdAssociatorESProducer : public edm::ESProducer {
public:
  DetIdAssociatorESProducer(const edm::ParameterSet&);
  ~DetIdAssociatorESProducer();
  
  typedef std::shared_ptr<DetIdAssociator> ReturnType;
  
  ReturnType produce(const DetIdAssociatorRecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string cName;
  edm::ParameterSet pSet;
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
{
  cName =iConfig.getParameter<std::string>("ComponentName");
  pSet = iConfig;
  setWhatProduced(this, cName);
}


DetIdAssociatorESProducer::~DetIdAssociatorESProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
DetIdAssociatorESProducer::ReturnType
DetIdAssociatorESProducer::produce(const DetIdAssociatorRecord& iRecord)
{
   using namespace edm::es;
   LogTrace("TrackAssociator") << "Making DetIdAssociatorRecord with label: " << cName;
   ReturnType dia(DetIdAssociatorFactory::get()->create(cName, pSet));
   dia->setGeometry(iRecord);
   dia->setConditions(iRecord);
   dia->buildMap();
   LogTrace("TrackAssociator") << "Map id built for DetIdAssociatorRecord with label: " << cName;
   return dia;
}


/// ParameterSet descriptions
void DetIdAssociatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("doME0",false);
  desc.setAllowAnything();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DetIdAssociatorESProducer);

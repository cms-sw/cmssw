#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <TClass.h>
#include <TBufferXML.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

class BeamSpotFakeConditionsWriter : public edm::EDAnalyzer {
    public:
	explicit BeamSpotFakeConditionsWriter(const edm::ParameterSet &params);
	virtual void analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup);
	virtual void endJob();
    private:
	std::string xmlCalibration;
};

BeamSpotFakeConditionsWriter::BeamSpotFakeConditionsWriter(const edm::ParameterSet &params) :
	xmlCalibration(params.getParameter<std::string>("xmlCalibration"))
{
}

void BeamSpotFakeConditionsWriter::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup){
  edm::ESHandle<BeamSpotObjects> theBeamSpotCalibrationHandle;
  iSetup.get<BeamSpotObjectsRcd>().get(theBeamSpotCalibrationHandle);
  
  std::ofstream of(xmlCalibration.c_str());
  if (!of.good()) throw cms::Exception("BeamSpotFakeConditionsWriter")<< "File \"" << xmlCalibration<< "\" could not be opened for writing." << std::endl;  
  of << TBufferXML(TBuffer::kWrite).ConvertToXML(const_cast<void*>(static_cast<const void*>(theBeamSpotCalibrationHandle.product())),
						 TClass::GetClass("BeamSpotObjects"),
						 kTRUE, kFALSE);
  
  of.close();
}

void BeamSpotFakeConditionsWriter::endJob()
{
}

DEFINE_ANOTHER_FWK_MODULE(BeamSpotFakeConditionsWriter);

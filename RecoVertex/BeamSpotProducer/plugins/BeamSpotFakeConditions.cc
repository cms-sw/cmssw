//

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <TClass.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

class BeamSpotFakeConditions : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
	typedef boost::shared_ptr<BeamSpotObjects> ReturnType;
	BeamSpotFakeConditions(const edm::ParameterSet &params);
	virtual ~BeamSpotFakeConditions();
	ReturnType produce(const BeamSpotObjectsRcd &record);
private:
	void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,const edm::IOVSyncValue &syncValue,edm::ValidityInterval &oValidity);
	edm::FileInPath xmlCalibration;
	bool usedummy;
	std::string BeamType;
	
};

BeamSpotFakeConditions::BeamSpotFakeConditions(const edm::ParameterSet &params) :
	
	//xmlCalibration(params.getParameter<edm::FileInPath>("xmlCalibration") ), //disable until xml writer is fixed
	usedummy(params.getParameter<bool>("UseDummy") ),
	BeamType(params.getParameter<std::string>("BeamType") ) {
		
		setWhatProduced(this);
		findingRecord<BeamSpotObjectsRcd>();
}

BeamSpotFakeConditions::~BeamSpotFakeConditions(){}

BeamSpotFakeConditions::ReturnType
BeamSpotFakeConditions::produce(const BeamSpotObjectsRcd &record){


	if ( ! usedummy ) {
	  //TBufferXML code removed from here...		
	}
	else {

		BeamSpotObjects *adummy = new BeamSpotObjects();

		// we are going to use the truth values defined at the generator stage,
		// see IOMC/EventVertexGenerators/data
		
		if ( BeamType == "SimpleGaussian" || BeamType == "DummySigmaZ_5p3cm") {
			adummy->SetPosition(0.,0.,0.);
			adummy->SetSigmaZ(5.3);
			adummy->Setdxdz(0.);
			adummy->Setdydz(0.);
			adummy->SetBeamWidth(15.e-4);
		}

		else if ( BeamType == "EarlyCollision" ) {
			adummy->SetPosition(0.032206,-1.97386e-05,-0.282702);
			adummy->SetSigmaZ(5.3); // temporal
			adummy->Setdxdz(1.76367e-06);
			adummy->Setdydz(-2.58129e-05);
			adummy->SetBeamWidth(31.7e-4);
			adummy->SetCovariance(0,0,pow(6.96e-05,2));
			adummy->SetCovariance(1,1,pow(6.74e-5,2));
			adummy->SetCovariance(2,2,pow(0.70,2));
			adummy->SetCovariance(3,3,pow(0.1,2));// temporal
			adummy->SetCovariance(4,4,pow(9.74e-6,2));
			adummy->SetCovariance(5,5,pow(9.64e-6,2));
			adummy->SetCovariance(6,6,pow(2.0e-4,2));
		}

		else if ( BeamType == "NominalCollision" ) {
			adummy->SetPosition(0.05,0.,0.);
			adummy->SetSigmaZ(5.3);
			adummy->Setdxdz(140.e-6);
			adummy->Setdydz(0.);
			adummy->SetBeamWidth(16.6e-4);
		}
		// extreme cases
		else if ( BeamType == "NominalCollision1" ) {
			adummy->SetPosition(0.05,0.025,0.);
			adummy->SetSigmaZ(5.3);
			adummy->Setdxdz(0.);
			adummy->Setdydz(0.);
			adummy->SetBeamWidth(16.6e-4);
		}
		
		else if ( BeamType == "NominalCollision2" ) {
			adummy->SetPosition(0.05,0.025,0.);
			adummy->SetSigmaZ(5.3);
			adummy->Setdxdz(140.e-6);
			adummy->Setdydz(0.);
			adummy->SetBeamWidth(16.6e-4);
		}

		else if ( BeamType == "NominalCollision3" ) {
			adummy->SetPosition(0.1,0.025,0.);
			adummy->SetSigmaZ(5.3);
			adummy->Setdxdz(0.);
			adummy->Setdydz(0.);
			adummy->SetBeamWidth(16.6e-4);
		}

		else if ( BeamType == "NominalCollision4" ) {
			adummy->SetPosition(0.2,0.025,0.);
			adummy->SetSigmaZ(5.3);
			adummy->Setdxdz(0.);
			adummy->Setdydz(0.);
			adummy->SetBeamWidth(16.6e-4);
		}

		
		return ReturnType(adummy);
	}
  
	
}

void BeamSpotFakeConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
						  const edm::IOVSyncValue &syncValue,
						  edm::ValidityInterval &oValidity){
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),
				    edm::IOVSyncValue::endOfTime());
}

DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(BeamSpotFakeConditions);

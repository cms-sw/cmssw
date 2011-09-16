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
  edm::FileInPath inputFilename_;
  bool getDataFromFile_;
  double x,y,z,sigmaZ,dxdz,dydz,beamWidthX,beamWidthY,emittanceX,emittanceY,betastar;
  std::string tag;
  double cov[7][7];
  int type;
	
};

BeamSpotFakeConditions::BeamSpotFakeConditions(const edm::ParameterSet &params)
{	
		setWhatProduced(this);
		findingRecord<BeamSpotObjectsRcd>();
		getDataFromFile_ = params.getParameter<bool>("getDataFromFile");
		if (getDataFromFile_) {
		  inputFilename_   = params.getParameter<edm::FileInPath>("InputFilename");
		  std::ifstream fasciiFile(inputFilename_.fullPath().c_str() );
		  fasciiFile >> tag >> type;
		  fasciiFile >> tag >> x;
		  fasciiFile >> tag >> y;
		  fasciiFile >> tag >> z;
		  fasciiFile >> tag >> sigmaZ;
		  fasciiFile >> tag >> dxdz;
		  fasciiFile >> tag >> dydz;
		  fasciiFile >> tag >> beamWidthX;
		  fasciiFile >> tag >> beamWidthY;
		  fasciiFile >> tag >> cov[0][0] >> cov[0][1] >> cov[0][2]>> cov[0][3] >> cov[0][4]>> cov[0][5] >> cov[0][6]
		    ;
		  fasciiFile >> tag >> cov[1][0] >> cov[1][1] >> cov[1][2] >> cov[1][3]>> cov[1][4] >> cov[1][5]>> cov[1][6]
		    ;
		  fasciiFile >> tag >> cov[2][0]  >> cov[2][1] >> cov[2][2] >> cov[2][3]>> cov[2][4] >> cov[2][5]>> cov[2][6
															   ];
		  fasciiFile >> tag >> cov[3][0]  >> cov[3][1] >> cov[3][2] >> cov[3][3]>> cov[3][4] >> cov[3][5]>> cov[3][6
															   ];
		  fasciiFile >> tag >> cov[4][0] >> cov[4][1] >> cov[4][2] >> cov[4][3]>> cov[4][4] >> cov[4][5]>> cov[4][6]
		    ;
		  fasciiFile >> tag >> cov[5][0] >> cov[5][1] >> cov[5][2] >> cov[5][3]>> cov[5][4] >> cov[5][5]>> cov[5][6]
		    ;
		  fasciiFile >> tag >> cov[6][0] >> cov[6][1] >> cov[6][2] >> cov[6][3]>> cov[6][4] >> cov[6][5]>> cov[6][6]
		    ;
		  fasciiFile >> tag >> emittanceX;
		  fasciiFile >> tag >> emittanceY;
		  fasciiFile >> tag >> betastar;

		}
		// input values by hand
		else {
		  x =              params.getParameter<double>(  "X0" );
		  y =              params.getParameter<double>(  "Y0" );
		  z =              params.getParameter<double>(  "Z0" );
		  dxdz =           params.getParameter<double>(  "dxdz" );
		  dydz =           params.getParameter<double>(  "dydz" );
		  sigmaZ =         params.getParameter<double>(  "sigmaZ" );
		  beamWidthX =     params.getParameter<double>(  "widthX" );
		  beamWidthY =     params.getParameter<double>(  "widthY" );
		  emittanceX =     params.getParameter<double>(  "emittanceX" );
		  emittanceY =     params.getParameter<double>(  "emittanceY" );
		  betastar =       params.getParameter<double>(  "betaStar"  );

		  // first set all elements (esp. off-diagonal elements to zero)
		  for (int i=0; i<7; i++ ) {
		    for (int j=0; j<7; j++) cov[i][j] = 0.0;
		  }

		  // we ignore correlations when values are given by hand
		  cov[0][0] =       pow( params.getParameter<double>(  "errorX0" ), 2 );
		  cov[1][1] =       pow(    params.getParameter<double>(  "errorY0" ), 2 );
		  cov[2][2] =       pow(    params.getParameter<double>(  "errorZ0" ), 2 );
		  cov[3][3] =       pow( params.getParameter<double>(  "errorSigmaZ" ), 2 );
		  cov[4][4] =       pow( params.getParameter<double>(  "errordxdz" ), 2 );
		  cov[5][5] =       pow( params.getParameter<double>(  "errordydz" ), 2 );
		  cov[6][6] =       pow( params.getParameter<double>(  "errorWidth" ), 2 );
		  
		}
}

BeamSpotFakeConditions::~BeamSpotFakeConditions(){}

BeamSpotFakeConditions::ReturnType
BeamSpotFakeConditions::produce(const BeamSpotObjectsRcd &record){


	BeamSpotObjects *adummy = new BeamSpotObjects();
	
	adummy->SetPosition( x, y , z );
	adummy->SetSigmaZ( sigmaZ);
	adummy->Setdxdz( dxdz );
	adummy->Setdydz( dydz );
	adummy->SetBeamWidthX( beamWidthX );
	adummy->SetBeamWidthY( beamWidthY );
	for (int i=0; i<7; i++ ) {
	  for (int j=0; j<7; j++) {

	    adummy->SetCovariance( i, j, cov[i][j] );
	  } 
	}
	adummy->SetEmittanceX( emittanceX );
	adummy->SetEmittanceY( emittanceY );
	adummy->SetBetaStar( betastar);

	return ReturnType(adummy);
}

void BeamSpotFakeConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
						  const edm::IOVSyncValue &syncValue,
						  edm::ValidityInterval &oValidity){
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),
				    edm::IOVSyncValue::endOfTime());
}

DEFINE_FWK_EVENTSETUP_SOURCE(BeamSpotFakeConditions);

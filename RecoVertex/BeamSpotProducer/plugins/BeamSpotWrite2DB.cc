/**_________________________________________________________________
   class:   BeamSpotWrite2DB.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotWrite2DB.cc,v 1.7 2010/02/20 21:01:52 wmtan Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotWrite2DB.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

BeamSpotWrite2DB::BeamSpotWrite2DB(const edm::ParameterSet& iConfig)
{

  fasciiFileName = iConfig.getUntrackedParameter<std::string>("OutputFileName");
  
  fasciiFile.open(fasciiFileName.c_str());
  
}


BeamSpotWrite2DB::~BeamSpotWrite2DB()
{
}


void
BeamSpotWrite2DB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}



void 
BeamSpotWrite2DB::beginJob()
{
}

void 
BeamSpotWrite2DB::endJob() {

	std::cout << " Read beam spot data from text file: " << fasciiFileName << std::endl;
	std::cout << " please see plugins/BeamSpotWrite2DB.cc for format of text file." << std::endl;
	/*
	std::cout << " Content of the file is expected to have this format with the first column as a keyword:" << std::endl;
	std::cout << " x\n y\n z\n sigmaZ\n dxdz\n dydz\n beamWidthX\n beamWidthY" << std::endl;
	for (int i =0; i<7; i++) {
		for (int j=0; j<7; j++ ) {
			
			std::cout << " cov["<<i<<"]["<<j<<"] cov["<<i<<"]["<<j<<"] cov["<<i<<"]["<<j<<"] cov["<<i<<"]["<<j<<"] cov["<<i<<"]["<<j<<"] cov["<<j<<"]["<<j<<"] cov["<<i<<"]["<<j<<"]" << std::endl;
		}
	}
	*/
	
	// extract from file
	double x,y,z,sigmaZ,dxdz,dydz,beamWidthX,beamWidthY,emittanceX,emittanceY,betastar;
	std::string tag;
	double cov[7][7];
	int type;

	fasciiFile >> tag >> type;
	fasciiFile >> tag >> x;
	fasciiFile >> tag >> y;
	fasciiFile >> tag >> z;
	fasciiFile >> tag >> sigmaZ;
	fasciiFile >> tag >> dxdz;
	fasciiFile >> tag >> dydz;
	fasciiFile >> tag >> beamWidthX;
	fasciiFile >> tag >> beamWidthY;
	fasciiFile >> tag >> cov[0][0] >> cov[0][1] >> cov[0][2]>> cov[0][3] >> cov[0][4]>> cov[0][5] >> cov[0][6];
	fasciiFile >> tag >> cov[1][0] >> cov[1][1] >> cov[1][2] >> cov[1][3]>> cov[1][4] >> cov[1][5]>> cov[1][6];
	fasciiFile >> tag >> cov[2][0]  >> cov[2][1] >> cov[2][2] >> cov[2][3]>> cov[2][4] >> cov[2][5]>> cov[2][6];
	fasciiFile >> tag >> cov[3][0]  >> cov[3][1] >> cov[3][2] >> cov[3][3]>> cov[3][4] >> cov[3][5]>> cov[3][6];
	fasciiFile >> tag >> cov[4][0] >> cov[4][1] >> cov[4][2] >> cov[4][3]>> cov[4][4] >> cov[4][5]>> cov[4][6];
	fasciiFile >> tag >> cov[5][0] >> cov[5][1] >> cov[5][2] >> cov[5][3]>> cov[5][4] >> cov[5][5]>> cov[5][6];
	fasciiFile >> tag >> cov[6][0] >> cov[6][1] >> cov[6][2] >> cov[6][3]>> cov[6][4] >> cov[6][5]>> cov[6][6];
	fasciiFile >> tag >> emittanceX;
	fasciiFile >> tag >> emittanceY;
	fasciiFile >> tag >> betastar;
	
	BeamSpotObjects *abeam = new BeamSpotObjects();
	
	abeam->SetType(type);
	abeam->SetPosition(x,y,z);
	abeam->SetSigmaZ(sigmaZ);
	abeam->Setdxdz(dxdz);
	abeam->Setdydz(dydz);
	abeam->SetBeamWidthX(beamWidthX);
	abeam->SetBeamWidthY(beamWidthY);
	abeam->SetEmittanceX( emittanceX );
	abeam->SetEmittanceY( emittanceY );
	abeam->SetBetaStar( betastar );
	
	for (int i=0; i<7; ++i) {
	  for (int j=0; j<7; ++j) {
	    abeam->SetCovariance(i,j,cov[i][j]);
	  }
	}

	std::cout << " write results to DB..." << std::endl;

	edm::Service<cond::service::PoolDBOutputService> poolDbService;
	if( poolDbService.isAvailable() ) {
	  std::cout << "poolDBService available"<<std::endl;
	  if ( poolDbService->isNewTagRequest( "BeamSpotObjectsRcd" ) ) {
	    std::cout << "new tag requested" << std::endl;
	    poolDbService->createNewIOV<BeamSpotObjects>( abeam, poolDbService->beginOfTime(),poolDbService->endOfTime(),
								  "BeamSpotObjectsRcd"  );
	  }
	  else {
	    std::cout << "no new tag requested" << std::endl;
	    poolDbService->appendSinceTime<BeamSpotObjects>( abeam, poolDbService->currentTime(),
							     "BeamSpotObjectsRcd" );
	  }
	  
	}

	std::cout << "[BeamSpotWrite2DB] endJob done \n" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotWrite2DB);

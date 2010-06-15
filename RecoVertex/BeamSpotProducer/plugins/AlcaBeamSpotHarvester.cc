
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */

#include "RecoVertex/BeamSpotProducer/interface/AlcaBeamSpotHarvester.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <iostream> 


using namespace edm;
using namespace reco;


AlcaBeamSpotHarvester::AlcaBeamSpotHarvester(const edm::ParameterSet&){}

AlcaBeamSpotHarvester::~AlcaBeamSpotHarvester(){}



void AlcaBeamSpotHarvester::beginJob() {}
void AlcaBeamSpotHarvester::endJob() {}  
void AlcaBeamSpotHarvester::analyze(const edm::Event&, const edm::EventSetup&) {}
void AlcaBeamSpotHarvester::beginRun(const edm::Run&, const edm::EventSetup&) {}
void AlcaBeamSpotHarvester::endRun(const edm::Run&, const edm::EventSetup&) {}
void AlcaBeamSpotHarvester::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}
void AlcaBeamSpotHarvester::endLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup&) {
  
  Handle<reco::BeamSpot> bsHandle;
  lumi.getByLabel("alcaBeamSpotProducer","alcaBeamSpot", bsHandle);

  const BeamSpot *bs =  0;
  if(bsHandle.isValid()) { // check the product
      bs = bsHandle.product();
      edm::LogInfo("AlcaBeamSpotHarvester")
          << "Lumi: " << lumi.luminosityBlock() << std::endl;
      edm::LogInfo("AlcaBeamSpotHarvester")
          << *bs << std::endl;

      // create the DB object
      BeamSpotObjects *abeam = new BeamSpotObjects();
      abeam->SetType(bs->type());
      abeam->SetPosition(bs->x0(),bs->y0(),bs->z0());
      abeam->SetSigmaZ(bs->sigmaZ());
      abeam->Setdxdz(bs->dxdz());
      abeam->Setdydz(bs->dydz());
      abeam->SetBeamWidthX(bs->BeamWidthX());
      abeam->SetBeamWidthY(bs->BeamWidthY());
      abeam->SetEmittanceX(bs->emittanceX());
      abeam->SetEmittanceY(bs->emittanceY());
      abeam->SetBetaStar(bs->betaStar() );
	
      for (int i=0; i<7; ++i) {
	for (int j=0; j<7; ++j) {
	  abeam->SetCovariance(i,j,bs->covariance(i,j));
	}
      }

      Service<cond::service::PoolDBOutputService> poolDbService;
      if(poolDbService.isAvailable() ) {
	if (poolDbService->isNewTagRequest( "BeamSpotObjectsRcd" ) ) {
            edm::LogInfo("AlcaBeamSpotHarvester")
	        << "new tag requested" << std::endl;
	    poolDbService->createNewIOV<BeamSpotObjects>(abeam, poolDbService->beginOfTime(),poolDbService->endOfTime(),
								  "BeamSpotObjectsRcd"  );
	} 
	else {
          edm::LogInfo("AlcaBeamSpotHarvester")
	      << "no new tag requested" << std::endl;
	  poolDbService->appendSinceTime<BeamSpotObjects>( abeam, poolDbService->currentTime(),
							   "BeamSpotObjectsRcd" );
	}
	
      }

  } 
  else {
    edm::LogInfo("AlcaBeamSpotHarvester")
        << "Lumi: " << lumi.luminosityBlock() << std::endl;
    edm::LogInfo("AlcaBeamSpotHarvester")
        << "   BS is not valid!" << std::endl;
  }
}


DEFINE_FWK_MODULE(AlcaBeamSpotHarvester);

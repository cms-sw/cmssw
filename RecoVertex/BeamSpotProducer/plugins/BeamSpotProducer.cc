
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotProducer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"

//
// constructors and destructor
//
BeamSpotProducer::BeamSpotProducer(const edm::ParameterSet& iConf) {
	
	edm::LogInfo("RecoVertex/BeamSpotProducer") 
		<< "Initializing Beam Spot producer " << "\n";
  
	//fVerbose=conf.getUntrackedParameter<bool>("verbose", false);
	
	produces<reco::BeamSpot>();

}


BeamSpotProducer::~BeamSpotProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
BeamSpotProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
	
	using namespace edm;

	std::auto_ptr<reco::BeamSpot> result(new reco::BeamSpot);

	reco::BeamSpot aSpot;

	
	//typedef math::XYZPoint Point;
    //enum { dimension = 7 };
    //typedef math::Error<dimension>::type CovarianceMatrix;

	
	//try {
	edm::LogInfo("RecoVertex/BeamSpotProducer") 
	  << "Reconstructing event number: " << iEvent.id() << "\n";

	edm::ESHandle< BeamSpotObjects > beamhandle;
	iSetup.get<BeamSpotObjectsRcd>().get(beamhandle);
	const BeamSpotObjects *spotDB = beamhandle.product();

	// translate from BeamSpotObjects to reco::BeamSpot
	reco::BeamSpot::Point apoint( spotDB->GetX(), spotDB->GetY(), spotDB->GetZ() );
		
	reco::BeamSpot::CovarianceMatrix matrix;
	for ( int i=0; i<7; ++i ) {
	  for ( int j=0; j<7; ++j ) {
	    matrix(i,j) = spotDB->GetCovariance(i,j);
	  }
	}
	
	// this assume beam width same in x and y
	aSpot = reco::BeamSpot( apoint,
				spotDB->GetSigmaZ(),
				spotDB->Getdxdz(),
				spotDB->Getdydz(),
				spotDB->GetBeamWidthX(),
				matrix );
	aSpot.setBeamWidthY( spotDB->GetBeamWidthY() );
	aSpot.setEmittanceX( spotDB->GetEmittanceX() );
	aSpot.setEmittanceY( spotDB->GetEmittanceY() );
	aSpot.setbetaStar( spotDB->GetBetaStar() );
		
	//}
	//
	//catch (std::exception & err) {
	//	edm::LogInfo("RecoVertex/BeamSpotProducer") 
	//		<< "Exception during event number: " << iEvent.id() 
	//		<< "\n" << err.what() << "\n";
	//}

	*result = aSpot;
	
	iEvent.put(result);
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotProducer);


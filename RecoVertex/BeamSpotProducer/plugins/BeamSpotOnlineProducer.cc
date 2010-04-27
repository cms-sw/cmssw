
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotOnlineProducer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;


BeamSpotOnlineProducer::BeamSpotOnlineProducer(const ParameterSet& iconf)
{
  
  scalertag_ = iconf.getParameter<InputTag>("label");
  changeFrame_ = iconf.getParameter<bool>("changeToCMSCoordinates");

  produces<reco::BeamSpot>();

} 

BeamSpotOnlineProducer::~BeamSpotOnlineProducer() {}

void
BeamSpotOnlineProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  // get scalar collection
  Handle<BeamSpotOnlineCollection> handleScaler;
  iEvent.getByLabel( scalertag_, handleScaler);

  // beam spot scalar object
  BeamSpotOnline spotOnline;

  // get one element
  spotOnline = * ( handleScaler->begin() );

  // product is a reco::BeamSpot object
  std::auto_ptr<reco::BeamSpot> result(new reco::BeamSpot);

  reco::BeamSpot aSpot;

  // in case we need to switch to LHC reference frame
  // ignore for the moment rotations, and translations
  double f = 1.;
  if (changeFrame_) f = -1.;

  reco::BeamSpot::Point apoint( f* spotOnline.x(), spotOnline.y(), f* spotOnline.z() );

  reco::BeamSpot::CovarianceMatrix matrix;
  matrix(0,0) = spotOnline.err_x()*spotOnline.err_x();
  matrix(1,1) = spotOnline.err_y()*spotOnline.err_y();
  matrix(2,2) = spotOnline.err_z()*spotOnline.err_z();
  matrix(3,3) = spotOnline.err_sigma_z()*spotOnline.err_sigma_z();

  aSpot = reco::BeamSpot( apoint,
			  spotOnline.sigma_z(),
			  spotOnline.dxdz(),
			  f* spotOnline.dydz(),
			  spotOnline.width_x(),
			  matrix);

  aSpot.setBeamWidthY( spotOnline.width_y() );
  aSpot.setEmittanceX( 0. );
  aSpot.setEmittanceY( 0. );
  aSpot.setbetaStar( 0.) ;
  aSpot.setType( reco::BeamSpot::LHC ); // flag value from scalars

  // check if we have a valid beam spot fit result from online DQM

  if ( spotOnline.x() == 0 &&
       spotOnline.y() == 0 &&
       spotOnline.z() == 0 &&
       spotOnline.width_x() == 0 &&
       spotOnline.width_y() == 0 ) {

    //edm::LogInfo("RecoVertex/BeamSpotProducer") 
    //<< "Online Beam Spot producer fall back to DB value " << "\n";

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
    aSpot.setType( reco::BeamSpot::Tracker );

  }
  
  *result = aSpot;

  iEvent.put(result);

}

DEFINE_FWK_MODULE(BeamSpotOnlineProducer);

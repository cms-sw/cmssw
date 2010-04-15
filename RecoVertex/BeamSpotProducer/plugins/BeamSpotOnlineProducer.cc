
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotOnlineProducer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"

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
BeamSpotOnlineProducer::produce(Event& iEvent, const EventSetup& iConfig)
{

  Handle<BeamSpotOnlineCollection> handleScaler;
  iEvent.getByLabel( scalertag_, handleScaler);

  BeamSpotOnline spotOnline;

  //  for (BeamSpotOnlineCollection::const_iterator iter = handleScaler->begin(); iter != handleScaler->end(); ++iter) {
  //  spotOnline = *iter;
  //}

  spotOnline = * ( handleScaler->begin() );

  std::auto_ptr<reco::BeamSpot> result(new reco::BeamSpot);

  reco::BeamSpot aSpot;

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

  *result = aSpot;

  iEvent.put(result);

}

DEFINE_FWK_MODULE(BeamSpotOnlineProducer);

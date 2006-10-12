
#include "SimTracker/TrackAssociation/test/testTrackAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "Math/GenVector/BitReproducible.h"

#include <memory>
#include <iostream>
#include <string>

testTrackAssociator::testTrackAssociator(edm::ParameterSet const& conf) : 
  conf_(conf),
  doPixel_( conf.getParameter<bool>("associatePixel") ),
  doStrip_( conf.getParameter<bool>("associateStrip") ) 
{
  std::cout << " Constructor " << std::endl;
}

testTrackAssociator::~testTrackAssociator()
{
  std::cout << " Destructor " << std::endl;
}

void testTrackAssociator::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  std::cout << "\nEvent ID = "<< event.id() << std::endl ;

  //get reco tracks from the event
  edm::Handle<reco::TrackCollection> trackCollection;
  event.getByType(trackCollection);
  const reco::TrackCollection tC = *(trackCollection.product()); 
  std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
    std::cout << "\tmomentum: " << track->momentum()<< std::endl;
  }  

  int minHitFraction = 0;

  if(!doPixel_ && !doStrip_)  throw edm::Exception(errors::Configuration,"Strip and pixel association disabled");
  TrackAssociatorByHits tassociator(event, conf_);
  tassociator.AssociateByHitsRecoTrack(tC, minHitFraction);

  std::cout << "\ndone for now!" << std::endl;


  }





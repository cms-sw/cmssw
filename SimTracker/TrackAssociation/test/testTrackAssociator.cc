

using namespace std;


#include "SimTracker/TrackAssociation/test/testTrackAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociation.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociator.h"
#include "Math/GenVector/BitReproducible.h"

#include <memory>
#include <iostream>
#include <string>

class TrackAssociator; 
class TrackerHitAssociator; 

using namespace reco;

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
  using namespace edm;
  using namespace reco;
  

  std::cout << "\nEvent ID = "<< event.id() << std::endl ;

  int minHitFraction = 0;

  if(!doPixel_ && !doStrip_)  throw edm::Exception(errors::Configuration,"Strip and pixel association disabled");
  
  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  event.getByType(TPCollectionH);
  const TrackingParticleCollection * tPC   = TPCollectionH.product();
  std::cout << "Found " << tPC->size() << " TrackingParticles" << std::endl;

  edm::Handle<reco::TrackCollection> trackCollectionH;
  event.getByType(trackCollectionH);
  const reco::TrackCollection tC = *(trackCollectionH.product()); 
  std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
    std::cout << "\tmomentum: " << track->momentum()<< std::endl;
  }  
  
  
  TrackAssociator tassociator(event, conf_);
  const RecoToSimCollection* assocmap = tassociator.AssociateByHitsRecoTrack(trackCollectionH, 
									      TPCollectionH, minHitFraction);
  //now test map 
  std::cout << "Found " << assocmap->size() << " matched reco tracks" << std::endl;
  
  std::cout << "\ndone for now!" << std::endl;
  
  
}





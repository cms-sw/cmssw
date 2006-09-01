

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
  
  TrackAssociator tassociator(event, conf_);
  
  const RecoToSimCollection assocmap = tassociator.AssociateByHitsRecoTrack(minHitFraction);
  //now test map 
  std::cout << "Found " << assocmap.size() << " matched reco tracks" << std::endl;


  edm::Handle<reco::TrackCollection> trackCollectionH;
  event.getByType(trackCollectionH);
  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 
  
  for(reco::TrackCollection::size_type i=0; i<tC.size(); ++i){
    reco::TrackRef track(trackCollectionH, i);
    try{
      TrackingParticleRefVector tp = assocmap[track];
      std::cout << "->   Track " << setw(2) << track.index() << " pT: " << setprecision(2) << setw(6) << track->pt() 
		<<  " matched to " << tp.size() << " MC Tracks" << std::endl;
      
      for (TrackingParticleRefVector::const_iterator it = tp.begin(); it != tp.end(); ++it) {
	std::cout << "   MCTrack " << setw(2) << (*it).index() << " pT: " << setprecision(2) << setw(6) << (**it).pt() << endl;
      }
    } catch (edm::Exception event) {
      std::cout << "->   Track " << setw(2) << track.index() << " pT: " << setprecision(2) << setw(6) << track->pt() 
		<<  " matched to 0  MC Tracks" << std::endl;
    }
  }
  
}





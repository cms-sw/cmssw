

using namespace std;


#include "SimTracker/TrackAssociation/test/testTrackAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociation.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "Math/GenVector/BitReproducible.h"

#include <memory>
#include <iostream>
#include <string>

class TrackAssociator; 
class TrackerHitAssociator; 

using namespace reco;
using namespace std;
using namespace edm;


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

void testTrackAssociator::beginJob(const EventSetup & setup) {

  associator = new TrackAssociatorByChi2(setup);

}

void testTrackAssociator::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  using namespace edm;
  using namespace reco;
  
  Handle<reco::TrackCollection> trackCollectionH;
  event.getByLabel("ctfWithMaterialTracks",trackCollectionH);
  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 
  
  Handle<SimTrackContainer> simTrackCollection;
  event.getByLabel("g4SimHits", simTrackCollection);
  const SimTrackContainer simTC = *(simTrackCollection.product());
  
  Handle<SimVertexContainer> simVertexCollection;
  event.getByLabel("g4SimHits", simVertexCollection);
  const SimVertexContainer simVC = *(simVertexCollection.product());

  cout << "\nEvent ID = "<< event.id() << endl ;

  //Test TrackAssociatorByChi2
  TrackAssociatorByChi2::RecoToSimPairAssociation q =  associator->compareTracksParam(tC,simTC,simVC);
  for (TrackAssociatorByChi2::RecoToSimPairAssociation::iterator vit=q.begin();vit!=q.end();++vit){
    double chi2 = vit->second.begin()->first;
    reco::Track& rt = vit->first;
    SimTrack& st = vit->second.begin()->second;
    cout << "Chi2 associator - chi2 value: " << chi2 << endl;
    cout << "Chi2 associator - pt residue: " << rt.pt()-st.momentum().perp() << endl;
  }


  //Test AssociateByHitsRecoTrack
  int minHitFraction = 0;

  if(!doPixel_ && !doStrip_)  throw Exception(errors::Configuration,"Strip and pixel association disabled");
  
  TrackAssociator tassociator(event, conf_);
  
  const RecoToSimCollection assocmap = tassociator.AssociateByHitsRecoTrack(minHitFraction);
  //now test map 
  cout << "Found " << assocmap.size() << " matched reco tracks" << std::endl;

  for(TrackCollection::size_type i=0; i<tC.size(); ++i){
    TrackRef track(trackCollectionH, i);
    try{
      TrackingParticleRefVector tp = assocmap[track];
      cout << "->   Track " << setw(2) << track.index() << " pT: " << setprecision(2) << setw(6) << track->pt() 
		<<  " matched to " << tp.size() << " MC Tracks" << std::endl;
      
      for (TrackingParticleRefVector::const_iterator it = tp.begin(); it != tp.end(); ++it) {
	cout << "   MCTrack " << setw(2) << (*it).index() << " pT: " << setprecision(2) << setw(6) << (**it).pt() << endl;
      }
    } catch (Exception event) {
      cout << "->   Track " << setw(2) << track.index() << " pT: " << setprecision(2) << setw(6) << track->pt() 
		<<  " matched to 0  MC Tracks" << endl;
    }
  }
  

}





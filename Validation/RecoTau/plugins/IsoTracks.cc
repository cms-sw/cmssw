////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class IsoTracks : public edm::EDProducer
{
public:
  // construction/destruction
  IsoTracks(const edm::ParameterSet& iConfig);
  virtual ~IsoTracks();
  
  // member functions
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup);
  void endJob();

private:  
  // member data
  edm::InputTag  src_        ;
  double         coneRadius_ ;
  double         threshold_  ;

};


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
IsoTracks::IsoTracks(const edm::ParameterSet& iConfig)
  : src_       (iConfig.getParameter<edm::InputTag>("src"))
  , coneRadius_(iConfig.getParameter<double>("radius"))
  , threshold_ (iConfig.getParameter<double>("SumPtFraction"))
{
  produces<std::vector<reco::Track> >();
}

//______________________________________________________________________________
IsoTracks::~IsoTracks(){}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void IsoTracks::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{

  std::auto_ptr<std::vector<reco::Track> > IsoTracks(new std::vector<reco::Track >);
  
  edm::Handle< std::vector<reco::Track> > dirtyTracks;
  iEvent.getByLabel(src_,dirtyTracks);
  
  if( dirtyTracks->size() == 0 ) 
  {
    iEvent.put(IsoTracks);
    return ;
  }
  
  std::vector<reco::Track>::const_iterator dirtyTrackIt    ;
  std::vector<reco::Track>::const_iterator dirtyTrackIt2   ;
//  typename std::vector<reco::Track>::const_iterator dirtyTrackIt    ;
//  typename std::vector<reco::Track>::const_iterator dirtyTrackIt2   ;
  double   sumPtInCone = 0 ;

  for ( dirtyTrackIt = dirtyTracks->begin(); dirtyTrackIt != dirtyTracks->end(); ++dirtyTrackIt ) {
    for ( dirtyTrackIt2 = dirtyTracks->begin(); dirtyTrackIt2 != dirtyTracks->end(); ++dirtyTrackIt2 ) {
      if ( dirtyTrackIt == dirtyTrackIt2) continue ;
      if ( deltaR(dirtyTrackIt  -> eta() , 
                  dirtyTrackIt  -> phi() , 
                  dirtyTrackIt2 -> eta() , 
                  dirtyTrackIt2 -> phi() ) < coneRadius_ ){
        sumPtInCone = sumPtInCone + dirtyTrackIt2 -> pt() ;
      }
	}
	if ( sumPtInCone <= threshold_*(dirtyTrackIt->pt()) ){
	  IsoTracks -> push_back( *dirtyTrackIt ) ; 
	}
  }
  iEvent.put(IsoTracks);
}

void IsoTracks::endJob()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(IsoTracks);

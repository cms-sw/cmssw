#ifndef TrackingToolsPatternToolsTrackCollectionTokens_H
#define	TrackingToolsPatternToolsTrackCollectionTokens_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FWCore/Framework/interface/Event.h"


struct TrackCollectionTokens {
    TrackCollectionTokens(edm::InputTag const & tag, edm::ConsumesCollector && iC) :
      hTrackToken_( iC.consumes<reco::TrackCollection>( tag ) ),
      hTrajToken_( iC.mayConsume< std::vector<Trajectory> >( tag ) ),
      hTTAssToken_( iC.mayConsume< TrajTrackAssociationCollection >( tag ) ){}
    
    /// source collection label
    edm::EDGetTokenT<reco::TrackCollection> hTrackToken_;
    edm::EDGetTokenT< std::vector<Trajectory> > hTrajToken_;
    edm::EDGetTokenT< TrajTrackAssociationCollection > hTTAssToken_;


    reco::TrackCollection const & tracks(edm::Event& evt) const {
       edm::Handle<reco::TrackCollection> h;
       evt.getByToken( hTrackToken_, h);
       return *h;
    }

    std::vector<Trajectory> const & trajectories(edm::Event& evt) const {
       edm::Handle<std::vector<Trajectory>> h;
       evt.getByToken( hTrajToken_, h );
       return *h;
    }


};


#endif


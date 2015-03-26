#ifndef SimDataFormats_Associations_MuonToTrackingParticleAssociatorBaseImpl_h
#define SimDataFormats_Associations_MuonToTrackingParticleAssociatorBaseImpl_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/MuonTrackType.h"

namespace reco {
  class MuonToTrackingParticleAssociatorBaseImpl  {
    
  public:
    
    MuonToTrackingParticleAssociatorBaseImpl ();
    virtual ~MuonToTrackingParticleAssociatorBaseImpl();
    
    virtual void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                                const edm::RefToBaseVector<reco::Muon> & muons, MuonTrackType type,
                                const edm::RefVector<TrackingParticleCollection>& tpColl) const  = 0;
    
    virtual void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                                const edm::Handle<edm::View<reco::Muon> > & muons, MuonTrackType type, 
                                const edm::Handle<TrackingParticleCollection>& tpColl) const = 0;
  };
}

#endif

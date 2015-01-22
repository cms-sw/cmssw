#ifndef SimDataFormats_Associations_MuonToTrackingParticleAssociator_h
#define SimDataFormats_Associations_MuonToTrackingParticleAssociator_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/MuonTrackType.h"
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociatorBaseImpl.h"

#include <memory>

namespace reco {
  class MuonToTrackingParticleAssociator  {
    
  public:
    
    MuonToTrackingParticleAssociator ();
    ~MuonToTrackingParticleAssociator ();
#ifndef __GCCXML__
    MuonToTrackingParticleAssociator(std::unique_ptr<MuonToTrackingParticleAssociatorBaseImpl>);
#endif
    
    void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                        const edm::RefToBaseVector<reco::Muon> & muons, MuonTrackType type,
                        const edm::RefVector<TrackingParticleCollection>& tpColl) const {
      impl_->associateMuons(recoToSim, simToReco, muons, type, tpColl);
    }
    void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                        const edm::Handle<edm::View<reco::Muon> > & muons, MuonTrackType type, 
                        const edm::Handle<TrackingParticleCollection>& tpColl) const {
      impl_->associateMuons(recoToSim, simToReco, muons, type, tpColl);
    }

    void swap(MuonToTrackingParticleAssociator& iOther) {
      std::swap(impl_, iOther.impl_);
    }

  private:
    MuonToTrackingParticleAssociator( const MuonToTrackingParticleAssociator&);
    MuonToTrackingParticleAssociator& operator=( const MuonToTrackingParticleAssociator&);

    MuonToTrackingParticleAssociatorBaseImpl const* impl_;
  };
}

#endif

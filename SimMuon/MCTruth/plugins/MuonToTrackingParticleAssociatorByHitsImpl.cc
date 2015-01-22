// -*- C++ -*-
//
// Package:     SimMuon/MCTruth
// Class  :     MuonToTrackingParticleAssociatorByHitsImpl
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 07 Jan 2015 21:35:56 GMT
//

// system include files

// user include files
#include "MuonToTrackingParticleAssociatorByHitsImpl.h"
#include "SimMuon/MCTruth/interface/TrackerMuonHitExtractor.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonToTrackingParticleAssociatorByHitsImpl::MuonToTrackingParticleAssociatorByHitsImpl(TrackerMuonHitExtractor const& iHitExtractor,
MuonAssociatorByHitsHelper::Resources const& iResources,
                                                                                             MuonAssociatorByHitsHelper const* iHelper):
  m_hitExtractor(&iHitExtractor),
  m_resources(iResources),
  m_helper(iHelper)
{
}


//
// member functions
//

//
// const member functions
//
void 
MuonToTrackingParticleAssociatorByHitsImpl::associateMuons(reco::MuonToSimCollection & recToSim, 
                                                           reco::SimToMuonCollection & simToRec,
                                                           const edm::RefToBaseVector<reco::Muon> & muons, 
                                                           reco::MuonTrackType type,
                                                           const edm::RefVector<TrackingParticleCollection>& tPC) const {
    /// PART 1: Fill MuonToSimAssociatorByHits::TrackHitsCollection
    MuonAssociatorByHitsHelper::TrackHitsCollection muonHitRefs;
    edm::OwnVector<TrackingRecHit> allTMRecHits;  // this I will fill in only for tracker muon hits from segments
    switch (type) {
        case reco::InnerTk: 
            for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                edm::RefToBase<reco::Muon> mur = *it;
                if (mur->track().isNonnull()) { 
                    muonHitRefs.push_back(std::make_pair(mur->track()->recHitsBegin(), mur->track()->recHitsEnd()));
                } else {
                    muonHitRefs.push_back(std::make_pair(allTMRecHits.data().end(), allTMRecHits.data().end()));
                }
            }
            break;
        case reco::OuterTk: 
            for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                edm::RefToBase<reco::Muon> mur = *it;
                if (mur->outerTrack().isNonnull()) { 
                    muonHitRefs.push_back(std::make_pair(mur->outerTrack()->recHitsBegin(), mur->outerTrack()->recHitsEnd()));
                } else {
                    muonHitRefs.push_back(std::make_pair(allTMRecHits.data().end(), allTMRecHits.data().end()));
                }
            }
            break;
        case reco::GlobalTk: 
            for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                edm::RefToBase<reco::Muon> mur = *it;
                if (mur->globalTrack().isNonnull()) { 
                    muonHitRefs.push_back(std::make_pair(mur->globalTrack()->recHitsBegin(), mur->globalTrack()->recHitsEnd()));
                } else {
                    muonHitRefs.push_back(std::make_pair(allTMRecHits.data().end(), allTMRecHits.data().end()));
               }
            }
            break;
        case reco::Segments: {
                // puts hits in the vector, and record indices
                std::vector<std::pair<size_t, size_t> >   muonHitIndices;
                for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                    edm::RefToBase<reco::Muon> mur = *it;
                    std::pair<size_t, size_t> indices(allTMRecHits.size(), allTMRecHits.size());
                    if (mur->isTrackerMuon()) {
                        std::vector<const TrackingRecHit *> hits = m_hitExtractor->getMuonHits(*mur);
                        for (std::vector<const TrackingRecHit *>::const_iterator ith = hits.begin(), edh = hits.end(); ith != edh; ++ith) {
                            allTMRecHits.push_back(**ith);
                        }
                        indices.second += hits.size();
                    }
                    muonHitIndices.push_back(indices);  
                }
                // convert indices into pairs of iterators to references
                typedef std::pair<size_t, size_t> index_pair;
                trackingRecHit_iterator hitRefBegin = allTMRecHits.data().begin();
                for (std::vector<std::pair<size_t, size_t> >::const_iterator idxs = muonHitIndices.begin(), idxend = muonHitIndices.end(); idxs != idxend; ++idxs) {
                    muonHitRefs.push_back(std::make_pair(hitRefBegin+idxs->first, 
                                                         hitRefBegin+idxs->second));
                }
                
            }
            break;
    }

    /// PART 2: call the association routines 
    auto recSimColl = m_helper->associateRecoToSimIndices(muonHitRefs,tPC, m_resources);
    for (auto it = recSimColl.begin(), ed = recSimColl.end(); it != ed; ++it) {
        edm::RefToBase<reco::Muon> rec = muons[it->first];
        std::vector<std::pair<TrackingParticleRef, double> > & tpAss  = recToSim[rec];
        for ( auto const & a : it->second) {
            tpAss.push_back(std::make_pair(tPC[a.idx], a.quality));
        }
    }
    auto simRecColl = m_helper->associateSimToRecoIndices(muonHitRefs,tPC,m_resources);
    for (auto it = simRecColl.begin(), ed = simRecColl.end(); it != ed; ++it) {
        TrackingParticleRef sim = tPC[it->first];
        std::vector<std::pair<edm::RefToBase<reco::Muon>, double> > & recAss = simToRec[sim];
        for ( auto const & a: it->second ) {
            recAss.push_back(std::make_pair(muons[a.idx], a.quality));
        }
    }
}
   
void
MuonToTrackingParticleAssociatorByHitsImpl::associateMuons(reco::MuonToSimCollection & recToSim, 
                                                           reco::SimToMuonCollection & simToRec,
                                                           const edm::Handle<edm::View<reco::Muon> > & tCH, 
                                                           reco::MuonTrackType type, 
                                                           const edm::Handle<TrackingParticleCollection>& tPCH) const {

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));
    
    edm::RefToBaseVector<reco::Muon> muonBaseRefVector;
    for (size_t i = 0; i < tCH->size(); ++i)
      muonBaseRefVector.push_back(tCH->refAt(i));

    associateMuons(recToSim, simToRec, muonBaseRefVector,type,tpc);
}

//
// static member functions
//

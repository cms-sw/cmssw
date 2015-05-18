#include "SimMuon/MCTruth/interface/MuonToSimAssociatorByHits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "SimMuon/MCTruth/interface/TrackerMuonHitExtractor.h"
#include <sstream>

using namespace reco;
using namespace std;

MuonToSimAssociatorByHits::MuonToSimAssociatorByHits (const edm::ParameterSet& conf, edm::ConsumesCollector && iC) :
  helper_(conf),
  conf_(conf),
  trackerHitAssociatorConfig_(conf,std::move(iC))
{
  TrackerMuonHitExtractor hitExtractor(conf_,std::move(iC)); 

  //hack for consumes
  RPCHitAssociator rpctruth(conf,std::move(iC));
  DTHitAssociator dttruth(conf,std::move(iC));
  CSCHitAssociator muonTruth(conf,std::move(iC));
}


MuonToSimAssociatorByHits::~MuonToSimAssociatorByHits()
{
}



void MuonToSimAssociatorByHits::associateMuons(MuonToSimCollection & recToSim, SimToMuonCollection & simToRec,
                                          const edm::Handle<edm::View<reco::Muon> > &tCH, MuonTrackType type,
                                          const edm::Handle<TrackingParticleCollection>&tPCH,
                                          const edm::Event * event, const edm::EventSetup * setup) const  {

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));
    
    edm::RefToBaseVector<reco::Muon> muonBaseRefVector;
    for (size_t i = 0; i < tCH->size(); ++i)
      muonBaseRefVector.push_back(tCH->refAt(i));

    associateMuons(recToSim, simToRec, muonBaseRefVector,type,tpc,event,setup);
}

void MuonToSimAssociatorByHits::associateMuons(MuonToSimCollection & recToSim, SimToMuonCollection & simToRec,
                      const edm::RefToBaseVector<reco::Muon> & muons, MuonTrackType trackType,
                      const edm::RefVector<TrackingParticleCollection>& tPC,
                      const edm::Event * event, const edm::EventSetup * setup) const {

    /// PART 1: Fill MuonToSimAssociatorByHits::TrackHitsCollection
    MuonAssociatorByHitsHelper::TrackHitsCollection muonHitRefs;
    edm::OwnVector<TrackingRecHit> allTMRecHits;  // this I will fill in only for tracker muon hits from segments
    switch (trackType) {
        case InnerTk: 
            for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                edm::RefToBase<reco::Muon> mur = *it;
                if (mur->track().isNonnull()) { 
                    muonHitRefs.push_back(std::make_pair(mur->track()->recHitsBegin(), mur->track()->recHitsEnd()));
                } else {
                    muonHitRefs.push_back(std::make_pair(allTMRecHits.data().end(), allTMRecHits.data().end()));
                }
            }
            break;
        case OuterTk: 
            for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                edm::RefToBase<reco::Muon> mur = *it;
                if (mur->outerTrack().isNonnull()) { 
                    muonHitRefs.push_back(std::make_pair(mur->outerTrack()->recHitsBegin(), mur->outerTrack()->recHitsEnd()));
                } else {
                    muonHitRefs.push_back(std::make_pair(allTMRecHits.data().end(), allTMRecHits.data().end()));
                }
            }
            break;
        case GlobalTk: 
            for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                edm::RefToBase<reco::Muon> mur = *it;
                if (mur->globalTrack().isNonnull()) { 
                    muonHitRefs.push_back(std::make_pair(mur->globalTrack()->recHitsBegin(), mur->globalTrack()->recHitsEnd()));
                } else {
                    muonHitRefs.push_back(std::make_pair(allTMRecHits.data().end(), allTMRecHits.data().end()));
               }
            }
            break;
        case Segments: {
                TrackerMuonHitExtractor hitExtractor(conf_); 
                hitExtractor.init(*event, *setup);
                // puts hits in the vector, and record indices
                std::vector<std::pair<size_t, size_t> >   muonHitIndices;
                for (edm::RefToBaseVector<reco::Muon>::const_iterator it = muons.begin(), ed = muons.end(); it != ed; ++it) {
                    edm::RefToBase<reco::Muon> mur = *it;
                    std::pair<size_t, size_t> indices(allTMRecHits.size(), allTMRecHits.size());
                    if (mur->isTrackerMuon()) {
                        std::vector<const TrackingRecHit *> hits = hitExtractor.getMuonHits(*mur);
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
    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHand;
    setup->get<TrackerTopologyRcd>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();
    
    
    // Tracker hit association  
    TrackerHitAssociator trackertruth(*event, trackerHitAssociatorConfig_);
    // CSC hit association
    CSCHitAssociator csctruth(*event,*setup,conf_);
    // DT hit association
    bool printRtS = true;
    DTHitAssociator dttruth(*event,*setup,conf_,printRtS);  
    // RPC hit association
    RPCHitAssociator rpctruth(*event,*setup,conf_);
   
    MuonAssociatorByHitsHelper::Resources resources = {tTopo, &trackertruth, &csctruth, &dttruth, &rpctruth};

    auto recSimColl = helper_.associateRecoToSimIndices(muonHitRefs,tPC,resources);
    for (auto it = recSimColl.begin(), ed = recSimColl.end(); it != ed; ++it) {
        edm::RefToBase<reco::Muon> rec = muons[it->first];
        std::vector<std::pair<TrackingParticleRef, double> > & tpAss  = recToSim[rec];
        for ( auto const & a : it->second) {
            tpAss.push_back(std::make_pair(tPC[a.idx], a.quality));
        }
    }
    auto simRecColl = helper_.associateSimToRecoIndices(muonHitRefs,tPC,resources);
    for (auto it = simRecColl.begin(), ed = simRecColl.end(); it != ed; ++it) {
        TrackingParticleRef sim = tPC[it->first];
        std::vector<std::pair<edm::RefToBase<reco::Muon>, double> > & recAss = simToRec[sim];
        for ( auto const & a: it->second ) {
            recAss.push_back(std::make_pair(muons[a.idx], a.quality));
        }
    }

}

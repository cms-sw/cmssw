#include <TMTrackTrigger/l1VertexFinder/interface/VertexProducer.h>

#include <iostream>
#include <vector>
#include <set>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/L1TVertex/interface/Vertex.h"

#include "TMTrackTrigger/l1VertexFinder/interface/Settings.h"
#include "TMTrackTrigger/l1VertexFinder/interface/VertexFinder.h"

#include "TMTrackTrigger/l1VertexFinder/interface/RecoVertexWithTP.h"

using namespace l1tVertexFinder;
using namespace std;

VertexProducer::VertexProducer(const edm::ParameterSet& iConfig):
  l1TracksToken_( consumes<TTTrackCollectionView>(iConfig.getParameter<edm::InputTag>("l1TracksInputTag")) )
{
  // Get configuration parameters
  settings_ = new Settings(iConfig);

  // Tame debug printout.
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(4);

  //--- Define EDM output to be written to file (if required)
  produces< l1t::VertexCollection >( "l1vertices" );
  produces< l1t::VertexCollection >( "l1vertextdr" );
}


void VertexProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

void VertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // bool runAnalysis = true;

  edm::Handle<TTTrackCollectionView> l1TracksHandle;
  iEvent.getByToken(l1TracksToken_, l1TracksHandle);

  std::vector<L1fittedTrackBase> l1Tracks;
  l1Tracks.reserve(l1TracksHandle->size());

  for(const auto& track : l1TracksHandle->ptrs())
    l1Tracks.push_back(L1fittedTrackBase(track));

  std::vector<const L1fittedTrackBase*> l1TrackPtrs;
  l1TrackPtrs.reserve(l1Tracks.size());
  for(const auto& track : l1Tracks){
    if(track.pt() > settings_->vx_TrackMinPt() ){
      if(track.pt() < 50 or track.getNumStubs() > 5 )
        l1TrackPtrs.push_back(&track);
    }
  }

  // FIXME: Check with Davide if the tracks should be filtered using the following cuts
  //   fittedTracks[i].second.accepted() and fittedTracks[i].second.chi2dof()< settings_->chi2OverNdfCut()
  VertexFinder vf(l1TrackPtrs, settings_);

  if(settings_->vx_algoId() == 0){
    cout << "Finding vertices using a gap clustering algorithm "<< endl;
    vf.GapClustering();
  } else if(settings_->vx_algoId() == 1){
    cout << "Finding vertices using a Simple Merge Clustering algorithm "<< endl;
    vf.AgglomerativeHierarchicalClustering();
  } else if(settings_->vx_algoId() == 2){
    cout << "Finding vertices using a DBSCAN algorithm "<< endl;
    vf.DBSCAN();
  } else if(settings_->vx_algoId() == 3){
    cout << "Finding vertices using a PVR algorithm "<< endl;
    vf.PVR();
  } else if(settings_->vx_algoId() == 4){
    cout << "Finding vertices using an AdaptiveVertexReconstruction algorithm "<< endl;
    vf.AdaptiveVertexReconstruction();
  } else if(settings_->vx_algoId() == 5){
    cout << "Finding vertices using an Highest Pt Vertex algorithm "<< endl;
    vf.HPV();
  } else if(settings_->vx_algoId() == 6){
    cout << "Finding vertices using a kmeans algorithm" << endl;
    vf.Kmeans();
  }
  else{
    cout << "No valid vertex reconstruction algorithm has been selected. Running a gap clustering algorithm "<< endl;
    vf.GapClustering();
  }

  vf.TDRalgorithm();
  vf.SortVerticesInZ0();
  vf.FindPrimaryVertex();

  // //=== Store output EDM track and hardware stub collections.
  std::unique_ptr<l1t::VertexCollection> lProduct(new std::vector<l1t::Vertex>());
  std::vector<edm::Ptr<l1t::Vertex::Track_t>> pvTracks;
  pvTracks.reserve(vf.PrimaryVertex().tracks().size());
  // store PV first in list
  for (const auto& track : vf.PrimaryVertex().tracks()) {
    pvTracks.emplace_back(track->getTTTrackPtr());
  }
  lProduct->emplace_back(l1t::Vertex(vf.PrimaryVertex().z0(), pvTracks));

  unsigned vIndex = 0;
  for (const auto& vtx : vf.Vertices()) {
    if (vIndex != vf.PrimaryVertexId()) {
      std::vector<edm::Ptr<l1t::Vertex::Track_t>> lVtxTracks;
      for (const auto& t : vtx.tracks() )
        lVtxTracks.push_back( t->getTTTrackPtr() );
      lProduct->emplace_back(l1t::Vertex(vtx.z0(), lVtxTracks));
    }
    ++vIndex;
  }
  iEvent.put(std::move(lProduct), "l1vertices");

  // //=== Store output EDM track and hardware stub collections.
  std::unique_ptr<l1t::VertexCollection> lProductTDR(new std::vector<l1t::Vertex>());
  std::vector<edm::Ptr<l1t::Vertex::Track_t>> lVtxTracksTDR;
  lVtxTracksTDR.reserve(vf.TDRPrimaryVertex().tracks().size());
  for (const auto& t : vf.TDRPrimaryVertex().tracks() )
    lVtxTracksTDR.emplace_back( t->getTTTrackPtr() );
  lProductTDR->emplace_back(l1t::Vertex(vf.TDRPrimaryVertex().z0(), lVtxTracksTDR));
  iEvent.put(std::move(lProductTDR), "l1vertextdr");
}

void VertexProducer::endJob() {}

DEFINE_FWK_MODULE(VertexProducer);

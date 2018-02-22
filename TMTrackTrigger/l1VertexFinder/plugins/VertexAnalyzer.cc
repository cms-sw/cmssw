#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

#include "TMTrackTrigger/l1VertexFinder/interface/InputData.h"
#include "TMTrackTrigger/l1VertexFinder/interface/Settings.h"
#include "TMTrackTrigger/l1VertexFinder/interface/Histos.h"
#include "TMTrackTrigger/l1VertexFinder/interface/L1fittedTrack.h"
#include "TMTrackTrigger/l1VertexFinder/interface/RecoVertexWithTP.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <map>
#include <vector>
#include <string>

#include <iostream>

using namespace std;
// using namespace l1tVertexFinder;

namespace l1tVertexFinder {

  class VertexAnalyzer : public edm::EDAnalyzer {

  public:
    explicit VertexAnalyzer(const edm::ParameterSet&);
    ~VertexAnalyzer();

  private:
    void beginJob() override;
    void analyze(const edm::Event& evt, const edm::EventSetup& setup);
    void endJob() override;

    // define types for stub-related classes
    typedef edmNew::DetSetVector< TTStub<Ref_Phase2TrackerDigi_> > DetSetVec;
    typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_>           TTStubAssMap;
    typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_>        TTClusterAssMap;
    typedef edm::View< TTTrack< Ref_Phase2TrackerDigi_ > > TTTrackCollectionView;

    // references to tags containing information relevant to perofrmance analysis
    const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag;
    const edm::EDGetTokenT<DetSetVec> stubInputTag;
    const edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag;
    const edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag;
    const edm::EDGetTokenT<TTTrackCollectionView> l1TracksToken_;

    const bool printResults_;

    // temporary histogramming class
    Histos * hists_;

    // storage class for configuration parameters
    Settings *settings_;
  };

  VertexAnalyzer::VertexAnalyzer(const edm::ParameterSet& iConfig):
    tpInputTag( consumes<TrackingParticleCollection>( iConfig.getParameter<edm::InputTag>("tpInputTag") ) ),
    stubInputTag( consumes<DetSetVec>( iConfig.getParameter<edm::InputTag>("stubInputTag") ) ),
    stubTruthInputTag( consumes<TTStubAssMap>( iConfig.getParameter<edm::InputTag>("stubTruthInputTag") ) ),
    clusterTruthInputTag( consumes<TTClusterAssMap>( iConfig.getParameter<edm::InputTag>("clusterTruthInputTag") ) ),
    l1TracksToken_( consumes<TTTrackCollectionView>(iConfig.getParameter<edm::InputTag>("l1TracksInputTag")) ),
    printResults_( iConfig.getParameter<bool>("printResults") )
  {
    // Get configuration parameters
    settings_ = new Settings(iConfig);

    // Book histograms.
    hists_ = new Histos( settings_ );
    hists_->book();
  }

  void VertexAnalyzer::beginJob() {};
  void VertexAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    std::cout << "processing " << std::endl;

    edm::Handle<TTStubAssMap>    mcTruthTTStubHandle;
    edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
    iEvent.getByToken(stubTruthInputTag, mcTruthTTStubHandle );
    iEvent.getByToken(clusterTruthInputTag, mcTruthTTClusterHandle );

    // Note useful info about MC truth particles and about reconstructed stubs .
    InputData inputData(iEvent, iSetup, settings_, tpInputTag, stubInputTag, stubTruthInputTag, clusterTruthInputTag );

    edm::Handle<TTTrackCollectionView> l1TracksHandle;
    iEvent.getByToken(l1TracksToken_, l1TracksHandle);

    std::vector<L1fittedTrack> l1Tracks;
    l1Tracks.reserve(l1TracksHandle->size());
    {
      // Get the tracker geometry info needed to unpack the stub info.
      edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
      iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometryHandle );
      edm::ESHandle<TrackerTopology> trackerTopologyHandle;
      iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);

      map<edm::Ptr< TrackingParticle >, const TP* > translateTP;
      for (const TP& tp : inputData.getTPs()) {
        TrackingParticlePtr tpPtr(tp);
        translateTP[tpPtr] = &tp;
      }

      edm::Handle<TTStubAssMap>    mcTruthTTStubHandle;
      edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
      iEvent.getByToken(stubTruthInputTag, mcTruthTTStubHandle );
      iEvent.getByToken(clusterTruthInputTag, mcTruthTTClusterHandle );

      for(const auto& track : l1TracksHandle->ptrs())
        l1Tracks.push_back(L1fittedTrack(track, *settings_, trackerGeometryHandle.product(), trackerTopologyHandle.product(), translateTP, mcTruthTTStubHandle, mcTruthTTClusterHandle, inputData.getStubGeoDetIdMap()));
    }

    std::vector<const L1fittedTrack*> l1TrackPtrs;
    l1TrackPtrs.reserve(l1Tracks.size());
    for(const auto& track : l1Tracks){
      if(track.pt() > settings_->vx_TrackMinPt() ){
        if(track.pt() < 50 or track.getNumStubs() > 5 )
          l1TrackPtrs.push_back(&track);
      }
    }

    /*
    if(settings_->vx_keepOnlyPV()){
      vf.FindPrimaryVertex();
    } else {
      vf.AssociatePrimaryVertex(inputData.getPrimaryVertex().z0());
    }
    */

    /*
    if(settings_->debug()==7 and vf.numVertices() > 0){
      cout << "Num Found Vertices " << vf.numVertices() << endl;
      cout << "Reconstructed Primary Vertex z0 "<<vf.PrimaryVertex().z0() << " pT "<< vf.PrimaryVertex().pT() << endl;
    }
    */

    //=== Fill histograms studying vertex reconstruction performance
    // hists_->fillVertexReconstruction(inputData, vf, l1Tracks);

    /*
    if (printResults_) {
      std::cout << vf.numVertices() << " vertices were found ... " << std::endl;
      for (const auto& vtx : vf.Vertices()) {
        std::cout << "  * z0 = " << vtx.z0() << "; contains " << vtx.numTracks() << " tracks ..." <<  std::endl;
        for (const auto& trackPtr : vtx.tracks())
          std::cout << "     - z0 = " << trackPtr->z0() << "; pt = " << trackPtr->pt() << ", eta = " << trackPtr->eta() << ", phi = " << trackPtr->phi0() << std::endl;
      }
    }
    */
  }
  void VertexAnalyzer::endJob() {};

  VertexAnalyzer::~VertexAnalyzer() {};
}

using namespace l1tVertexFinder;

//define this as a plug-in
DEFINE_FWK_MODULE(VertexAnalyzer);

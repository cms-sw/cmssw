
#include "Validation/RecoMuon/plugins/ME0MuonTrackCollProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
//#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "RecoMuon/MuonIdentification/plugins/ME0MuonSelector.cc"
#include <FWCore/Framework/interface/ESHandle.h>


#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>

#include <sstream>

ME0MuonTrackCollProducer::ME0MuonTrackCollProducer(const edm::ParameterSet& parset) :
  muonsTag(parset.getParameter< edm::InputTag >("muonsTag")),
  vxtTag(parset.getParameter< edm::InputTag >("vxtTag")),
  useIPxy(parset.getUntrackedParameter< bool >("useIPxy", true)),
  useIPz(parset.getUntrackedParameter< bool >("useIPz", true)),
  selectionTags(parset.getParameter< std::vector<std::string> >("selectionTags")),
  trackType(parset.getParameter< std::string >("trackType")),
  parset_(parset)
{
  produces<reco::TrackCollection>();
  edm::InputTag OurMuonsTag ("me0SegmentMatching");
  OurMuonsToken_ = consumes<ME0MuonCollection>(OurMuonsTag);
}

ME0MuonTrackCollProducer::~ME0MuonTrackCollProducer() {
}

void ME0MuonTrackCollProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;
  using namespace edm;
  Handle <ME0MuonCollection> OurMuons;
  iEvent.getByToken(OurMuonsToken_,OurMuons);

  edm::ESHandle<ME0Geometry> me0Geom;
  iSetup.get<MuonGeometryRecord>().get(me0Geom);

  
  std::auto_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
 
  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
  


  for(std::vector<reco::ME0Muon>::const_iterator thismuon = OurMuons->begin();
       thismuon != OurMuons->end(); ++thismuon) {

    if (!muon::isGoodMuon(me0Geom, *thismuon, muon::Tight)) continue;
    reco::TrackRef trackref;    

    if (thismuon->innerTrack().isNonnull()) trackref = thismuon->innerTrack();

      const reco::Track* trk = &(*trackref);
      // pointer to old track:
      //reco::Track* newTrk = new reco::Track(*trk);

      selectedTracks->push_back( *trk );
      //selectedTrackExtras->push_back( *newExtra );
  }
  iEvent.put(selectedTracks);

}

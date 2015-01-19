#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class dso_hidden QualityFilter final : public edm::stream::EDProducer<> {
 public:
  explicit QualityFilter(const edm::ParameterSet&);
  ~QualityFilter();
  
 private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
  
  // ----------member data ---------------------------
 private:
  
  edm::EDGetTokenT<std::vector<Trajectory> >       trajTag; 
  edm::EDGetTokenT<TrajTrackAssociationCollection> tassTag; 
  reco::TrackBase::TrackQuality trackQuality_;
  bool copyExtras_;  
};

#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class decleration
//

using namespace edm;
using namespace reco;
using namespace std;


QualityFilter::QualityFilter(const edm::ParameterSet& iConfig)
{
  copyExtras_ = iConfig.getUntrackedParameter<bool>("copyExtras", false);

  produces<reco::TrackCollection>();
  if (copyExtras_) {
      produces<TrackingRecHitCollection>();
      produces<reco::TrackExtraCollection>();
  }
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

  trajTag = consumes<std::vector<Trajectory> >(iConfig.getParameter<edm::InputTag>("recTracks"));
  tassTag = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("recTracks"));
  trackQuality_=TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));
}


QualityFilter::~QualityFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
QualityFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  
  auto_ptr<TrackCollection> selTracks(new TrackCollection);
  auto_ptr<TrackingRecHitCollection> selHits(copyExtras_ ? new TrackingRecHitCollection() : 0);
  auto_ptr<TrackExtraCollection> selTrackExtras(copyExtras_ ? new TrackExtraCollection() : 0);
  auto_ptr<vector<Trajectory> > outputTJ(new vector<Trajectory> );
  auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
  
  
  TrackExtraRefProd rTrackExtras; 
  TrackingRecHitRefProd rHits;
  if (copyExtras_) {
    rTrackExtras = iEvent.getRefBeforePut<TrackExtraCollection>();
    rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
  }

  Handle<std::vector<Trajectory> > TrajectoryCollection;
  Handle<TrajTrackAssociationCollection> assoMap;
  
  iEvent.getByToken(trajTag,TrajectoryCollection);
  iEvent.getByToken(tassTag,assoMap);



  TrajTrackAssociationCollection::const_iterator it = assoMap->begin();
  TrajTrackAssociationCollection::const_iterator lastAssoc = assoMap->end();
  for( ; it != lastAssoc; ++it ) {
    const Ref<vector<Trajectory> > traj = it->key;
    const reco::TrackRef itc = it->val;
    bool goodTk = (itc->quality(trackQuality_));
 
    
    if (goodTk){
      auto const & track =(*itc);
      //tracks and trajectories
      selTracks->push_back( track );
      outputTJ->push_back( *traj );
      if (copyExtras_) {
          //TRACKING HITS
          trackingRecHit_iterator irhit   =(*itc).recHitsBegin();
          trackingRecHit_iterator lasthit =(*itc).recHitsEnd();
          for (; irhit!=lasthit; ++irhit) {
            selHits->push_back((*irhit)->clone() );
          }
      }
          
    }

  }

  unsigned nTracks = selTracks->size();
  if (copyExtras_) {
      //PUT TRACKING HITS IN THE EVENT
      OrphanHandle<TrackingRecHitCollection> theRecoHits = iEvent.put(selHits );
      edm::RefProd<TrackingRecHitCollection> theRecoHitsProd(theRecoHits);

      //PUT TRACK EXTRA IN THE EVENT
      selTrackExtras->reserve(nTracks);
      unsigned hits=0;

      for ( unsigned index = 0; index<nTracks; ++index ) { 
        auto const & aTrack = (*selTracks)[index];
        selTrackExtras->emplace_back(aTrack.outerPosition(),
                               aTrack.outerMomentum(),
                               aTrack.outerOk(),
                               aTrack.innerPosition(),
                               aTrack.innerMomentum(),
                               aTrack.innerOk(),
                               aTrack.outerStateCovariance(),
                               aTrack.outerDetId(),
                               aTrack.innerStateCovariance(),
                               aTrack.innerDetId(),
                               aTrack.seedDirection(),
                               aTrack.seedRef()
                               );
            
        auto & aTrackExtra = selTrackExtras->back();
        //unsigned nHits = aTrack.numberOfValidHits();
        auto nHits = aTrack.recHitsSize();
        aTrackExtra.setHits(theRecoHitsProd, hits, nHits);
        hits += nHits;
        selTrackExtras->push_back(aTrackExtra);
      }

      //CORRECT REF TO TRACK
      OrphanHandle<TrackExtraCollection> theRecoTrackExtras = iEvent.put(selTrackExtras); 
      for ( unsigned index = 0; index<nTracks; ++index ) { 
        const reco::TrackExtraRef theTrackExtraRef(theRecoTrackExtras,index);
          (*selTracks)[index].setExtra(theTrackExtraRef);
      }
  } // END IF COPY EXTRAS

  //TRACKS AND TRAJECTORIES
  OrphanHandle<TrackCollection> theRecoTracks = iEvent.put(selTracks);
  OrphanHandle<vector<Trajectory> > theRecoTrajectories = iEvent.put(outputTJ);

  //TRACKS<->TRAJECTORIES MAP 
  nTracks = theRecoTracks->size();
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    Ref<vector<Trajectory> > trajRef( theRecoTrajectories, index );
    Ref<TrackCollection>    tkRef( theRecoTracks, index );
    trajTrackMap->insert(trajRef,tkRef);
  }
  //MAP IN THE EVENT
  iEvent.put( trajTrackMap );
}

// ------------ method called once each job just after ending the event loop  ------------
void 
QualityFilter::endJob() {
}



DEFINE_FWK_MODULE(QualityFilter);


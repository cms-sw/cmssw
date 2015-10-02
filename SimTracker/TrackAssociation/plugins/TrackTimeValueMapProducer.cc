// This producer assigns vertex times (with a specified resolution) to tracks.
// The times are produced as valuemaps associated to tracks, so the track dataformat doesn't
// need to be modified.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include <memory>

#include "ResolutionModel.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "CLHEP/Random/RandGauss.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/transform.h"

class TrackTimeValueMapProducer : public edm::global::EDProducer<> {
public:    
  TrackTimeValueMapProducer(const edm::ParameterSet&);
  ~TrackTimeValueMapProducer() { }
  
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
private:
  // inputs
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_;
  const edm::EDGetTokenT<edm::View<reco::Track> > gsfTracks_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_;
  const edm::EDGetTokenT<TrackingVertexCollection> trackingVertices_;
  const edm::EDGetTokenT<edm::HepMCProduct> hepMCProduct_;
  // tracking particle associators by order of preference
  const std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> > associators_;  
  // options
  std::vector<std::unique_ptr<const ResolutionModel> > resolutions_;
  // functions
  void calculateTrackTimes( const edm::View<reco::Track>&, 
                            const std::vector<reco::RecoToSimCollection>&,
                            std::vector<float>& ) const;
  std::pair<float,float> extractTrackVertexTime(const std::vector<std::pair<TrackingParticleRef, double> >&) const;
};

DEFINE_FWK_MODULE(TrackTimeValueMapProducer);

namespace {
  constexpr float fakeBeamSpotTimeWidth = 0.175f; // ns
  static const std::string generalTracksName("generalTracks");
  static const std::string gsfTracksName("gsfTracks");
  static const std::string resolution("Resolution");

  template<typename ParticleType, typename T>
  void writeValueMap(edm::Event &iEvent,
                     const edm::Handle<edm::View<ParticleType> > & handle,
                     const std::vector<T> & values,
                     const std::string    & label) {
    std::auto_ptr<edm::ValueMap<T> > valMap(new edm::ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap, label);
  }
}

TrackTimeValueMapProducer::TrackTimeValueMapProducer(const edm::ParameterSet& conf) :
  tracks_(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("trackSrc") ) ),
  gsfTracks_(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("gsfTrackSrc") ) ),
  trackingParticles_(consumes<TrackingParticleCollection>( conf.getParameter<edm::InputTag>("trackingParticleSrc") ) ),
  trackingVertices_(consumes<TrackingVertexCollection>( conf.getParameter<edm::InputTag>("trackingVertexSrc") ) ),
  associators_( edm::vector_transform( conf.getParameter<std::vector<edm::InputTag> >("associators"), [this](const edm::InputTag& tag){ return this->consumes<reco::TrackToTrackingParticleAssociator>(tag); } ) )
{
  // setup resolution models
  const std::vector<edm::ParameterSet>& resos = conf.getParameterSetVector("resolutionModels");
  for( const auto& reso : resos ) {
    const std::string& name = reso.getParameter<std::string>("modelName");
    ResolutionModel* resomod = ResolutionModelFactory::get()->create(name,reso);
    resolutions_.emplace_back( resomod );  

    // times and time resolutions for general tracks
    produces<edm::ValueMap<float> >(generalTracksName+name);
    produces<edm::ValueMap<float> >(generalTracksName+name+resolution);
    
    //for gsf tracks
    produces<edm::ValueMap<float> >(gsfTracksName+name);
    produces<edm::ValueMap<float> >(gsfTracksName+name+resolution);
  }
  // get RNG engine
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "TrackTimeValueMapProducer::TrackTimeValueMapProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }

}

void TrackTimeValueMapProducer::produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& es) const {
  // get RNG engine
  edm::Service<edm::RandomNumberGenerator> rng;  
  auto rng_engine = &(rng->getEngine(sid));

  // get sim track associators
  std::vector<edm::Handle<reco::TrackToTrackingParticleAssociator> > associators;
  for( const auto& token : associators_ ) {
    associators.emplace_back();
    auto& back = associators.back();
    evt.getByToken(token,back);
  }

  std::vector<float> generalTrackTimes, gsfTrackTimes;
 
  //get track collections
  edm::Handle<edm::View<reco::Track> > TrackCollectionH;
  evt.getByToken(tracks_, TrackCollectionH);
  const edm::View<reco::Track>& TrackCollection = *TrackCollectionH;

  edm::Handle<edm::View<reco::Track> > GsfTrackCollectionH;
  evt.getByToken(gsfTracks_, GsfTrackCollectionH);
  const edm::View<reco::Track>& GsfTrackCollection = *GsfTrackCollectionH;

  //get tracking particle collections
  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  evt.getByToken(trackingParticles_, TPCollectionH);
  //const TrackingParticleCollection&  TPCollection = *TPCollectionH;
  
  // associate the reco tracks / gsf Tracks
  std::vector<reco::RecoToSimCollection> associatedTracks, associatedTracksGsf;  
  for( auto associator : associators ) {
    associatedTracks.emplace_back(associator->associateRecoToSim(TrackCollectionH, TPCollectionH));
    associatedTracksGsf.emplace_back(associator->associateRecoToSim(GsfTrackCollectionH, TPCollectionH));
  }
  

  calculateTrackTimes(TrackCollection, associatedTracks, generalTrackTimes);
  calculateTrackTimes(GsfTrackCollection, associatedTracksGsf, gsfTrackTimes);

  for( const auto& reso : resolutions_ ) {
    const std::string& name = reso->name();
    std::vector<float> times, resos;
    std::vector<float> gsf_times, gsf_resos;
    
    times.reserve(TrackCollection.size());
    resos.reserve(TrackCollection.size());
    gsf_times.reserve(GsfTrackCollection.size());
    gsf_resos.reserve(GsfTrackCollection.size());

    for( unsigned i = 0; i < TrackCollection.size(); ++i ) {
      const reco::Track& tk = TrackCollection[i];
      if( edm::isFinite( generalTrackTimes[i] ) && generalTrackTimes[i] != 0.f) {
        const float resolution = reso->getTimeResolution(tk);
        times.push_back( CLHEP::RandGauss::shoot(rng_engine, generalTrackTimes[i], resolution) );
        resos.push_back( resolution );
      } else {
        times.push_back( generalTrackTimes[i] );
        resos.push_back( fakeBeamSpotTimeWidth );
      }
    }

    for( unsigned i = 0; i < GsfTrackCollection.size(); ++i ) {
      const reco::Track& tk = GsfTrackCollection[i];
      if( edm::isFinite( gsfTrackTimes[i] )  && gsfTrackTimes[i] != 0.f ) {
        const float resolution = reso->getTimeResolution(tk);
        gsf_times.push_back( CLHEP::RandGauss::shoot(rng_engine, gsfTrackTimes[i], resolution) );
        gsf_resos.push_back( resolution ); 
      } else {
        gsf_times.push_back( gsfTrackTimes[i] );
        gsf_resos.push_back( fakeBeamSpotTimeWidth );
      }
    }

    writeValueMap( evt, TrackCollectionH, times, generalTracksName+name );
    writeValueMap( evt, TrackCollectionH, resos, generalTracksName+name+resolution );
    writeValueMap( evt, GsfTrackCollectionH, gsf_times, gsfTracksName+name );
    writeValueMap( evt, GsfTrackCollectionH, gsf_resos, gsfTracksName+name+resolution );
  }
}

void TrackTimeValueMapProducer::calculateTrackTimes( const edm::View<reco::Track>& tkcoll,
                                                     const std::vector<reco::RecoToSimCollection>& assocs,
                                                     std::vector<float>& tvals ) const { 
  constexpr float flt_max = std::numeric_limits<float>::quiet_NaN();
  
  for( unsigned itk = 0; itk < tkcoll.size(); ++itk ) {
    const auto tkref = tkcoll.refAt(itk);
    reco::RecoToSimCollection::const_iterator track_tps = assocs.back().end();
    for( const auto& association : assocs ) {
      track_tps = association.find(tkref);
      if( track_tps != association.end() ) break;
    }
    if( track_tps != assocs.back().end() ) {
      if( !track_tps->val.size() ) {
        tvals.push_back(flt_max);
      } else {        
        const std::pair<float,float> time_info = extractTrackVertexTime(track_tps->val);
        tvals.push_back(time_info.first);
      }
    } else {
      tvals.push_back(flt_max);
    }
  } 
}

std::pair<float,float> TrackTimeValueMapProducer::
extractTrackVertexTime( const std::vector<std::pair<TrackingParticleRef, double> >& tp_list ) const {
  float result = 0.f;
  float result_z = 0.f;
  for( const auto& tpref : tp_list ) {
    const auto& tvertex = tpref.first->parentVertex();
    result = tvertex->position().T()*CLHEP::second; // convert into nano-seconds
    result_z = tvertex->position().Z();
    // account for secondary vertices...
    
    if( tvertex->nSourceTracks() ) {
      auto pvertex = tvertex->sourceTracks()[0]->parentVertex();
      result = pvertex->position().T()*CLHEP::second;
      result_z = pvertex->position().Z();
      while( pvertex->nSourceTracks() ) {
        pvertex = pvertex->sourceTracks()[0]->parentVertex();
        result = pvertex->position().T()*CLHEP::second;
        result_z = pvertex->position().Z();
      }
    }    
  }
  if( tp_list.size() > 1 ) LogDebug("TooManyTracks") << "track matched to " << tp_list.size() << " tracking particles!" << std::endl;
  return std::make_pair(result,result_z);
}

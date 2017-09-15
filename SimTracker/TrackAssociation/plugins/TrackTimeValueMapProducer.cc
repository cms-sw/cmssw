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
#include "DataFormats/Math/interface/deltaPhi.h"

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
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"


#include <memory>

#include "SimTracker/TrackAssociation/interface/ResolutionModel.h"
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
  ~TrackTimeValueMapProducer() override { }
  
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
private:
  // inputs
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_;
  const std::string tracksName_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_;
  const edm::EDGetTokenT<TrackingVertexCollection> trackingVertices_;
  const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupSummaryInfo_;
  // tracking particle associators by order of preference
  const std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> > associators_;  
  // eta bounds
  const float etaMin_, etaMax_, ptMin_, pMin_, etaMaxForPtThreshold_;
  // options
  std::vector<std::unique_ptr<const ResolutionModel> > resolutions_;
  // functions
  float extractTrackVertexTime(const TrackingParticle&, const reco::TransientTrack&) const;
};

DEFINE_FWK_MODULE(TrackTimeValueMapProducer);

namespace {
  constexpr float fakeBeamSpotTimeWidth = 0.300f; // ns
  constexpr float m_pion = 139.57061e-3;
  const std::string resolution("Resolution");

  template<typename ParticleType, typename T>
  void writeValueMap(edm::Event &iEvent,
                     const edm::Handle<edm::View<ParticleType> > & handle,
                     const std::vector<T> & values,
                     const std::string    & label) {
    std::unique_ptr<edm::ValueMap<T> > valMap(new edm::ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(std::move(valMap), label);
  }
}

TrackTimeValueMapProducer::TrackTimeValueMapProducer(const edm::ParameterSet& conf) :
  tracks_(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("trackSrc") ) ),
  tracksName_(conf.getParameter<edm::InputTag>("trackSrc").label()),
  trackingParticles_(consumes<TrackingParticleCollection>( conf.getParameter<edm::InputTag>("trackingParticleSrc") ) ),
  trackingVertices_(consumes<TrackingVertexCollection>( conf.getParameter<edm::InputTag>("trackingVertexSrc") ) ),
  pileupSummaryInfo_(consumes<std::vector<PileupSummaryInfo> >( conf.getParameter<edm::InputTag>("pileupSummaryInfo") ) ),
  associators_( edm::vector_transform( conf.getParameter<std::vector<edm::InputTag> >("associators"), [this](const edm::InputTag& tag){ return this->consumes<reco::TrackToTrackingParticleAssociator>(tag); } ) ),
  etaMin_( conf.getParameter<double>("etaMin") ),
  etaMax_( conf.getParameter<double>("etaMax") ),
  ptMin_( conf.getParameter<double>("ptMin") ),
  pMin_( conf.getParameter<double>("pMin") ),
  etaMaxForPtThreshold_( conf.getParameter<double>("etaMaxForPtThreshold") )
{
  // setup resolution models
  const std::vector<edm::ParameterSet>& resos = conf.getParameterSetVector("resolutionModels");
  for( const auto& reso : resos ) {
    const std::string& name = reso.getParameter<std::string>("modelName");
    ResolutionModel* resomod = ResolutionModelFactory::get()->create(name,reso);
    resolutions_.emplace_back( resomod );  

    // times and time resolutions for general tracks
    produces<edm::ValueMap<float> >(tracksName_+name);
    produces<edm::ValueMap<float> >(tracksName_+name+resolution);
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

  std::vector<float> generalTrackTimes;
 
  //get track collections
  edm::Handle<edm::View<reco::Track> > TrackCollectionH;
  evt.getByToken(tracks_, TrackCollectionH);
  const edm::View<reco::Track>& TrackCollection = *TrackCollectionH;

  //get tracking particle collections
  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  evt.getByToken(trackingParticles_, TPCollectionH);
  
  edm::Handle<std::vector<PileupSummaryInfo> > pileupSummaryH;
  evt.getByToken(pileupSummaryInfo_, pileupSummaryH);
  
  //transient track builder
  edm::ESHandle<TransientTrackBuilder> theB;
  es.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
  
  // associate the reco tracks / gsf Tracks
  std::vector<reco::RecoToSimCollection> associatedTracks;  
  for( auto associator : associators ) {
    associatedTracks.emplace_back(associator->associateRecoToSim(TrackCollectionH, TPCollectionH));
  }
  

  double sumSimTime = 0.;
  double sumSimTimeSq = 0.;
  int nsim = 0;
  for (const PileupSummaryInfo &puinfo : *pileupSummaryH) {
    if (puinfo.getBunchCrossing() == 0) {
      for (const float &time : puinfo.getPU_times()) {
        double simtime = time;
        sumSimTime += simtime;
        sumSimTimeSq += simtime*simtime;
        ++nsim;
      }
      break;
    }
  }
  
  double meanSimTime = sumSimTime/double(nsim);
  double varSimTime = sumSimTimeSq/double(nsim) - meanSimTime*meanSimTime;
  double rmsSimTime = std::sqrt(std::max(0.1*0.1,varSimTime));

  for( unsigned itk = 0; itk < TrackCollection.size(); ++itk ) {
    const auto tkref = TrackCollection.refAt(itk);
    reco::RecoToSimCollection::const_iterator track_tps = associatedTracks.back().end();

    for( const auto& association : associatedTracks ) {
      track_tps = association.find(tkref);
      if( track_tps != association.end() ) break;
    }

    if (track_tps != associatedTracks.back().end() && track_tps->val.size() == 1) {
      reco::TransientTrack tt = theB->build(*tkref);
      float time = extractTrackVertexTime(*track_tps->val[0].first,tt);
      generalTrackTimes.push_back(time);
    }
    else {
      float rndtime = CLHEP::RandGauss::shoot(rng_engine, meanSimTime, rmsSimTime);
      generalTrackTimes.push_back(rndtime);
      if (track_tps != associatedTracks.back().end() && track_tps->val.size() > 1) {
        LogDebug("TooManyTracks") << "track matched to " << track_tps->val.size() << " tracking particles!" << std::endl;
      }
    }
  }
  
  for( const auto& reso : resolutions_ ) {
    const std::string& name = reso->name();
    std::vector<float> times, resos;
    
    times.reserve(TrackCollection.size());
    resos.reserve(TrackCollection.size());

    for( unsigned i = 0; i < TrackCollection.size(); ++i ) {
      const reco::Track& tk = TrackCollection[i];
      const float absEta = std::abs(tk.eta());
      bool inAcceptance = absEta < etaMax_ && absEta >= etaMin_ && tk.p()>pMin_ && (absEta>etaMaxForPtThreshold_ || tk.pt()>ptMin_);
      if (inAcceptance) {
        const float resolution = reso->getTimeResolution(tk);
        times.push_back( CLHEP::RandGauss::shoot(rng_engine, generalTrackTimes[i], resolution) );
        resos.push_back( resolution );
      }
      else {
        times.push_back(0.0f);
        resos.push_back(-1.);
      }
    }

    writeValueMap( evt, TrackCollectionH, times, tracksName_+name );
    writeValueMap( evt, TrackCollectionH, resos, tracksName_+name+resolution );
  }
}

float TrackTimeValueMapProducer::
extractTrackVertexTime( const TrackingParticle &tp, const reco::TransientTrack &tt ) const {
  int pdgid = tp.pdgId();
  const auto& tvertex = tp.parentVertex();
  math::XYZTLorentzVectorD result = tvertex->position();
  
  // account for secondary vertices...
  if( tvertex->nSourceTracks() && tvertex->sourceTracks()[0]->pdgId()==pdgid ) {
    auto pvertex = tvertex->sourceTracks()[0]->parentVertex();
    result = pvertex->position();
    while( pvertex->nSourceTracks() && pvertex->sourceTracks()[0]->pdgId()==pdgid ) {
      pvertex = pvertex->sourceTracks()[0]->parentVertex();
      result = pvertex->position();
    }
  }
  
  float time = result.T()*CLHEP::second;
  //correct for time of flight from track reference position
  GlobalPoint result_pos(result.x(),result.y(),result.z());
  const auto &tkstate = tt.trajectoryStateClosestToPoint(result_pos);
  float tkphi = tkstate.momentum().phi();
  float tkz = tkstate.position().z();
  float dphi = reco::deltaPhi(tkphi,tt.track().phi());
  float dz = tkz - tt.track().vz();
  
  float radius = 100.*tt.track().pt()/(0.3*tt.field()->inTesla(GlobalPoint(0,0,0)).z());
  float pathlengthrphi = tt.track().charge()*dphi*radius;
  
  float pathlength = std::sqrt(pathlengthrphi*pathlengthrphi + dz*dz);
  float p = tt.track().p();
  
  float speed = std::sqrt(1./(1.+m_pion/p))*CLHEP::c_light/CLHEP::cm;  //speed in cm/ns
  float dt = pathlength/speed;

  return time-dt;
}

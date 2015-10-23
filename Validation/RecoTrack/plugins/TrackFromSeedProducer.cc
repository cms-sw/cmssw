// -*- C++ -*-
//
// Package:    FastSimulation/TrackFromSeedProducer
// Class:      TrackFromSeedProducer
// 
/**\class TrackFromSeedProducer TrackFromSeedProducer.cc FastSimulation/TrackFromSeedProducer/plugins/TrackFromSeedProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Lukas Vanelderen
//         Created:  Thu, 28 May 2015 13:27:33 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// class declaration
//

class TrackFromSeedProducer : public edm::global::EDProducer<> {
public:
  explicit TrackFromSeedProducer(const edm::ParameterSet&);
  ~TrackFromSeedProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<TrajectorySeed> > seedsToken;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;
  std::string tTRHBuilderName;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
TrackFromSeedProducer::TrackFromSeedProducer(const edm::ParameterSet& iConfig)
{
  //register your products
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<int> >();

  // read parametes
  edm::InputTag seedsTag(iConfig.getParameter<edm::InputTag>("src"));
  edm::InputTag beamSpotTag(iConfig.getParameter<edm::InputTag>("beamSpot"));
  tTRHBuilderName = iConfig.getParameter<std::string>("TTRHBuilder");
  
  //consumes
  seedsToken = consumes<std::vector<TrajectorySeed> >(seedsTag);
  beamSpotToken = consumes<reco::BeamSpot>(beamSpotTag);
}


TrackFromSeedProducer::~TrackFromSeedProducer() {}

// ------------ method called to produce the data  ------------
void
TrackFromSeedProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   using namespace edm;
   using namespace reco;
   using namespace std;
   
   // output collection
   auto_ptr<TrackCollection> tracks(new TrackCollection);
   auto_ptr<TrackingRecHitCollection> rechits(new TrackingRecHitCollection);
   auto_ptr<TrackExtraCollection> trackextras(new TrackExtraCollection);
   auto_ptr<vector<int> > seedToTrack(new vector<int>());
   
   // product references 
   TrackExtraRefProd ref_trackextras = iEvent.getRefBeforePut<TrackExtraCollection>();
   TrackingRecHitRefProd ref_rechits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

   // input collection
   Handle<vector<TrajectorySeed> > seeds;
   iEvent.getByToken(seedsToken,seeds);

   // beam spot
   edm::Handle<reco::BeamSpot> beamSpot;
   iEvent.getByToken(beamSpotToken,beamSpot);

   // some objects to build to tracks
   TSCBLBuilderNoMaterial tscblBuilder;

   edm::ESHandle<TransientTrackingRecHitBuilder> tTRHBuilder;
   iSetup.get<TransientRecHitRecord>().get(tTRHBuilderName,tTRHBuilder);

   edm::ESHandle<MagneticField> theMF;
   iSetup.get<IdealMagneticFieldRecord>().get(theMF);

   edm::ESHandle<TrackerTopology> httopo;
   iSetup.get<TrackerTopologyRcd>().get(httopo);
   const TrackerTopology& ttopo = *httopo;

   // create tracks from seeds
   int nfailed  = 0;
   for (auto const & seed : *seeds.product()){
     // try to create a track
     TransientTrackingRecHit::RecHitPointer lastRecHit = tTRHBuilder->build(&*(seed.recHits().second-1));
     TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( seed.startingState(), lastRecHit->surface(), theMF.product());
     TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),*beamSpot);//as in TrackProducerAlgorithm
     if(!(tsAtClosestApproachSeed.isValid())){
       edm::LogVerbatim("SeedValidator")<<"TrajectoryStateClosestToBeamLine not valid";
       seedToTrack->push_back(-1);
       nfailed++;
       continue;
     }
     const reco::TrackBase::Point vSeed1(tsAtClosestApproachSeed.trackStateAtPCA().position().x(),
					 tsAtClosestApproachSeed.trackStateAtPCA().position().y(),
					 tsAtClosestApproachSeed.trackStateAtPCA().position().z());
     const reco::TrackBase::Vector pSeed(tsAtClosestApproachSeed.trackStateAtPCA().momentum().x(),
					 tsAtClosestApproachSeed.trackStateAtPCA().momentum().y(),
					 tsAtClosestApproachSeed.trackStateAtPCA().momentum().z());
     //GlobalPoint vSeed(vSeed1.x()-beamSpot->x0(),vSeed1.y()-beamSpot->y0(),vSeed1.z()-beamSpot->z0());
     PerigeeTrajectoryError seedPerigeeErrors = PerigeeConversions::ftsToPerigeeError(tsAtClosestApproachSeed.trackStateAtPCA());
     tracks->push_back(Track(0.,0., vSeed1, pSeed, 1, seedPerigeeErrors.covarianceMatrix()));
     seedToTrack->push_back(tracks->size()-1);
     tracks->back().appendHits(seed.recHits().first,seed.recHits().second,ttopo);
     // store the hits
     size_t firsthitindex = rechits->size();
     for(auto hitit = seed.recHits().first;hitit != seed.recHits().second;++hitit){
       rechits->push_back(*hitit);
     }
     // create a trackextra, just to store the hit range
     trackextras->push_back(TrackExtra());
     trackextras->back().setHits(ref_rechits,firsthitindex,rechits->size() - firsthitindex);
     // create link between track and trackextra
     tracks->back().setExtra( TrackExtraRef( ref_trackextras, trackextras->size() - 1) );
   }
   
   if (nfailed > 0) {
     edm::LogWarning("SeedValidator") << "failed to create tracks from " << nfailed <<  " out of " << seeds->size() << " seeds ";
   }
   iEvent.put(tracks);
   iEvent.put(seedToTrack);
   iEvent.put(rechits);
   iEvent.put(trackextras);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TrackFromSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackFromSeedProducer);

// -*- C++ -*-
//
// Package:    SeedToTrackProducer
// Class:      SeedToTrackProducer
// 
/**\class SeedToTrackProducer SeedToTrackProducer.cc hugues/SeedToTrackProducer/plugins/SeedToTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Hugues Brun
//         Created:  Tue, 05 Nov 2013 13:42:04 GMT
// $Id$
//
//




#include "SeedToTrackProducer.h"

//
// class declaration
//



//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
SeedToTrackProducer::SeedToTrackProducer(const edm::ParameterSet& iConfig)
{
    
  L2seedsTagT_ = consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("L2seedsCollection"));
  L2seedsTagS_ = consumes<edm::View<TrajectorySeed> >(iConfig.getParameter<edm::InputTag>("L2seedsCollection"));


    
    
    produces<reco::TrackCollection>();
    produces<reco::TrackExtraCollection>();
    produces<TrackingRecHitCollection>();
    
  
}


SeedToTrackProducer::~SeedToTrackProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SeedToTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace std;

    std::auto_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
    std::auto_ptr<reco::TrackExtraCollection> selectedTrackExtras( new reco::TrackExtraCollection() );
    std::auto_ptr<TrackingRecHitCollection> selectedTrackHits( new TrackingRecHitCollection() );
    
    reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
    reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
    TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
    
    edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
    edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
    
    // magnetic fied and detector geometry
    iSetup.get<IdealMagneticFieldRecord>().get(theMGField);
    iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

    // now read the L2 seeds collection :
    edm::Handle<TrajectorySeedCollection> L2seedsCollection;
    iEvent.getByToken(L2seedsTagT_,L2seedsCollection);
    const std::vector<TrajectorySeed>* L2seeds = 0;
    if (L2seedsCollection.isValid()) L2seeds = L2seedsCollection.product();
    else  edm::LogError("SeedToTrackProducer") << "L2 seeds collection not found !! " << endl;
    
    edm::Handle<edm::View<TrajectorySeed> > seedHandle;
    iEvent.getByToken(L2seedsTagS_, seedHandle);
    
    
    int countRH = 0;

    // now  loop on the seeds :
    for (unsigned int i = 0; i < L2seeds->size() ; i++){
        
        
        //get the kinematic extrapolation from the seed
        TrajectoryStateOnSurface theTrajectory = seedTransientState(L2seeds->at(i));
        float seedEta =  theTrajectory.globalMomentum().eta();
        float seedPhi =  theTrajectory.globalMomentum().phi();
        float seedPt =  theTrajectory.globalMomentum().perp();
        CovarianceMatrix matrixSeedErr = theTrajectory.curvilinearError().matrix();
        edm::LogVerbatim("SeedToTrackProducer") <<  "seedPt=" << seedPt << " seedEta=" << seedEta << " seedPhi=" << seedPhi << endl;
        /*AlgebraicSymMatrix66 errors = theTrajectory.cartesianError().matrix();
        double partialPterror = errors(3,3)*pow(theTrajectory.globalMomentum().x(),2) + errors(4,4)*pow(theTrajectory.globalMomentum().y(),2);
	edm::LogVerbatim("SeedToTrackProducer") <<  "seedPtError=" << sqrt(partialPterror)/theTrajectory.globalMomentum().perp() << "seedPhiError=" << theTrajectory.curvilinearError().matrix()(2,2) << endl;*/ 
        //fill the track in a way that its pt, phi and eta will be the same as the seed
        math::XYZPoint initPoint(0,0,0);
        math::XYZVector initMom(seedPt*cos(seedPhi),seedPt*sin(seedPhi),seedPt*sinh(seedEta));
        reco::Track theTrack(1, 1, //dummy Chi2 and ndof
                             initPoint, initMom,
                             1, matrixSeedErr,
                             reco::TrackBase::TrackAlgorithm::globalMuon, reco::TrackBase::TrackQuality::tight);

        //fill the extra track with dummy information
        math::XYZPoint dummyFinalPoint(1,1,1);
        math::XYZVector dummyFinalMom(0,0,10);
        edm::RefToBase<TrajectorySeed> seed(seedHandle, i);
        CovarianceMatrix matrixExtra = ROOT::Math::SMatrixIdentity();
        reco::TrackExtra theTrackExtra(dummyFinalPoint, dummyFinalMom, true, initPoint, initMom, true,
                                  matrixSeedErr, 1,
                                  matrixExtra, 2,
                                  (L2seeds->at(i)).direction(), seed);
        theTrack.setExtra( reco::TrackExtraRef( rTrackExtras, idx++ ) );
        edm::LogVerbatim("SeedToTrackProducer") << "trackPt=" << theTrack.pt() << " trackEta=" << theTrack.eta() << " trackPhi=" << theTrack.phi() << endl;
        edm::LogVerbatim("SeedToTrackProducer") << "trackPtError=" << theTrack.ptError() << "trackPhiError=" << theTrack.phiError() << endl;
 
        //fill the seed segments in the track
        unsigned int nHitsAdded = 0;
        for(TrajectorySeed::recHitContainer::const_iterator itRecHits=(L2seeds->at(i)).recHits().first; itRecHits!=(L2seeds->at(i)).recHits().second; ++itRecHits, ++countRH) {
            TrackingRecHit* hit = (itRecHits)->clone();
            theTrack.appendHitPattern(*hit);
            selectedTrackHits->push_back(hit);
            nHitsAdded++;
        }
        theTrackExtra.setHits( rHits, hidx, nHitsAdded );
        hidx += nHitsAdded;
        selectedTracks->push_back(theTrack);
        selectedTrackExtras->push_back(theTrackExtra);

    }
    iEvent.put(selectedTracks);
    iEvent.put(selectedTrackExtras);
    iEvent.put(selectedTrackHits);
    
}

TrajectoryStateOnSurface SeedToTrackProducer::seedTransientState(const TrajectorySeed& tmpSeed){
    
    PTrajectoryStateOnDet tmpTSOD = tmpSeed.startingState();
    DetId tmpDetId(tmpTSOD.detId());
    const GeomDet* tmpGeomDet = theTrackingGeometry->idToDet(tmpDetId);
    TrajectoryStateOnSurface tmpTSOS = trajectoryStateTransform::transientState(tmpTSOD, &(tmpGeomDet->surface()), &(*theMGField));
    return tmpTSOS;
}

// ------------ method called once each job just before starting event loop  ------------
void 
SeedToTrackProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SeedToTrackProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
SeedToTrackProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
SeedToTrackProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
SeedToTrackProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
SeedToTrackProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SeedToTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SeedToTrackProducer);

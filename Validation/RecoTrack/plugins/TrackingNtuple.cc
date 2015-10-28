// -*- C++ -*-
//
// Package:    NtupleDump/TrackingNtuple
// Class:      TrackingNtuple
//
/**\class TrackingNtuple TrackingNtuple.cc NtupleDump/TrackingNtuple/plugins/TrackingNtuple.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Tue, 25 Aug 2015 13:22:49 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoPixelVertexing/PixelTrackFitting/src/RZLine.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/QuickTrackAssociatorByHits.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"

#include "TTree.h"

/*
todo: 
add refitted hit position after track/seed fit
add original algo (needs >=CMSSW746)
add vertices
add local angle, path length!
add n 3d hits for sim tracks
*/

//
// class declaration
//

class TrackingNtuple : public edm::EDAnalyzer {
public:
  explicit TrackingNtuple(const edm::ParameterSet&);
  ~TrackingNtuple();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void clearVariables();

  //copied from QuickTrackAssociatorByHits
  static bool clusterTPAssociationListGreater(std::pair<OmniClusterRef, TrackingParticleRef> i,std::pair<OmniClusterRef, TrackingParticleRef> j) { return (i.first.rawIndex()>j.first.rawIndex()); }

  static bool intIntListGreater(std::pair<int, int> i,std::pair<int, int> j) { return (i.first>j.first); }

  // ----------member data ---------------------------
  std::vector<edm::InputTag> seedTags_;
  edm::InputTag trackTag_;
  std::string builderName_;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
  TrackAssociatorBase*  associatorByHits;
  const ParametersDefinerForTP* parametersDefiner;
  bool debug;
  const TrackerTopology* tTopo;

  TTree* t;
  //tracks
  std::vector<float> trk_px       ;
  std::vector<float> trk_py       ;
  std::vector<float> trk_pz       ;
  std::vector<float> trk_pt       ;
  std::vector<float> trk_eta      ;
  std::vector<float> trk_phi      ;
  std::vector<float> trk_dxy      ;
  std::vector<float> trk_dz       ;
  std::vector<float> trk_ptErr    ;
  std::vector<float> trk_etaErr   ;
  std::vector<float> trk_phiErr   ;
  std::vector<float> trk_dxyErr   ;
  std::vector<float> trk_dzErr    ;
  std::vector<float> trk_nChi2    ;
  std::vector<float> trk_shareFrac;
  std::vector<int> trk_q       ;
  std::vector<int> trk_nValid  ;
  std::vector<int> trk_nInvalid;
  std::vector<int> trk_nPixel  ;
  std::vector<int> trk_nStrip  ;
  std::vector<int> trk_n3DLay  ;
  std::vector<int> trk_algo    ;
  std::vector<int> trk_isHP    ;
  std::vector<int> trk_seedIdx ;
  std::vector<int> trk_simIdx  ;
  std::vector<std::vector<int> > trk_pixelIdx;
  std::vector<std::vector<int> > trk_stripIdx;
  //sim tracks
  std::vector<float> sim_px       ;
  std::vector<float> sim_py       ;
  std::vector<float> sim_pz       ;
  std::vector<float> sim_pt       ;
  std::vector<float> sim_eta      ;
  std::vector<float> sim_phi      ;
  std::vector<float> sim_dxy      ;
  std::vector<float> sim_dz       ;
  std::vector<float> sim_prodx    ;
  std::vector<float> sim_prody    ;
  std::vector<float> sim_prodz    ;
  std::vector<float> sim_shareFrac;
  std::vector<int> sim_q       ;
  std::vector<int> sim_nValid  ;
  std::vector<int> sim_nPixel  ;
  std::vector<int> sim_nStrip  ;
  std::vector<int> sim_n3DLay  ;
  std::vector<int> sim_trkIdx  ;
  std::vector<std::vector<int> > sim_pixelIdx;
  std::vector<std::vector<int> > sim_stripIdx;
  //pixels: reco and sim hits
  std::vector<int> pix_isBarrel ;
  std::vector<int> pix_lay      ;
  std::vector<int> pix_detId    ;
  std::vector<int> pix_nSimTrk  ;
  std::vector<int> pix_simTrkIdx;
  std::vector<int> pix_particle ;
  std::vector<int> pix_process  ;
  std::vector<int> pix_bunchXing;
  std::vector<int> pix_event    ;
  std::vector<float> pix_x    ;
  std::vector<float> pix_y    ;
  std::vector<float> pix_z    ;
  std::vector<float> pix_xx   ;
  std::vector<float> pix_xy   ;
  std::vector<float> pix_yy   ;
  std::vector<float> pix_yz   ;
  std::vector<float> pix_zz   ;
  std::vector<float> pix_zx   ;
  std::vector<float> pix_xsim ;
  std::vector<float> pix_ysim ;
  std::vector<float> pix_zsim ;
  std::vector<float> pix_eloss;
  std::vector<float> pix_radL ;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> pix_bbxi ;
  //strips: reco and sim hits
  std::vector<int> str_isBarrel ;
  std::vector<int> str_isStereo ;
  std::vector<int> str_det      ;
  std::vector<int> str_lay      ;
  std::vector<int> str_detId    ;
  std::vector<int> str_nSimTrk  ;
  std::vector<int> str_simTrkIdx;
  std::vector<int> str_particle ;
  std::vector<int> str_process  ;
  std::vector<int> str_bunchXing;
  std::vector<int> str_event    ;
  std::vector<float> str_x    ;
  std::vector<float> str_y    ;
  std::vector<float> str_z    ;
  std::vector<float> str_xx   ;
  std::vector<float> str_xy   ;
  std::vector<float> str_yy   ;
  std::vector<float> str_yz   ;
  std::vector<float> str_zz   ;
  std::vector<float> str_zx   ;
  std::vector<float> str_xsim ;
  std::vector<float> str_ysim ;
  std::vector<float> str_zsim ;
  std::vector<float> str_eloss;
  std::vector<float> str_radL ;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> str_bbxi ;
  //strip matched hits: reco hits
  std::vector<int> glu_isBarrel ;
  std::vector<int> glu_det      ;
  std::vector<int> glu_lay      ;
  std::vector<int> glu_detId    ;
  std::vector<int> glu_monoIdx  ;
  std::vector<int> glu_stereoIdx;
  std::vector<float> glu_x    ;
  std::vector<float> glu_y    ;
  std::vector<float> glu_z    ;
  std::vector<float> glu_xx   ;
  std::vector<float> glu_xy   ;
  std::vector<float> glu_yy   ;
  std::vector<float> glu_yz   ;
  std::vector<float> glu_zz   ;
  std::vector<float> glu_zx   ;
  std::vector<float> glu_radL ;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> glu_bbxi ;
  //beam spot
  float bsp_x;
  float bsp_y;
  float bsp_z;
  float bsp_sigmax;
  float bsp_sigmay;
  float bsp_sigmaz;
  //seeds
  std::vector<float> see_px       ;
  std::vector<float> see_py       ;
  std::vector<float> see_pz       ;
  std::vector<float> see_pt       ;
  std::vector<float> see_eta      ;
  std::vector<float> see_phi      ;
  std::vector<float> see_dxy      ;
  std::vector<float> see_dz       ;
  std::vector<float> see_ptErr    ;
  std::vector<float> see_etaErr   ;
  std::vector<float> see_phiErr   ;
  std::vector<float> see_dxyErr   ;
  std::vector<float> see_dzErr    ;
  std::vector<float> see_chi2     ;
  std::vector<int> see_q       ;
  std::vector<int> see_nValid  ;
  std::vector<int> see_nPixel  ;
  std::vector<int> see_nGlued  ;
  std::vector<int> see_nStrip  ;
  std::vector<int> see_algo    ;
  std::vector<std::vector<int> > see_pixelIdx;
  std::vector<std::vector<int> > see_gluedIdx;
  std::vector<std::vector<int> > see_stripIdx;
  //seed algo offset
  std::vector<int> algo_offset  ;

};

//
// constructors and destructor
//
TrackingNtuple::TrackingNtuple(const edm::ParameterSet& iConfig):
  seedTags_(iConfig.getUntrackedParameter<std::vector<edm::InputTag> >("seeds")),
  trackTag_(iConfig.getUntrackedParameter<edm::InputTag>("tracks")),
  builderName_(iConfig.getParameter<std::string>("TTRHBuilder")),
  debug(iConfig.getParameter<bool>("debug")) 
{
  //now do what ever initialization is needed
}


TrackingNtuple::~TrackingNtuple() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//
void TrackingNtuple::clearVariables() {

  //tracks
  trk_px       .clear();
  trk_py       .clear();
  trk_pz       .clear();
  trk_pt       .clear();
  trk_eta      .clear();
  trk_phi      .clear();
  trk_dxy      .clear();
  trk_dz       .clear();
  trk_ptErr    .clear();
  trk_etaErr   .clear();
  trk_phiErr   .clear();
  trk_dxyErr   .clear();
  trk_dzErr    .clear();
  trk_nChi2    .clear();
  trk_shareFrac.clear();
  trk_q        .clear();
  trk_nValid   .clear();
  trk_nInvalid .clear();
  trk_nPixel   .clear();
  trk_nStrip   .clear();
  trk_n3DLay   .clear();
  trk_algo     .clear();
  trk_isHP     .clear();
  trk_seedIdx  .clear();
  trk_simIdx   .clear();
  trk_pixelIdx .clear();
  trk_stripIdx .clear();
  //sim tracks
  sim_px       .clear();
  sim_py       .clear();
  sim_pz       .clear();
  sim_pt       .clear();
  sim_eta      .clear();
  sim_phi      .clear();
  sim_dxy      .clear();
  sim_dz       .clear();
  sim_prodx    .clear();
  sim_prody    .clear();
  sim_prodz    .clear();
  sim_shareFrac.clear();
  sim_q        .clear();
  sim_nValid   .clear();
  sim_nPixel   .clear();
  sim_nStrip   .clear();
  sim_n3DLay   .clear();
  sim_trkIdx   .clear();
  sim_pixelIdx .clear();
  sim_stripIdx .clear();
  //pixels
  pix_isBarrel .clear();
  pix_lay      .clear();
  pix_detId    .clear();
  pix_nSimTrk  .clear();
  pix_simTrkIdx.clear();
  pix_particle .clear();
  pix_process  .clear();
  pix_bunchXing.clear();
  pix_event    .clear();
  pix_x    .clear();
  pix_y    .clear();
  pix_z    .clear();
  pix_xx   .clear();
  pix_xy   .clear();
  pix_yy   .clear();
  pix_yz   .clear();
  pix_zz   .clear();
  pix_zx   .clear();
  pix_xsim .clear();
  pix_ysim .clear();
  pix_zsim .clear();
  pix_eloss.clear();
  pix_radL .clear();
  pix_bbxi .clear();
  //strips
  str_isBarrel .clear();
  str_isStereo .clear();
  str_det      .clear();
  str_lay      .clear();
  str_detId    .clear();
  str_nSimTrk  .clear();
  str_simTrkIdx.clear();
  str_particle .clear();
  str_process  .clear();
  str_bunchXing.clear();
  str_event    .clear();
  str_x    .clear();
  str_y    .clear();
  str_z    .clear();
  str_xx   .clear();
  str_xy   .clear();
  str_yy   .clear();
  str_yz   .clear();
  str_zz   .clear();
  str_zx   .clear();
  str_xsim .clear();
  str_ysim .clear();
  str_zsim .clear();
  str_eloss.clear();
  str_radL .clear();
  str_bbxi .clear();
  //matched hits
  glu_isBarrel .clear();
  glu_det      .clear();
  glu_lay      .clear();
  glu_detId    .clear();
  glu_monoIdx  .clear();
  glu_stereoIdx.clear();
  glu_x        .clear();
  glu_y        .clear();
  glu_z        .clear();
  glu_xx       .clear();
  glu_xy       .clear();
  glu_yy       .clear();
  glu_yz       .clear();
  glu_zz       .clear();
  glu_zx       .clear();
  glu_radL     .clear();
  glu_bbxi     .clear();
  //beamspot
  bsp_x = -9999.;
  bsp_y = -9999.;
  bsp_z = -9999.;
  bsp_sigmax = -9999.;
  bsp_sigmay = -9999.;
  bsp_sigmaz = -9999.;
  //seeds
  see_px      .clear();
  see_py      .clear();
  see_pz      .clear();
  see_pt      .clear();
  see_eta     .clear();
  see_phi     .clear();
  see_dxy     .clear();
  see_dz      .clear();
  see_ptErr   .clear();
  see_etaErr  .clear();
  see_phiErr  .clear();
  see_dxyErr  .clear();
  see_dzErr   .clear();
  see_chi2    .clear();
  see_q       .clear();
  see_nValid  .clear();
  see_nPixel  .clear();
  see_nGlued  .clear();
  see_nStrip  .clear();
  see_algo    .clear();
  see_pixelIdx.clear();
  see_gluedIdx.clear();
  see_stripIdx.clear();
  //seed algo offset
  algo_offset .clear();
  for (int i=0; i<20; ++i) algo_offset.push_back(-1);

}


// ------------ method called for each event  ------------
void TrackingNtuple::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace std;

  //initialize tree variables
  clearVariables();

  //get association maps, etc.
  Handle<TrackingParticleCollection>  TPCollectionH;
  iEvent.getByLabel("mix","MergedTrackTruth",TPCollectionH);
  TrackingParticleCollection const & tPC = *(TPCollectionH.product());
  Handle<ClusterTPAssociationProducer::ClusterTPAssociationList> pCluster2TPListH;
  iEvent.getByLabel("tpClusterProducer", pCluster2TPListH);
  ClusterTPAssociationProducer::ClusterTPAssociationList clusterToTPMap( *(pCluster2TPListH.product()) );//has to be non-const to sort
  //make sure it is properly sorted
  sort( clusterToTPMap.begin(), clusterToTPMap.end(), clusterTPAssociationListGreater );
  edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
  iEvent.getByLabel("simHitTPAssocProducer",simHitsTPAssoc);
  //make a list to link TrackingParticles to its hits in recHit collections
  //note only the first TP is saved so we ignore merged hits...
  vector<pair<int, int> > tpPixList;
  vector<pair<int, int> > tpRPhiList;
  vector<pair<int, int> > tpStereoList;

  //beamspot
  Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel("offlineBeamSpot",recoBeamSpotHandle);
  BeamSpot const & bs = *recoBeamSpotHandle;
  bsp_x = bs.x0();
  bsp_y = bs.y0();
  bsp_z = bs.x0();
  bsp_sigmax = bs.BeamWidthX();
  bsp_sigmay = bs.BeamWidthY();
  bsp_sigmaz = bs.sigmaZ();  

  //pixel hits
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  iEvent.getByLabel("siPixelRecHits", pixelHits);
  for (auto it = pixelHits->begin(); it!=pixelHits->end(); it++ ) {
    DetId hitId = it->detId();
    for (auto hit = it->begin(); hit!=it->end(); hit++ ) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder->build(&*hit);
      int firstMatchingTp = -999;
      int nMatchingTp = 0;
      GlobalPoint simHitPos = GlobalPoint(0,0,0);
      float energyLoss = -999.;
      int particleType = -999;
      int processType = -999;
      int bunchCrossing = -999;
      int event = -999;
      //get the TP that produced the hit
      pair < OmniClusterRef, TrackingParticleRef > clusterTPpairWithDummyTP( hit->firstClusterRef(), TrackingParticleRef() );
      //note: TP is dummy in clusterTPpairWithDummyTP since for clusterTPAssociationListGreater sorting only the cluster is needed
      auto range=equal_range( clusterToTPMap.begin(), clusterToTPMap.end(), clusterTPpairWithDummyTP, clusterTPAssociationListGreater );
      if( range.first != range.second ) {
	nMatchingTp = range.second-range.first;
	for( auto ip=range.first; ip != range.second; ++ip ) {
	  const TrackingParticleRef trackingParticle=(ip->second);
	  if( trackingParticle->numberOfHits() == 0 ) continue;
	  firstMatchingTp = trackingParticle.key();
	  tpPixList.push_back( make_pair<int, int>( trackingParticle.key(), hit->cluster().key() ) );
	  //now get the corresponding sim hit
	  std::pair<TrackingParticleRef, TrackPSimHitRef> simHitTPpairWithDummyTP(trackingParticle,TrackPSimHitRef());
	  //SimHit is dummy: for simHitTPAssociationListGreater sorting only the TP is needed
	  auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(),
					simHitTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
	  for(auto ip = range.first; ip != range.second; ++ip) {
	    TrackPSimHitRef TPhit = ip->second;
	    DetId dId = DetId(TPhit->detUnitId());
	    if (dId.rawId()==hitId.rawId()) {
	      simHitPos = ttrh->surface()->toGlobal(TPhit->localPosition());
	      energyLoss = TPhit->energyLoss();
	      particleType = TPhit->particleType();
	      processType = TPhit->processType();
	      bunchCrossing = TPhit->eventId().bunchCrossing();
	      event = TPhit->eventId().event();
	    }
	  }
	  break;
	}
      }
      pix_isBarrel .push_back( hitId.subdetId()==1 );
      pix_lay      .push_back( tTopo->layer(hitId) );
      pix_detId    .push_back( hitId.rawId() );
      pix_nSimTrk  .push_back( nMatchingTp );
      pix_simTrkIdx.push_back( firstMatchingTp );
      pix_particle .push_back( particleType );
      pix_process  .push_back( processType );
      pix_bunchXing.push_back( bunchCrossing );
      pix_event    .push_back( event );
      pix_x    .push_back( ttrh->globalPosition().x() );
      pix_y    .push_back( ttrh->globalPosition().y() );
      pix_z    .push_back( ttrh->globalPosition().z() );
      pix_xx   .push_back( ttrh->globalPositionError().cxx() );
      pix_xy   .push_back( ttrh->globalPositionError().cyx() );
      pix_yy   .push_back( ttrh->globalPositionError().cyy() );
      pix_yz   .push_back( ttrh->globalPositionError().czy() );
      pix_zz   .push_back( ttrh->globalPositionError().czz() );
      pix_zx   .push_back( ttrh->globalPositionError().czx() );
      pix_xsim .push_back( simHitPos.x() );
      pix_ysim .push_back( simHitPos.y() );
      pix_zsim .push_back( simHitPos.z() );
      pix_eloss.push_back( energyLoss );
      pix_radL .push_back( ttrh->surface()->mediumProperties().radLen() );
      pix_bbxi .push_back( ttrh->surface()->mediumProperties().xi() );
      if (debug) cout << "pixHit cluster=" << hit->cluster().key()
		      << " subdId=" << hitId.subdetId()
		      << " lay=" << tTopo->layer(hitId)
		      << " rawId=" << hitId.rawId()
		      << " pos =" << ttrh->globalPosition()
		      << " firstMatchingTp=" << firstMatchingTp
		      << " nMatchingTp=" << nMatchingTp
		      << " simHitPos=" << simHitPos
		      << " energyLoss=" << energyLoss
		      << " particleType=" << particleType
		      << " processType=" << processType
		      << " bunchCrossing=" << bunchCrossing
		      << " event=" << event
		      << endl;
    }
  }

  //strip hits
  //index strip hit branches by cluster index
  edm::Handle<SiStripRecHit2DCollection> rphiHits;
  iEvent.getByLabel("siStripMatchedRecHits","rphiRecHit", rphiHits);
  edm::Handle<SiStripRecHit2DCollection> stereoHits;
  iEvent.getByLabel("siStripMatchedRecHits","stereoRecHit", stereoHits);
  int totalStripHits = rphiHits->dataSize()+stereoHits->dataSize();
  str_isBarrel .resize(totalStripHits);
  str_isStereo .resize(totalStripHits);
  str_det      .resize(totalStripHits);
  str_lay      .resize(totalStripHits);
  str_detId    .resize(totalStripHits);
  str_nSimTrk  .resize(totalStripHits);
  str_simTrkIdx.resize(totalStripHits);
  str_particle .resize(totalStripHits);
  str_process  .resize(totalStripHits);
  str_bunchXing.resize(totalStripHits);
  str_event    .resize(totalStripHits);
  str_x    .resize(totalStripHits);
  str_y    .resize(totalStripHits);
  str_z    .resize(totalStripHits);
  str_xx   .resize(totalStripHits);
  str_xy   .resize(totalStripHits);
  str_yy   .resize(totalStripHits);
  str_yz   .resize(totalStripHits);
  str_zz   .resize(totalStripHits);
  str_zx   .resize(totalStripHits);
  str_xsim .resize(totalStripHits);
  str_ysim .resize(totalStripHits);
  str_zsim .resize(totalStripHits);
  str_eloss.resize(totalStripHits);
  str_radL .resize(totalStripHits);
  str_bbxi .resize(totalStripHits);
  //rphi
  for (auto it = rphiHits->begin(); it!=rphiHits->end(); it++ ) {
    DetId hitId = it->detId();
    for (auto hit = it->begin(); hit!=it->end(); hit++ ) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder->build(&*hit);
      int lay = tTopo->layer(hitId);
      int firstMatchingTp = -1;
      int nMatchingTp = 0;
      GlobalPoint simHitPos = GlobalPoint(0,0,0);
      float energyLoss = -999.;
      int particleType = -999;
      int processType = -999;
      int bunchCrossing = -999;
      int event = -999;
      pair < OmniClusterRef, TrackingParticleRef > clusterTPpairWithDummyTP( hit->firstClusterRef(), TrackingParticleRef() );
      //note: TP is dummy in clusterTPpairWithDummyTP since for clusterTPAssociationListGreater sorting only the cluster is needed
      auto range=equal_range( clusterToTPMap.begin(), clusterToTPMap.end(), clusterTPpairWithDummyTP, clusterTPAssociationListGreater );
      if( range.first != range.second ) {
	nMatchingTp = range.second-range.first;
	for( auto ip=range.first; ip != range.second; ++ip ) {
	  const TrackingParticleRef trackingParticle=(ip->second);
	  if( trackingParticle->numberOfHits() == 0 ) continue;
	  firstMatchingTp = trackingParticle.key();
	  tpRPhiList.push_back( make_pair<int, int>( trackingParticle.key(), hit->cluster().key() ) );
	  //now get the corresponding sim hit
	  std::pair<TrackingParticleRef, TrackPSimHitRef> simHitTPpairWithDummyTP(trackingParticle,TrackPSimHitRef());
	  //SimHit is dummy: for simHitTPAssociationListGreater sorting only the TP is needed
	  auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(),
					simHitTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
	  for(auto ip = range.first; ip != range.second; ++ip) {
	    TrackPSimHitRef TPhit = ip->second;
	    DetId dId = DetId(TPhit->detUnitId());
	    if (dId.rawId()==hitId.rawId()) {
	      simHitPos = ttrh->surface()->toGlobal(TPhit->localPosition());
	      energyLoss = TPhit->energyLoss();
	      particleType = TPhit->particleType();
	      processType = TPhit->processType();
	      bunchCrossing = TPhit->eventId().bunchCrossing();
	      event = TPhit->eventId().event();
	    }
	  }
	  break;
	}
      }
      int key = hit->cluster().key();
      str_isBarrel [key] = (hitId.subdetId()==StripSubdetector::TIB || hitId.subdetId()==StripSubdetector::TOB);
      str_isStereo [key] = 0;
      str_det      [key] = hitId.subdetId();
      str_lay      [key] = tTopo->layer(hitId);
      str_detId    [key] = hitId.rawId();
      str_nSimTrk  [key] = nMatchingTp;
      str_simTrkIdx[key] = firstMatchingTp;
      str_particle [key] = particleType;
      str_process  [key] = processType;
      str_bunchXing[key] = bunchCrossing;
      str_event    [key] = event;
      str_x    [key] = ttrh->globalPosition().x();
      str_y    [key] = ttrh->globalPosition().y();
      str_z    [key] = ttrh->globalPosition().z();
      str_xx   [key] = ttrh->globalPositionError().cxx();
      str_xy   [key] = ttrh->globalPositionError().cyx();
      str_yy   [key] = ttrh->globalPositionError().cyy();
      str_yz   [key] = ttrh->globalPositionError().czy();
      str_zz   [key] = ttrh->globalPositionError().czz();
      str_zx   [key] = ttrh->globalPositionError().czx();
      str_xsim [key] = simHitPos.x();
      str_ysim [key] = simHitPos.y();
      str_zsim [key] = simHitPos.z();
      str_eloss[key] = energyLoss;
      str_radL [key] = ttrh->surface()->mediumProperties().radLen();
      str_bbxi [key] = ttrh->surface()->mediumProperties().xi();
      if (debug) cout << "stripRPhiHit cluster=" << key
		      << " subdId=" << hitId.subdetId()
		      << " lay=" << lay
		      << " rawId=" << hitId.rawId()
		      << " pos =" << ttrh->globalPosition()
		      << " firstMatchingTp=" << firstMatchingTp
		      << " nMatchingTp=" << nMatchingTp
		      << " simHitPos=" << simHitPos
		      << " energyLoss=" << energyLoss
		      << " particleType=" << particleType
		      << " processType=" << processType
		      << " bunchCrossing=" << bunchCrossing
		      << " event=" << event
		      << endl;
    }
  }
  //stereo
  for (auto it = stereoHits->begin(); it!=stereoHits->end(); it++ ) {
    DetId hitId = it->detId();
    for (auto hit = it->begin(); hit!=it->end(); hit++ ) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder->build(&*hit);
      int lay = tTopo->layer(hitId);
      int firstMatchingTp = -1;
      int nMatchingTp = 0;
      GlobalPoint simHitPos = GlobalPoint(0,0,0);
      float energyLoss = -999.;
      int particleType = -999;
      int processType = -999;
      int bunchCrossing = -999;
      int event = -999;
      pair < OmniClusterRef, TrackingParticleRef > clusterTPpairWithDummyTP( hit->firstClusterRef(), TrackingParticleRef() );
      //note: TP is dummy in clusterTPpairWithDummyTP since for clusterTPAssociationListGreater sorting only the cluster is needed
      auto range=equal_range( clusterToTPMap.begin(), clusterToTPMap.end(), clusterTPpairWithDummyTP, clusterTPAssociationListGreater );
      if( range.first != range.second ) {
	nMatchingTp = range.second-range.first;
	for( auto ip=range.first; ip != range.second; ++ip ) {
	  const TrackingParticleRef trackingParticle=(ip->second);
	  if( trackingParticle->numberOfHits() == 0 ) continue;
	  firstMatchingTp = trackingParticle.key();
	  tpStereoList.push_back( make_pair<int, int>( trackingParticle.key(), hit->cluster().key() ) );
	  //now get the corresponding sim hit
	  std::pair<TrackingParticleRef, TrackPSimHitRef> simHitTPpairWithDummyTP(trackingParticle,TrackPSimHitRef());
	  //SimHit is dummy: for simHitTPAssociationListGreater sorting only the TP is needed
	  auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(),
					simHitTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
	  for(auto ip = range.first; ip != range.second; ++ip) {
	    TrackPSimHitRef TPhit = ip->second;
	    DetId dId = DetId(TPhit->detUnitId());
	    if (dId.rawId()==hitId.rawId()) {
	      simHitPos = ttrh->surface()->toGlobal(TPhit->localPosition());
	      energyLoss = TPhit->energyLoss();
	      particleType = TPhit->particleType();
	      processType = TPhit->processType();
	      bunchCrossing = TPhit->eventId().bunchCrossing();
	      event = TPhit->eventId().event();
	    }
	  }
	  break;
	}
      }
      int key = hit->cluster().key();
      str_isBarrel [key] = (hitId.subdetId()==StripSubdetector::TIB || hitId.subdetId()==StripSubdetector::TOB);
      str_isStereo [key] = 1;
      str_det      [key] = hitId.subdetId();
      str_lay      [key] = tTopo->layer(hitId);
      str_detId    [key] = hitId.rawId();
      str_nSimTrk  [key] = nMatchingTp;
      str_simTrkIdx[key] = firstMatchingTp;
      str_particle [key] = particleType;
      str_process  [key] = processType;
      str_bunchXing[key] = bunchCrossing;
      str_event    [key] = event;
      str_x    [key] = ttrh->globalPosition().x();
      str_y    [key] = ttrh->globalPosition().y();
      str_z    [key] = ttrh->globalPosition().z();
      str_xx   [key] = ttrh->globalPositionError().cxx();
      str_xy   [key] = ttrh->globalPositionError().cyx();
      str_yy   [key] = ttrh->globalPositionError().cyy();
      str_yz   [key] = ttrh->globalPositionError().czy();
      str_zz   [key] = ttrh->globalPositionError().czz();
      str_zx   [key] = ttrh->globalPositionError().czx();
      str_xsim [key] = simHitPos.x();
      str_ysim [key] = simHitPos.y();
      str_zsim [key] = simHitPos.z();
      str_eloss[key] = energyLoss;
      str_radL [key] = ttrh->surface()->mediumProperties().radLen();
      str_bbxi [key] = ttrh->surface()->mediumProperties().xi();
      if (debug) cout << "stripStereoHit cluster=" << key
		      << " subdId=" << hitId.subdetId()
		      << " lay=" << lay
		      << " rawId=" << hitId.rawId()
		      << " pos =" << ttrh->globalPosition()
		      << " firstMatchingTp=" << firstMatchingTp
		      << " nMatchingTp=" << nMatchingTp
		      << " simHitPos=" << simHitPos
		      << " energyLoss=" << energyLoss
		      << " particleType=" << particleType
		      << " processType=" << processType
		      << " bunchCrossing=" << bunchCrossing
		      << " event=" << event
		      << endl;
    }
  }

  //matched hits
  //prapare list to link matched hits to collection
  vector<pair<int,int> > monoStereoClusterList;
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
  iEvent.getByLabel("siStripMatchedRecHits","matchedRecHit", matchedHits);
  for (auto it = matchedHits->begin(); it!=matchedHits->end(); it++ ) {
    DetId hitId = it->detId();
    for (auto hit = it->begin(); hit!=it->end(); hit++ ) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder->build(&*hit);
      int lay = tTopo->layer(hitId);
      monoStereoClusterList.push_back(make_pair<int,int>(hit->monoHit().cluster().key(),hit->stereoHit().cluster().key()));
      glu_isBarrel .push_back( (hitId.subdetId()==StripSubdetector::TIB || hitId.subdetId()==StripSubdetector::TOB) );
      glu_det      .push_back( hitId.subdetId() );
      glu_lay      .push_back( tTopo->layer(hitId) );
      glu_detId    .push_back( hitId.rawId() );
      glu_monoIdx  .push_back( hit->monoHit().cluster().key() );
      glu_stereoIdx.push_back( hit->stereoHit().cluster().key() );
      glu_x        .push_back( ttrh->globalPosition().x() );
      glu_y        .push_back( ttrh->globalPosition().y() );
      glu_z        .push_back( ttrh->globalPosition().z() );
      glu_xx       .push_back( ttrh->globalPositionError().cxx() );
      glu_xy       .push_back( ttrh->globalPositionError().cyx() );
      glu_yy       .push_back( ttrh->globalPositionError().cyy() );
      glu_yz       .push_back( ttrh->globalPositionError().czy() );
      glu_zz       .push_back( ttrh->globalPositionError().czz() );
      glu_zx       .push_back( ttrh->globalPositionError().czx() );
      glu_radL     .push_back( ttrh->surface()->mediumProperties().radLen() );
      glu_bbxi     .push_back( ttrh->surface()->mediumProperties().xi() );
      if (debug) cout << "stripMatchedHit"
		      << " cluster0=" << hit->stereoHit().cluster().key()
		      << " cluster1=" << hit->monoHit().cluster().key()
		      << " subdId=" << hitId.subdetId()
		      << " lay=" << lay
		      << " rawId=" << hitId.rawId()
		      << " pos =" << ttrh->globalPosition() << endl;
    }
  }

  //seeds
  int offset = 0;
  TSCBLBuilderNoMaterial tscblBuilder;
  for (unsigned int itsd=0;itsd<seedTags_.size();++itsd) {
    InputTag seedTag_ = seedTags_[itsd];
    Handle<TrajectorySeedCollection> seeds;
    iEvent.getByLabel(seedTag_,seeds);
    TString label = seedTag_.label();
    if (debug) cout << "NEW SEED LABEL: " << label << " size: " << seeds->size();
    //format label to match algoName
    label.ReplaceAll("Seeds","");
    label.ReplaceAll("muonSeeded","muonSeededStep");
    if (debug) cout << " algo=" << TrackBase::algoByName(label.Data()) << endl;
    int algo = TrackBase::algoByName(label.Data());
    algo_offset[algo] = offset;
    int seedCount = 0;
    for(TrajectorySeedCollection::const_iterator itSeed = seeds->begin(); itSeed != seeds->end(); ++itSeed,++seedCount) {
      TransientTrackingRecHit::RecHitPointer lastRecHit = theTTRHBuilder->build(&*(itSeed->recHits().second-1));
      TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( itSeed->startingState(), lastRecHit->surface(), theMF.product());
      int charge = state.charge();
      float pt  = state.globalParameters().momentum().perp();
      float eta = state.globalParameters().momentum().eta();
      float phi = state.globalParameters().momentum().phi();
      int nHits = itSeed->nHits();
      see_px      .push_back( state.globalParameters().momentum().x() );
      see_py      .push_back( state.globalParameters().momentum().y() );
      see_pz      .push_back( state.globalParameters().momentum().z() );
      see_pt      .push_back( pt );
      see_eta     .push_back( eta );
      see_phi     .push_back( phi );
      see_q       .push_back( charge );
      see_nValid  .push_back( nHits );
      //convert seed into track to access parameters
      TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithm
      if(!(tsAtClosestApproachSeed.isValid())){
        cout<<"TrajectoryStateClosestToBeamLine for seed not valid"<<endl;;
        continue;
      }
      const reco::TrackBase::Point vSeed1(tsAtClosestApproachSeed.trackStateAtPCA().position().x(),
					  tsAtClosestApproachSeed.trackStateAtPCA().position().y(),
					  tsAtClosestApproachSeed.trackStateAtPCA().position().z());
      const reco::TrackBase::Vector pSeed(tsAtClosestApproachSeed.trackStateAtPCA().momentum().x(),
					  tsAtClosestApproachSeed.trackStateAtPCA().momentum().y(),
					  tsAtClosestApproachSeed.trackStateAtPCA().momentum().z());
      PerigeeTrajectoryError seedPerigeeErrors = PerigeeConversions::ftsToPerigeeError(tsAtClosestApproachSeed.trackStateAtPCA());
      reco::Track* matchedTrackPointer = new reco::Track(0.,0., vSeed1, pSeed, 1, seedPerigeeErrors.covarianceMatrix());
      see_dxy     .push_back(matchedTrackPointer->dxy(bs.position()));
      see_dz      .push_back(matchedTrackPointer->dz(bs.position()));
      see_ptErr   .push_back(matchedTrackPointer->ptError());
      see_etaErr  .push_back(matchedTrackPointer->etaError());
      see_phiErr  .push_back(matchedTrackPointer->phiError());
      see_dxyErr  .push_back(matchedTrackPointer->dxyError());
      see_dzErr   .push_back(matchedTrackPointer->dzError());
      see_algo    .push_back(algo);
      vector<int> pixelIdx;
      vector<int> gluedIdx;
      vector<int> stripIdx;
      for (auto hit=itSeed->recHits().first; hit!=itSeed->recHits().second; ++hit) {
	TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*hit);
	int subid = recHit->geographicalId().subdetId();
	if (subid == (int) PixelSubdetector::PixelBarrel || subid == (int) PixelSubdetector::PixelEndcap) {
	  const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*recHit);
	  pixelIdx.push_back( bhit->firstClusterRef().cluster_pixel().key() );
	} else {
	  if (trackerHitRTTI::isMatched(*recHit)) {
	    const SiStripMatchedRecHit2D * matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(&*recHit);
	    int monoIdx = matchedHit->monoClusterRef().key();
	    int stereoIdx = matchedHit->stereoClusterRef().key();
	    vector<pair<int,int> >::iterator pos = find( monoStereoClusterList.begin(), monoStereoClusterList.end(), make_pair(monoIdx,stereoIdx) );
	    gluedIdx.push_back( pos - monoStereoClusterList.begin() );
	  } else {
	    const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*recHit);
	    stripIdx.push_back( bhit->firstClusterRef().cluster_strip().key() );
	  }
	}
      }
      see_pixelIdx.push_back( pixelIdx );
      see_gluedIdx.push_back( gluedIdx );
      see_stripIdx.push_back( stripIdx );
      see_nPixel  .push_back( pixelIdx.size() );
      see_nGlued  .push_back( gluedIdx.size() );
      see_nStrip  .push_back( stripIdx.size() );
      //the part below is not strictly needed
      float chi2 = -1;
      if (nHits==2) {
	TransientTrackingRecHit::RecHitPointer recHit0 = theTTRHBuilder->build(&*(itSeed->recHits().first));
	TransientTrackingRecHit::RecHitPointer recHit1 = theTTRHBuilder->build(&*(itSeed->recHits().first+1));
	vector<GlobalPoint> gp(2);
	vector<GlobalError> ge(2);
	gp[0] = recHit0->globalPosition();
	ge[0] = recHit0->globalPositionError();
	gp[1] = recHit1->globalPosition();
	ge[1] = recHit1->globalPositionError();
	if (debug) {
	  cout << "seed " << seedCount
	       << " pt=" << pt << " eta=" << eta << " phi=" << phi << " q=" << charge
	       << " - PAIR - ids: " << recHit0->geographicalId().rawId() << " " << recHit1->geographicalId().rawId()
	       << " hitpos: " << gp[0] << " " << gp[1]
	       << " trans0: " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
	       << " " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
	       << " trans1: " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
	       << " " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
	       << " eta,phi: " << gp[0].eta() << "," << gp[0].phi()
	       << endl;
	}
      } else if (nHits==3) {
	TransientTrackingRecHit::RecHitPointer recHit0 = theTTRHBuilder->build(&*(itSeed->recHits().first));
	TransientTrackingRecHit::RecHitPointer recHit1 = theTTRHBuilder->build(&*(itSeed->recHits().first+1));
	TransientTrackingRecHit::RecHitPointer recHit2 = theTTRHBuilder->build(&*(itSeed->recHits().first+2));
	vector<GlobalPoint> gp(3);
	vector<GlobalError> ge(3);
	vector<bool> bl(3);
	gp[0] = recHit0->globalPosition();
	ge[0] = recHit0->globalPositionError();
	int subid0 = recHit0->geographicalId().subdetId();
	bl[0] = (subid0 == StripSubdetector::TIB || subid0 == StripSubdetector::TOB || subid0 == (int) PixelSubdetector::PixelBarrel);
	gp[1] = recHit1->globalPosition();
	ge[1] = recHit1->globalPositionError();
	int subid1 = recHit1->geographicalId().subdetId();
	bl[1] = (subid1 == StripSubdetector::TIB || subid1 == StripSubdetector::TOB || subid1 == (int) PixelSubdetector::PixelBarrel);
	gp[2] = recHit2->globalPosition();
	ge[2] = recHit2->globalPositionError();
	int subid2 = recHit2->geographicalId().subdetId();
	bl[2] = (subid2 == StripSubdetector::TIB || subid2 == StripSubdetector::TOB || subid2 == (int) PixelSubdetector::PixelBarrel);
	RZLine rzLine(gp,ge,bl);
	float  cottheta, intercept, covss, covii, covsi;
	rzLine.fit(cottheta, intercept, covss, covii, covsi);
	float seed_chi2 = rzLine.chi2(cottheta, intercept);
	float seed_pt = state.globalParameters().momentum().perp();
	if (debug) {
	  cout << "seed " << seedCount
	       << " pt=" << pt << " eta=" << eta << " phi=" << phi << " q=" << charge
	       << " - TRIPLET - ids: " << recHit0->geographicalId().rawId() << " " << recHit1->geographicalId().rawId() << " " << recHit2->geographicalId().rawId()
	       << " hitpos: " << gp[0] << " " << gp[1] << " " << gp[2]
	       << " trans0: " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
	       << " " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
	       << " trans1: " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
	       << " " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
	       << " trans2: " << (recHit2->transientHits().size()>1 ? recHit2->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
	       << " " << (recHit2->transientHits().size()>1 ? recHit2->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
	       << " local: " << recHit2->localPosition()
	       << " tsos pos, mom: " << state.globalPosition()<<" "<<state.globalMomentum()
	       << " eta,phi: " << gp[0].eta() << "," << gp[0].phi()
	       << " pt,chi2: " << seed_pt << "," << seed_chi2 << endl;
	}
	chi2 = seed_chi2;
      }
      see_chi2   .push_back( chi2 );
      offset++;
    }
  }

  //tracks
  edm::Handle<View<Track> > tracks;
  iEvent.getByLabel(trackTag_,tracks);
  reco::RecoToSimCollection recSimColl = associatorByHits->associateRecoToSim(tracks,TPCollectionH,&iEvent,&iSetup);
  if (debug) cout << "NEW TRACK LABEL: " << trackTag_.label() << endl;
  for(unsigned int i=0; i<tracks->size(); ++i){
    RefToBase<Track> itTrack(tracks, i);
    int nSimHits = 0;
    double sharedFraction = 0.;
    bool isSimMatched(false);
    int tpIdx = -1;
    if (recSimColl.find(itTrack) != recSimColl.end()) { 
      auto const & tp = recSimColl[itTrack];
      if (!tp.empty()) {
	nSimHits = tp[0].first->numberOfTrackerHits();
	sharedFraction = tp[0].second;
	isSimMatched = true;
	tpIdx = tp[0].first.key();
      }
    }
    int charge = itTrack->charge();
    float pt = itTrack->pt();
    float eta = itTrack->eta();
    float chi2 = itTrack->normalizedChi2();
    float phi = itTrack->phi();
    int nHits = itTrack->numberOfValidHits();
    HitPattern hp = itTrack->hitPattern();
    trk_px       .push_back(itTrack->px());
    trk_py       .push_back(itTrack->py());
    trk_pz       .push_back(itTrack->pz());
    trk_pt       .push_back(pt);
    trk_eta      .push_back(eta);
    trk_phi      .push_back(phi);
    trk_dxy      .push_back(itTrack->dxy(bs.position()));
    trk_dz       .push_back(itTrack->dz(bs.position()));
    trk_ptErr    .push_back(itTrack->ptError());
    trk_etaErr   .push_back(itTrack->etaError());
    trk_phiErr   .push_back(itTrack->phiError());
    trk_dxyErr   .push_back(itTrack->dxyError());
    trk_dzErr    .push_back(itTrack->dzError());
    trk_nChi2    .push_back( itTrack->normalizedChi2());
    trk_shareFrac.push_back(sharedFraction);
    trk_q        .push_back(charge);
    trk_nValid   .push_back(hp.numberOfValidHits());
    //trk_nInvalid .push_back(hp.numberOfLostHits(HitPattern::TRACK_HITS)); / for 80X
    trk_nInvalid .push_back(hp.numberOfLostHits());
    trk_nPixel   .push_back(hp.numberOfValidPixelHits());
    trk_nStrip   .push_back(hp.numberOfValidStripHits());
    trk_n3DLay   .push_back(hp.numberOfValidStripLayersWithMonoAndStereo()+hp.pixelLayersWithMeasurement());
    trk_algo     .push_back(itTrack->algo());
    trk_isHP     .push_back(itTrack->quality(TrackBase::highPurity));
    trk_seedIdx  .push_back( algo_offset[itTrack->algo()] + itTrack->seedRef().key() );
    trk_simIdx   .push_back(tpIdx);
    if (debug) { 
      cout << "Track #" << i << " with q=" << charge
	   << ", pT=" << pt << " GeV, eta: " << eta << ", phi: " << phi
	   << ", chi2=" << chi2
	   << ", Nhits=" << nHits
	   << ", algo=" << itTrack->algoName(itTrack->algo()).c_str()
	   << " hp=" << itTrack->quality(TrackBase::highPurity)
	   << " seed#=" << itTrack->seedRef().key()
	   << " simMatch=" << isSimMatched
	   << " nSimHits=" << nSimHits
	   << " sharedFraction=" << sharedFraction
	   << " tpIdx=" << tpIdx
	   << endl;
    }
    vector<int> pixelCluster;
    vector<int> stripCluster;
    int nhit = 0;
    for (trackingRecHit_iterator i=itTrack->recHitsBegin(); i!=itTrack->recHitsEnd(); i++){
      if (debug) cout << "hit #" << nhit;
      TransientTrackingRecHit::RecHitPointer hit=theTTRHBuilder->build(&**i );
      DetId hitId = hit->geographicalId();
      if (debug) cout << " subdet=" << hitId.subdetId();
      if(hitId.det() == DetId::Tracker) {
	if (debug) {
	  if (hitId.subdetId() == StripSubdetector::TIB )      cout << " - TIB ";
	  else if (hitId.subdetId() == StripSubdetector::TOB ) cout << " - TOB ";
	  else if (hitId.subdetId() == StripSubdetector::TEC ) cout << " - TEC ";
	  else if (hitId.subdetId() == StripSubdetector::TID ) cout << " - TID ";
	  else if (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel ) cout << " - PixBar ";
	  else if (hitId.subdetId() == (int) PixelSubdetector::PixelEndcap ) cout << " - PixFwd ";
	  else cout << " UNKNOWN TRACKER HIT TYPE ";
	  cout << tTopo->layer(hitId);
	}
	bool isPixel = (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel || hitId.subdetId() == (int) PixelSubdetector::PixelEndcap );
	if (hit->isValid()) {
	  //ugly... but works
	  const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*hit);
	  if (debug) cout << " id: " << hitId.rawId() << " - globalPos =" << hit->globalPosition()
			  << " cluster=" << (bhit->firstClusterRef().isPixel() ? bhit->firstClusterRef().cluster_pixel().key() :  bhit->firstClusterRef().cluster_strip().key())
			  << " eta,phi: " << hit->globalPosition().eta() << "," << hit->globalPosition().phi()  << endl;
	  if (isPixel) pixelCluster.push_back( bhit->firstClusterRef().cluster_pixel().key() );
	  else         stripCluster.push_back( bhit->firstClusterRef().cluster_strip().key() );
	} else  {
	  if (debug) cout << " - invalid hit" << endl;
	  if (isPixel) pixelCluster.push_back( -1 );
	  else         stripCluster.push_back( -1 );
	}
      }
      nhit++;
    }
    if (debug) cout << endl;
    trk_pixelIdx.push_back(pixelCluster);
    trk_stripIdx.push_back(stripCluster);
  }

  //tracking particles
  //sort association maps with clusters
  sort( tpPixList.begin(), tpPixList.end(), intIntListGreater );
  sort( tpRPhiList.begin(), tpRPhiList.end(), intIntListGreater );
  sort( tpStereoList.begin(), tpStereoList.end(), intIntListGreater );
  for (auto itp = tPC.begin(); itp != tPC.end(); ++itp) {
    TrackingParticleRef tp(TPCollectionH,itp-tPC.begin());
    if (debug) cout << "tracking particle pt=" << tp->pt() << " eta=" << tp->eta() << " phi=" << tp->phi() << endl;
    reco::SimToRecoCollection simRecColl = associatorByHits->associateSimToReco(tracks,TPCollectionH,&iEvent,&iSetup);
    bool isRecoMatched(false);
    int tkIdx = -1;
    float sharedFraction = -1;
    if (simRecColl.find(tp) != simRecColl.end()) { 
      auto const & tk = simRecColl[tp];
      if (!tk.empty()) {
	isRecoMatched = true;
	tkIdx = tk[0].first.key();
	sharedFraction = tk[0].second;
      }
    }
    if (debug) cout << "matched to track = " << tkIdx << " isRecoMatched=" << isRecoMatched << endl;
    sim_px       .push_back(tp->px());
    sim_py       .push_back(tp->py());
    sim_pz       .push_back(tp->pz());
    sim_pt       .push_back(tp->pt());
    sim_eta      .push_back(tp->eta());
    sim_phi      .push_back(tp->phi());
    sim_shareFrac.push_back(sharedFraction);
    sim_q        .push_back(tp->charge());
    sim_trkIdx   .push_back(tkIdx);
    sim_prodx    .push_back(tp->vertex().x());
    sim_prody    .push_back(tp->vertex().y());
    sim_prodz    .push_back(tp->vertex().z());
    //Calcualte the impact parameters w.r.t. PCA
    TrackingParticle::Vector momentum = parametersDefiner->momentum(iEvent,iSetup,tp);
    TrackingParticle::Point vertex = parametersDefiner->vertex(iEvent,iSetup,tp);
    float dxySim = (-vertex.x()*sin(momentum.phi())+vertex.y()*cos(momentum.phi()));
    float dzSim = vertex.z() - (vertex.x()*momentum.x()+vertex.y()*momentum.y())/sqrt(momentum.perp2())
      * momentum.z()/sqrt(momentum.perp2());
    sim_dxy      .push_back(dxySim);
    sim_dz       .push_back(dzSim);
    vector<int> pixelCluster;
    vector<int> stripCluster;
    pair<int, int> tpPixPairDummy(tp.key(),-1);
    auto rangePix = std::equal_range(tpPixList.begin(), tpPixList.end(), tpPixPairDummy, intIntListGreater);
    for(auto ip = rangePix.first; ip != rangePix.second; ++ip) {
      if (debug) cout << "pixHit cluster=" << ip->second << endl;
      pixelCluster.push_back(ip->second);
    }
    pair<int, int> tpRPhiPairDummy(tp.key(),-1);
    auto rangeRPhi = std::equal_range(tpRPhiList.begin(), tpRPhiList.end(), tpRPhiPairDummy, intIntListGreater);
    for(auto ip = rangeRPhi.first; ip != rangeRPhi.second; ++ip) {
      if (debug) cout << "rphiHit cluster=" << ip->second << endl;
      stripCluster.push_back(ip->second);
    }
    pair<int, int> tpStereoPairDummy(tp.key(),-1);
    auto rangeStereo = std::equal_range(tpStereoList.begin(), tpStereoList.end(), tpStereoPairDummy, intIntListGreater);
    for(auto ip = rangeStereo.first; ip != rangeStereo.second; ++ip) {
      if (debug) cout << "stereoHit cluster=" << ip->second << endl;
      stripCluster.push_back(ip->second);
    }
    sim_nValid   .push_back( pixelCluster.size()+stripCluster.size() );
    sim_nPixel   .push_back( pixelCluster.size() );
    sim_nStrip   .push_back( stripCluster.size() );
    sim_n3DLay   .push_back( -1 );//fixme
    sim_pixelIdx.push_back(pixelCluster);
    sim_stripIdx.push_back(stripCluster);
  }

  t->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void TrackingNtuple::beginJob() {
  
  edm::Service<TFileService> fs;
  fs->make<TTree>("tree","tree");
  t = fs->getObject<TTree>("tree");

  //tracks
  t->Branch("trk_px"       , &trk_px);
  t->Branch("trk_py"       , &trk_py);
  t->Branch("trk_pz"       , &trk_pz);
  t->Branch("trk_pt"       , &trk_pt);
  t->Branch("trk_eta"      , &trk_eta);
  t->Branch("trk_phi"      , &trk_phi);
  t->Branch("trk_dxy"      , &trk_dxy      );
  t->Branch("trk_dz"       , &trk_dz       );
  t->Branch("trk_ptErr"    , &trk_ptErr    );
  t->Branch("trk_etaErr"   , &trk_etaErr   );
  t->Branch("trk_phiErr"   , &trk_phiErr   );
  t->Branch("trk_dxyErr"   , &trk_dxyErr   );
  t->Branch("trk_dzErr"    , &trk_dzErr    );
  t->Branch("trk_nChi2"    , &trk_nChi2);
  t->Branch("trk_shareFrac", &trk_shareFrac);
  t->Branch("trk_q"        , &trk_q);
  t->Branch("trk_nValid"   , &trk_nValid  );
  t->Branch("trk_nInvalid" , &trk_nInvalid);
  t->Branch("trk_nPixel"   , &trk_nPixel  );
  t->Branch("trk_nStrip"   , &trk_nStrip  );
  t->Branch("trk_n3DLay"   , &trk_n3DLay  );
  t->Branch("trk_algo"     , &trk_algo    );
  t->Branch("trk_isHP"     , &trk_isHP    );
  t->Branch("trk_seedIdx"  , &trk_seedIdx );
  t->Branch("trk_simIdx"   , &trk_simIdx  );
  t->Branch("trk_pixelIdx" , &trk_pixelIdx);
  t->Branch("trk_stripIdx" , &trk_stripIdx);
  //sim tracks
  t->Branch("sim_px"       , &sim_px       );
  t->Branch("sim_py"       , &sim_py       );
  t->Branch("sim_pz"       , &sim_pz       );
  t->Branch("sim_pt"       , &sim_pt       );
  t->Branch("sim_eta"      , &sim_eta      );
  t->Branch("sim_phi"      , &sim_phi      );
  t->Branch("sim_dxy"      , &sim_dxy      );
  t->Branch("sim_dz"       , &sim_dz       );
  t->Branch("sim_prodx"    , &sim_prodx    );
  t->Branch("sim_prody"    , &sim_prody    );
  t->Branch("sim_prodz"    , &sim_prodz    );
  t->Branch("sim_shareFrac", &sim_shareFrac);
  t->Branch("sim_q"        , &sim_q        );
  t->Branch("sim_nValid"   , &sim_nValid   );
  t->Branch("sim_nPixel"   , &sim_nPixel   );
  t->Branch("sim_nStrip"   , &sim_nStrip   );
  t->Branch("sim_n3DLay"   , &sim_n3DLay   );
  t->Branch("sim_trkIdx"   , &sim_trkIdx   );
  t->Branch("sim_pixelIdx" , &sim_pixelIdx );
  t->Branch("sim_stripIdx" , &sim_stripIdx );
  //pixels
  t->Branch("pix_isBarrel"  , &pix_isBarrel );
  t->Branch("pix_lay"       , &pix_lay      );
  t->Branch("pix_detId"     , &pix_detId    );
  t->Branch("pix_nSimTrk"   , &pix_nSimTrk  );
  t->Branch("pix_simTrkIdx" , &pix_simTrkIdx);
  t->Branch("pix_particle"  , &pix_particle );
  t->Branch("pix_process"   , &pix_process  );
  t->Branch("pix_bunchXing" , &pix_bunchXing);
  t->Branch("pix_event"     , &pix_event    );
  t->Branch("pix_x"     , &pix_x    );
  t->Branch("pix_y"     , &pix_y    );
  t->Branch("pix_z"     , &pix_z    );
  t->Branch("pix_xx"    , &pix_xx   );
  t->Branch("pix_xy"    , &pix_xy   );
  t->Branch("pix_yy"    , &pix_yy   );
  t->Branch("pix_yz"    , &pix_yz   );
  t->Branch("pix_zz"    , &pix_zz   );
  t->Branch("pix_zx"    , &pix_zx   );
  t->Branch("pix_xsim"  , &pix_xsim );
  t->Branch("pix_ysim"  , &pix_ysim );
  t->Branch("pix_zsim"  , &pix_zsim );
  t->Branch("pix_eloss" , &pix_eloss);
  t->Branch("pix_radL"  , &pix_radL );
  t->Branch("pix_bbxi"  , &pix_bbxi );
  //strips
  t->Branch("str_isBarrel"  , &str_isBarrel );
  t->Branch("str_isStereo"  , &str_isStereo );
  t->Branch("str_det"       , &str_det      );
  t->Branch("str_lay"       , &str_lay      );
  t->Branch("str_detId"     , &str_detId    );
  t->Branch("str_nSimTrk"   , &str_nSimTrk  );
  t->Branch("str_simTrkIdx" , &str_simTrkIdx);
  t->Branch("str_particle"  , &str_particle );
  t->Branch("str_process"   , &str_process  );
  t->Branch("str_bunchXing" , &str_bunchXing);
  t->Branch("str_event"     , &str_event    );
  t->Branch("str_x"     , &str_x    );
  t->Branch("str_y"     , &str_y    );
  t->Branch("str_z"     , &str_z    );
  t->Branch("str_xx"    , &str_xx   );
  t->Branch("str_xy"    , &str_xy   );
  t->Branch("str_yy"    , &str_yy   );
  t->Branch("str_yz"    , &str_yz   );
  t->Branch("str_zz"    , &str_zz   );
  t->Branch("str_zx"    , &str_zx   );
  t->Branch("str_xsim"  , &str_xsim );
  t->Branch("str_ysim"  , &str_ysim );
  t->Branch("str_zsim"  , &str_zsim );
  t->Branch("str_eloss" , &str_eloss);
  t->Branch("str_radL"  , &str_radL );
  t->Branch("str_bbxi"  , &str_bbxi );
  //matched hits
  t->Branch("glu_isBarrel"  , &glu_isBarrel );
  t->Branch("glu_det"       , &glu_det      );
  t->Branch("glu_lay"       , &glu_lay      );
  t->Branch("glu_detId"     , &glu_detId    );
  t->Branch("glu_monoIdx"   , &glu_monoIdx  );
  t->Branch("glu_stereoIdx" , &glu_stereoIdx);
  t->Branch("glu_x"         , &glu_x        );
  t->Branch("glu_y"         , &glu_y        );
  t->Branch("glu_z"         , &glu_z        );
  t->Branch("glu_xx"        , &glu_xx       );
  t->Branch("glu_xy"        , &glu_xy       );
  t->Branch("glu_yy"        , &glu_yy       );
  t->Branch("glu_yz"        , &glu_yz       );
  t->Branch("glu_zz"        , &glu_zz       );
  t->Branch("glu_zx"        , &glu_zx       );
  t->Branch("glu_radL"      , &glu_radL     );
  t->Branch("glu_bbxi"      , &glu_bbxi     );
  //beam spot
  t->Branch("bsp_x" , &bsp_x , "bsp_x/F");
  t->Branch("bsp_y" , &bsp_y , "bsp_y/F");
  t->Branch("bsp_z" , &bsp_z , "bsp_z/F");
  t->Branch("bsp_sigmax" , &bsp_sigmax , "bsp_sigmax/F");
  t->Branch("bsp_sigmay" , &bsp_sigmay , "bsp_sigmay/F");
  t->Branch("bsp_sigmaz" , &bsp_sigmaz , "bsp_sigmaz/F");
  //seeds
  t->Branch("see_px"       , &see_px      );
  t->Branch("see_py"       , &see_py      );
  t->Branch("see_pz"       , &see_pz      );
  t->Branch("see_pt"       , &see_pt      );
  t->Branch("see_eta"      , &see_eta     );
  t->Branch("see_phi"      , &see_phi     );
  t->Branch("see_dxy"      , &see_dxy     );
  t->Branch("see_dz"       , &see_dz      );
  t->Branch("see_ptErr"    , &see_ptErr   );
  t->Branch("see_etaErr"   , &see_etaErr  );
  t->Branch("see_phiErr"   , &see_phiErr  );
  t->Branch("see_dxyErr"   , &see_dxyErr  );
  t->Branch("see_dzErr"    , &see_dzErr   );
  t->Branch("see_chi2"     , &see_chi2    );
  t->Branch("see_q"        , &see_q       );
  t->Branch("see_nValid"   , &see_nValid  );
  t->Branch("see_nPixel"   , &see_nPixel  );
  t->Branch("see_nGlued"   , &see_nGlued  );
  t->Branch("see_nStrip"   , &see_nStrip  );
  t->Branch("see_algo"     , &see_algo    );
  t->Branch("see_pixelIdx" , &see_pixelIdx);
  t->Branch("see_gluedIdx" , &see_gluedIdx);
  t->Branch("see_stripIdx" , &see_stripIdx);
  //seed algo offset
  t->Branch("algo_offset"  , &algo_offset );

  //t->Branch("" , &);


}

// ------------ method called once each job just after ending the event loop  ------------
void TrackingNtuple::endJob() {}

// ------------ method called when starting to processes a run  ------------
void TrackingNtuple::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {

  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  iSetup.get<TransientRecHitRecord>().get(builderName_,theTTRHBuilder);
  
  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  iSetup.get<TrackAssociatorRecord>().get("quickTrackAssociatorByHits",theHitsAssociator);
  associatorByHits = (TrackAssociatorBase *) theHitsAssociator.product();

  //get parameters definer
  edm::ESHandle<ParametersDefinerForTP> parametersDefinerH;
  iSetup.get<TrackAssociatorRecord>().get("LhcParametersDefinerForTP",parametersDefinerH);
  parametersDefiner = parametersDefinerH.product();
  
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  tTopo = tTopoHandle.product();

}

// ------------ method called when ending the processing of a run  ------------
/*
  void TrackingNtuple::endRun(edm::Run const&, edm::EventSetup const&) {}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void TrackingNtuple::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void TrackingNtuple::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackingNtuple::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackingNtuple);

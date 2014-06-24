// -*- C++ -*-
//
// Package:    UserCode/L1TriggerDPG
// Class:      L1MuonRecoTreeProducer
// 
/**\class L1MuonRecoTreeProducer L1MuonRecoTreeProducer.cc UserCode/L1TriggerDPG/src/L1MuonRecoTreeProducer.cc

 Description: Produce Muon Reco tree

 Implementation:
     
*/
//
// Original Author:  Luigi Guiducci
//         Created:  
//
//


// system include files
#include <memory>
// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

// Muons & Tracks Data Formats
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"

// Transient tracks (for extrapolations)
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

// B Field
#include "MagneticField/Engine/interface/MagneticField.h"


// Geometry
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"
#include "TMath.h"

#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisRecoMuon.h"
#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisRecoRpcHit.h"

// GP
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"


//vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <typeinfo>


// RECO TRIGGER MATCHING:
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "TString.h"
#include "TRegexp.h"
#include <utility>

//
// class declaration
//

class L1MuonRecoTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1MuonRecoTreeProducer(const edm::ParameterSet&);
  ~L1MuonRecoTreeProducer();
  TrajectoryStateOnSurface  cylExtrapTrkSam  (reco::TrackRef track, double rho);
  TrajectoryStateOnSurface  surfExtrapTrkSam (reco::TrackRef track, double z);
  void empty_global();
  void empty_tracker();
  void empty_standalone();
  double match_trigger(std::vector<int> &trigIndices, const trigger::TriggerObjectCollection &trigObjs, 
    edm::Handle<trigger::TriggerEvent>  &triggerEvent, const reco::Muon &mu);

private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void beginRun(const edm::Run &, const edm::EventSetup &);
  virtual void endRun(const edm::Run &, const edm::EventSetup &);


public:
  
 L1Analysis::L1AnalysisRecoMuon* muon;
 L1Analysis::L1AnalysisRecoMuonDataFormat * muonData;

  L1Analysis::L1AnalysisRecoRpcHit* rpcHit;
  L1Analysis::L1AnalysisRecoRpcHitDataFormat * rpcHitData;

private:
  
  unsigned maxMuon_;
  unsigned maxRpcHit_;

  //---------------------------------------------------------------------------
  // TRIGGER MATCHING 
  // member variables needed for matching, using reco muons instead of PAT
  //---------------------------------------------------------------------------
  bool triggerMatching_;
  edm::InputTag triggerSummaryLabel_;
  std::string triggerProcessLabel_;
  std::vector<std::string> isoTriggerNames_;
  std::vector<std::string> triggerNames_;

  std::vector<int> isoTriggerIndices_;
  std::vector<int> triggerIndices_;
  double triggerMaxDeltaR_;
  HLTConfigProvider hltConfig_;
  
  enum {
    GL_MUON    = 0,
    SA_MUON    = 1,
    TR_MUON    = 2,
    TRSA_MUON  = 3
  };
 
  // GP start
  edm::ESHandle<CSCGeometry> cscGeom;
  // GP end
 
  // DT Geometry
  edm::ESHandle<DTGeometry> dtGeom;

  // RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;

  // The Magnetic field
  edm::ESHandle<MagneticField> theBField;

  // The GlobalTrackingGeometry
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;


  // Extrapolator to cylinder
  edm::ESHandle<Propagator> propagatorAlong;
  edm::ESHandle<Propagator> propagatorOpposite;
  
  FreeTrajectoryState freeTrajStateMuon(reco::TrackRef track);
  
  // output file
  edm::Service<TFileService> fs_;
  
  // tree
  TTree * tree_;
  
  // EDM input tags
  edm::InputTag muonTag_;
  edm::InputTag rpcHitTag_;
  
};



L1MuonRecoTreeProducer::L1MuonRecoTreeProducer(const edm::ParameterSet& iConfig)
{
  
  maxMuon_ = iConfig.getParameter<unsigned int>("maxMuon");
  maxRpcHit_ = iConfig.getParameter<unsigned int>("maxMuon");

  muonTag_   = iConfig.getParameter<edm::InputTag>("muonTag");
  rpcHitTag_ = iConfig.getParameter<edm::InputTag>("rpcHitTag");

  muon = new L1Analysis::L1AnalysisRecoMuon();
  muonData = muon->getData();

  rpcHit = new L1Analysis::L1AnalysisRecoRpcHit();
  rpcHitData = rpcHit->getData();
   
  // set up output
  tree_=fs_->make<TTree>("MuonRecoTree", "MuonRecoTree");  
  tree_->Branch("Muon",   "L1Analysis::L1AnalysisRecoMuonDataFormat", &muonData, 32000, 3);
  tree_->Branch("RpcHit", "L1Analysis::L1AnalysisRecoRpcHitDataFormat", &rpcHitData, 32000, 3);

  //---------------------------------------------------------------------------
  // TRIGGER MATCHING 
  // member variables needed for matching, if using reco muons instead of PAT
  //---------------------------------------------------------------------------
  triggerMatching_     = iConfig.getUntrackedParameter<bool>("triggerMatching");
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerProcessLabel_ = iConfig.getUntrackedParameter<std::string>("triggerProcessLabel");
  triggerNames_        = iConfig.getParameter<std::vector<std::string> > ("triggerNames");
  isoTriggerNames_     = iConfig.getParameter<std::vector<std::string> > ("isoTriggerNames");
  triggerMaxDeltaR_    = iConfig.getParameter<double> ("triggerMaxDeltaR");

}


L1MuonRecoTreeProducer::~L1MuonRecoTreeProducer()
{
  
}


//
// member functions
//

double L1MuonRecoTreeProducer::match_trigger(
    std::vector<int> &trigIndices, const trigger::TriggerObjectCollection &trigObjs, 
    edm::Handle<trigger::TriggerEvent>  &triggerEvent, const reco::Muon &mu
    ) 
{
  double matchDeltaR = 9999;

  for(size_t iTrigIndex = 0; iTrigIndex < trigIndices.size(); ++iTrigIndex) {
    int triggerIndex = trigIndices[iTrigIndex];
    const std::vector<std::string> moduleLabels(hltConfig_.moduleLabels(triggerIndex));
    // find index of the last module:
    const unsigned moduleIndex = hltConfig_.size(triggerIndex)-2;
    // find index of HLT trigger name:
    const unsigned hltFilterIndex = triggerEvent->filterIndex( edm::InputTag ( moduleLabels[moduleIndex], "", triggerProcessLabel_ ) );

    if (hltFilterIndex < triggerEvent->sizeFilters()) {
      const trigger::Keys triggerKeys(triggerEvent->filterKeys(hltFilterIndex));
      const trigger::Vids triggerVids(triggerEvent->filterIds(hltFilterIndex));

      const unsigned nTriggers = triggerVids.size();
      for (size_t iTrig = 0; iTrig < nTriggers; ++iTrig) {
        // loop over all trigger objects:
        const trigger::TriggerObject trigObject = trigObjs[triggerKeys[iTrig]];

        double dRtmp = deltaR( mu, trigObject );

        if ( dRtmp < matchDeltaR ) {
          matchDeltaR = dRtmp;
        }

      } // loop over different trigger objects
    } // if trigger is in event (should apply hltFilter with used trigger...)
  } // loop over muon candidates

  return matchDeltaR;
}

void L1MuonRecoTreeProducer::empty_global(){
    	muonData->ch.push_back(-999999);
    	muonData->pt.push_back(-999999);
    	muonData->p.push_back(-999999);
    	muonData->eta.push_back(-999999);
    	muonData->phi.push_back(-999999);
    	muonData->normchi2.push_back(-999999);
    	muonData->validhits.push_back(-999999); 
    	muonData->numberOfMatchedStations.push_back(-999999); 
    	muonData->numberOfValidMuonHits.push_back(-999999); 
    	muonData->imp_point_x.push_back(-999999);
    	muonData->imp_point_y.push_back(-999999);
    	muonData->imp_point_z.push_back(-999999);
    	muonData->imp_point_p.push_back(-999999);
    	muonData->imp_point_pt.push_back(-999999);
    	muonData->phi_hb.push_back(-999999);
    	muonData->z_hb.push_back(-999999);
    	muonData->r_he_p.push_back(-999999);
    	muonData->phi_he_p.push_back(-999999);
    	muonData->r_he_n.push_back(-999999);
    	muonData->phi_he_n.push_back(-999999);
    	muonData->calo_energy.push_back(-999999); 
    	muonData->calo_energy3x3.push_back(-999999);	  
    	muonData->ecal_time.push_back(-999999);   
    	muonData->ecal_terr.push_back(-999999);   
    	muonData->hcal_time.push_back(-999999);   
    	muonData->hcal_terr.push_back(-999999);   
    	muonData->time_dir.push_back(-999999);
    	muonData->time_inout.push_back(-999999);
    	muonData->time_inout_err.push_back(-999999);
    	muonData->time_outin.push_back(-999999);
    	muonData->time_outin_err.push_back(-999999);

        muonData->hlt_isomu.push_back(-999999);
        muonData->hlt_mu.push_back(-999999);
        muonData->hlt_isoDeltaR.push_back(-999999);
        muonData->hlt_deltaR.push_back(-999999);
}

void L1MuonRecoTreeProducer::empty_tracker(){
	muonData->tr_ch.push_back(-999999);
	muonData->tr_pt.push_back(-999999);
	muonData->tr_p.push_back(-999999);
	muonData->tr_eta.push_back(-999999);
	muonData->tr_phi.push_back(-999999);
	muonData->tr_normchi2.push_back(-999999);
	muonData->tr_validhits.push_back(-999999);
	muonData->tr_validpixhits.push_back(-999999);
	muonData->tr_d0.push_back(-999999);
	muonData->tr_imp_point_x.push_back(-999999);
	muonData->tr_imp_point_y.push_back(-999999);
	muonData->tr_imp_point_z.push_back(-999999);
	muonData->tr_imp_point_p.push_back(-999999);
	muonData->tr_imp_point_pt.push_back(-999999);

	muonData->tr_z_mb2.push_back(-999999);
	muonData->tr_phi_mb2.push_back(-999999);
	muonData->tr_r_me2_p.push_back(-999999);
	muonData->tr_phi_me2_p.push_back(-999999);
	muonData->tr_r_me2_n.push_back(-999999);
	muonData->tr_phi_me2_n.push_back(-999999);

	muonData->tr_z_mb1.push_back(-999999);
	muonData->tr_phi_mb1.push_back(-999999);
	muonData->tr_r_me1_p.push_back(-999999);
	muonData->tr_phi_me1_p.push_back(-999999);
	muonData->tr_r_me1_n.push_back(-999999);
	muonData->tr_phi_me1_n.push_back(-999999);

}

void L1MuonRecoTreeProducer::empty_standalone(){
	muonData->sa_imp_point_x.push_back(-999999);
	muonData->sa_imp_point_y.push_back(-999999);
	muonData->sa_imp_point_z.push_back(-999999);
	muonData->sa_imp_point_p.push_back(-999999);
	muonData->sa_imp_point_pt.push_back(-999999);
	muonData->sa_z_mb2.push_back(-999999);
	muonData->sa_phi_mb2.push_back(-999999);
	muonData->sa_pseta.push_back(-999999);

	muonData->sa_z_hb.push_back(-999999);
	muonData->sa_phi_hb.push_back(-999999);	

	muonData->sa_r_he_p.push_back(-999999);
	muonData->sa_phi_he_p.push_back(-999999);
	muonData->sa_r_he_n.push_back(-999999);
	muonData->sa_phi_he_n.push_back(-999999);

	muonData->sa_normchi2.push_back(-999999);
	muonData->sa_validhits.push_back(-999999);
	muonData->sa_ch.push_back(-999999);
	muonData->sa_pt.push_back(-999999);
	muonData->sa_p.push_back(-999999);
	muonData->sa_eta.push_back(-999999);
	muonData->sa_phi.push_back(-999999);
	muonData->sa_outer_pt.push_back(-999999);
	muonData->sa_inner_pt.push_back(-999999);
	muonData->sa_outer_eta.push_back(-999999);
	muonData->sa_inner_eta.push_back(-999999);
	muonData->sa_outer_phi.push_back(-999999);
	muonData->sa_inner_phi.push_back(-999999);
	muonData->sa_outer_x.push_back(-999999);
	muonData->sa_outer_y.push_back(-999999);
	muonData->sa_outer_z.push_back(-999999);
	muonData->sa_inner_x.push_back(-999999);
	muonData->sa_inner_y.push_back(-999999);
	muonData->sa_inner_z.push_back(-999999);
	
	muonData->sa_r_me2_p.push_back(-999999);
	muonData->sa_phi_me2_p.push_back(-999999);

	muonData->sa_r_me2_n.push_back(-999999);
	muonData->sa_phi_me2_n.push_back(-999999);

	muonData->sa_z_mb1.push_back(-999999);
	muonData->sa_phi_mb1.push_back(-999999);
	muonData->sa_r_me1_p.push_back(-999999);
	muonData->sa_phi_me1_p.push_back(-999999);
	muonData->sa_r_me1_n.push_back(-999999);
	muonData->sa_phi_me1_n.push_back(-999999);

	muonData->sa_nChambers.push_back(-999);
	muonData->sa_nMatches.push_back(-999);

}


//
// ------------ method called to for each event  ------------
//
void
L1MuonRecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  float pig=TMath::Pi();  

  muon->Reset();
  rpcHit->Reset();

  //GP start
  // Get the CSC Geometry 
  iSetup.get<MuonGeometryRecord>().get(cscGeom);
  //GP end 

  // Get the DT Geometry from the setup 
  iSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the RPC Geometry from the setup 
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);

  //Get the Magnetic field from the setup
  iSetup.get<IdealMagneticFieldRecord>().get(theBField);

  // Get the GlobalTrackingGeometry from the setup
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  edm::Handle<RPCRecHitCollection> rpcRecHits;

  iEvent.getByLabel(rpcHitTag_,rpcRecHits);
  if (!rpcRecHits.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find RPCRecHitCollection with label " << rpcHitTag_.label() ;
    }
  else {

    RPCRecHitCollection::const_iterator recHitIt  = rpcRecHits->begin();
    RPCRecHitCollection::const_iterator recHitEnd = rpcRecHits->end();

    int iRpcRecHits=0;

    for(; recHitIt != recHitEnd; ++recHitIt){ 

      if((unsigned int) iRpcRecHits>maxRpcHit_-1) continue;

      int cls        = recHitIt->clusterSize();
      int firststrip = recHitIt->firstClusterStrip();
      int bx         = recHitIt->BunchX();

      RPCDetId rpcId = recHitIt->rpcId();
      int region    = rpcId.region();
      int stat      = rpcId.station();
      int sect      = rpcId.sector();
      int layer     = rpcId.layer();
      int subsector = rpcId.subsector();
      int roll      = rpcId.roll();
      int ring      = rpcId.ring();

      LocalPoint recHitPosLoc       = recHitIt->localPosition();
      const BoundPlane & RPCSurface = rpcGeom->roll(rpcId)->surface();
      GlobalPoint recHitPosGlob     = RPCSurface.toGlobal(recHitPosLoc);

      float xLoc    = recHitPosLoc.x();
      float phiGlob = recHitPosGlob.phi();
      
      rpcHitData->region.push_back(region);
      rpcHitData->clusterSize.push_back(cls);
      rpcHitData->strip.push_back(firststrip);
      rpcHitData->bx.push_back(bx);
      rpcHitData->xLoc.push_back(xLoc);
      rpcHitData->phiGlob.push_back(phiGlob);
      rpcHitData->station.push_back(stat);
      rpcHitData->sector.push_back(sect);
      rpcHitData->layer.push_back(layer);
      rpcHitData->subsector.push_back(subsector);
      rpcHitData->roll.push_back(roll);
      rpcHitData->ring.push_back(ring);
      rpcHitData->muonId.push_back(-999); // CB set to invalid now, updated when looking at muon info 
      
      iRpcRecHits++;
      
    }
    
    rpcHitData->nRpcHits = iRpcRecHits;
    
  }
  
  // Get the muon candidates
  edm::Handle<reco::MuonCollection> mucand;
  iEvent.getByLabel(muonTag_,mucand);
  if (!mucand.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find Muon Collection with label " << muonTag_.label() ;
    return;
  }
  
  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpot;       
  iEvent.getByLabel("offlineBeamSpot", beamSpot);

  // Get the primary vertices
  edm::Handle<std::vector<reco::Vertex> > vertex;
  iEvent.getByLabel(edm::InputTag("offlinePrimaryVertices"), vertex);
    
  // Get the propagators 
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAny", propagatorAlong   );
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyOpposite", propagatorOpposite);


  for(reco::MuonCollection::const_iterator imu = mucand->begin(); 
      // for(pat::MuonCollection::const_iterator imu = mucand->begin(); 
      imu != mucand->end() && (unsigned) muonData->nMuons < maxMuon_; imu++) {

    int type=0;
    if (imu->isGlobalMuon()) type=type+1;
    if (imu->isStandAloneMuon()) type=type+2;
    if (imu->isTrackerMuon()) type=type+4;
    if (imu->isCaloMuon()) type=type+8;

    bool isTIGHT = (vertex->size() > 0                                                &&
		    imu->isGlobalMuon() && imu->globalTrack()->normalizedChi2() < 10. &&
		    imu->globalTrack()->hitPattern().numberOfValidMuonHits() > 0      &&
		    imu->numberOfMatchedStations() > 1                                && 
		    fabs(imu->innerTrack()->dxy(vertex->at(0).position())) < 0.2      &&
		    fabs(imu->innerTrack()->dz(vertex->at(0).position())) < 0.5       &&
		    imu->innerTrack()->hitPattern().numberOfValidPixelHits() > 0      &&
		    imu->innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5);

    if (isTIGHT) type=type+16;
    if (imu->isPFMuon()) type=type+32;

    muonData->howmanytypes.push_back(type);   // muon type is counting calo muons and multiple assignments

    // bool type identifiers are exclusive. is this correct? CB to check 
    bool isSA = (!imu->isGlobalMuon() && imu->isStandAloneMuon() && !imu->isTrackerMuon());
    bool isTR = (!imu->isGlobalMuon() && imu->isTrackerMuon() && !imu->isStandAloneMuon());
    bool isGL = (imu->isGlobalMuon());//&&!(imu->isStandAloneMuon())&&!(imu->isTrackerMuon()));
    bool isTRSA  = (!imu->isGlobalMuon() && imu->isStandAloneMuon()&&imu->isTrackerMuon());	    

    
    // How we fill this. We have 3 blocks of variables: muons_; muons_sa_; muons_tr_
    //   GL   SA   TR      Description               muon_type     Bool id   Filled Vars               Variables to -99999 
    //   0    0    0       Muon does not exist         /             /         /                    	     /                
    //   0    0    1       It is a Tracker muon      TR_MUON       isTR	     muons_tr_               	muons_sa_,muons_    
    //   0    1    0       It is a SA muon           SA_MUON       isSA      muons_sa                   muons_tr,muons_         
    //   0    1    1       It is a SA+Tracker muon   TRSA_MUON     isTRSA    muons_sa,muons_tr          muons_                  
    //   1    0    0       It is a Global only muon  GL_MUON       isGL	     muons_,muons_sa,muons_tr	     /                
    //   1    0    1       Gl w/out SA cannot exist    /            /	        /                    	     /                
    //   1    1    0       Gl+SA (no trk-mu match)   GL_MUON       isGL	     muons_,muons_sa,muons_tr	  none                
    //   1    1    1       GL+SA+Tr (all matched)    GL_MUON       isGL	     muons_,muons_sa,muons_tr	  none
    

    //---------------------------------------------------------------------
    // TRIGGER MATCHING:
    // if specified the reconstructed muons are matched to a trigger
    //---------------------------------------------------------------------
    if (triggerMatching_) {
        double isoMatchDeltaR = 9999.;
        double matchDeltaR = 9999.;
        int hasIsoTriggered = 0;
        int hasTriggered = 0;

	    // first check if the trigger results are valid:
    	edm::Handle<edm::TriggerResults> triggerResults;
		iEvent.getByLabel(edm::InputTag("TriggerResults", "", triggerProcessLabel_), triggerResults);
		  
		if (triggerResults.isValid()) {
		  edm::Handle<trigger::TriggerEvent> triggerEvent;
		  iEvent.getByLabel(triggerSummaryLabel_, triggerEvent);
		  if (triggerEvent.isValid()) {
			// get trigger objects:
			const trigger::TriggerObjectCollection triggerObjects = triggerEvent->getObjects();

			matchDeltaR = match_trigger(triggerIndices_, triggerObjects, triggerEvent, (*imu));
			if (matchDeltaR < triggerMaxDeltaR_)
				hasTriggered = 1;

			isoMatchDeltaR = match_trigger(isoTriggerIndices_, triggerObjects, triggerEvent, (*imu));
			if (isoMatchDeltaR < triggerMaxDeltaR_) 
				hasIsoTriggered = 1;
		  } // end if (triggerEvent.isValid())
		} // end if (triggerResults.isValid())

        // fill trigger matching variables:
        muonData->hlt_isomu.push_back(hasIsoTriggered);
        muonData->hlt_mu.push_back(hasTriggered);
        muonData->hlt_isoDeltaR.push_back(isoMatchDeltaR);
        muonData->hlt_deltaR.push_back(matchDeltaR);
    } // end if (triggerMatching_)

    if (isGL || isTR || isSA || isTRSA){
      muonData->nMuons = muonData->nMuons + 1;
      if (isTR)      muonData->type.push_back(TR_MUON);
      if (isGL)      muonData->type.push_back(GL_MUON);
      if (isSA)      muonData->type.push_back(SA_MUON);
      if (isTRSA)    muonData->type.push_back(TRSA_MUON);
      if (!isGL) empty_global();
      if (!isTR && !isGL && !isTRSA) empty_tracker();
      if (!isSA && !isGL && !isTRSA) empty_standalone();

      // begin GP
      //---------------------------------------------------------------------
      // RECHIT information in CSC: only for standalone/global muons!
      //---------------------------------------------------------------------
      // An artificial rank to sort RecHits 
      // the closer RecHit to key station/layer -> the smaller the rank
      // KK's original idea
      const int lutCSC[4][6] = { {26,24,22,21,23,25} ,
				 { 6, 4, 2, 1, 3, 5} ,
				 {16,14,12,11,13,15} ,
				 {36,34,32,31,33,35} };

      int   globalTypeRCH = -999999;
      // float  localEtaRCH = -999999;
      // float  localPhiRCH = -999999;
      float globalEtaRCH = -999999;
      float globalPhiRCH = -999999;
      
      if(isSA || isGL){
	
	trackingRecHit_iterator hit     = imu->outerTrack()->recHitsBegin();
	trackingRecHit_iterator hitEnd  = imu->outerTrack()->recHitsEnd();
	
	for(; hit != hitEnd; ++hit) {

	  if ( !((*hit)->isValid()) ) continue;

	  // Hardware ID of the RecHit (in terms of wire/strips/chambers)
	  DetId detid = (*hit)->geographicalId();

	  // Interested in muon systems only
	  // Look only at CSC Hits (CSC id is 2)
	  if ( detid.det() != DetId::Muon ) continue;
	  if (detid.subdetId() != MuonSubdetId::CSC) continue;

	  CSCDetId id(detid.rawId());
          //std::cout << "before Lut id.station = " <<id.station() 
          //          << " id.layer() = " << id.layer() << " globalTypeRCH = " 
          //          << globalTypeRCH  << std::endl; 
	  // another sanity check
	  if  (id.station() < 1) continue;

	  // Look up some stuff specific to CSCRecHit2D
          // std::cout << " typeid().name() " << typeid(**hit).name() << std::endl; 
	  // const CSCRecHit2D* CSChit =dynamic_cast<const CSCRecHit2D*>(&**hit);

	  const CSCSegment* cscSegment =dynamic_cast<const CSCSegment*>(&**hit);
          //std::cout << "cscSegment = " << cscSegment << std::endl;
	  if (cscSegment == NULL) continue;
          // const CSCRecHit2D* CSChit =(CSCRecHit2D*)(&**hit);

	  // std::cout << " after CSCRecHit2D, CSChit = "  << CSChit << std::endl; 
	  // LocalPoint rhitlocal = CSChit->localPosition();
	  LocalPoint rhitlocal = cscSegment->localPosition();

	  // for debugging purpouses
	  //if (printLevel > 0) {
	  //std::cout << "!!!!rhitlocal.phi = "<<rhitlocal.phi() << std::endl;
          //std::cout << "rhitlocal.y="<<rhitlocal.y();
          //std::cout << "rhitlocal.z="<<rhitlocal.z();
	  //}

	  GlobalPoint gp = GlobalPoint(0.0, 0.0, 0.0);

	  const CSCChamber* cscchamber = cscGeom->chamber(id);

	  if (!cscchamber) continue;

	  gp = cscchamber->toGlobal(rhitlocal);

	  // identify the rechit position
	  //int pos = ( ((id.station()-1)*6) + (id.layer()-1) ) + (MAX_CSC_RECHIT*whichMuon);

	  // --------------------------------------------------
	  // this part has to be deprecated once we are sure of 
	  // the TMatrixF usage ;)
	  // fill the rechits array
	  //rchEtaList[pos]      = gp.eta();
	  //rchPhiList[pos]      = gp.phi();
	  //float phi02PI = gp.phi();
	  //if (gp.phi() < 0) phi02PI += (2*PI); 
	  
	  //rchPhiList_02PI[pos] = phi02PI;
	  // --------------------------------------------------

	  // --------------------------------------------------
	  // See if this hit is closer to the "key" position
	  //if( lutCSC[id.station()-1][id.layer()-1]<globalTypeRCH || globalTypeRCH<0 ){
	  if( lutCSC[id.station()-1][3-1]<globalTypeRCH || globalTypeRCH<0 ){
	    //globalTypeRCH    = lutCSC[id.station()-1][id.layer()-1];
	    globalTypeRCH    = lutCSC[id.station()-1][3-1];
	    // localEtaRCH      = rhitlocal.eta();
	    // localPhiRCH      = rhitlocal.phi();
	    globalEtaRCH     = gp.eta();
	    globalPhiRCH     = gp.phi();// phi from -pi to pi
	    //std::cout << "globalEtaRCH =  "  <<globalEtaRCH 
            //          << " globalPhiRCH = " << globalPhiRCH << std::endl; 
	    if(globalPhiRCH < 0) globalPhiRCH = globalPhiRCH + 2*pig;// convert to [0; 2pi]
	  }
	  // -------------------------------------------------- 
	}

	hit = imu->outerTrack()->recHitsBegin();

	for(; hit != hitEnd; hit++) {

	  if ( !((*hit)->isValid()) ) continue;
	  
	  DetId detId = (*hit)->geographicalId();
	    
	  if (detId.det() != DetId::Muon ) continue;
	  if (detId.subdetId() != MuonSubdetId::RPC) continue;

	  RPCDetId rpcId = (RPCDetId)(*hit)->geographicalId();
	 
	  int region    = rpcId.region();
	  int stat      = rpcId.station();
	  int sect      = rpcId.sector();
	  int layer     = rpcId.layer();
	  int subsector = rpcId.subsector();
	  int roll      = rpcId.roll();
	  int ring      = rpcId.ring();

	  float xLoc = (*hit)->localPosition().x();
 
	  for (int iRpcHit = 0; iRpcHit < rpcHitData->nRpcHits; ++ iRpcHit) {

	    if ( region    == rpcHitData->region.at(iRpcHit)     &&
		 stat      == rpcHitData->station.at(iRpcHit)    &&
		 sect      == rpcHitData->sector.at(iRpcHit)     &&
		 layer     == rpcHitData->layer.at(iRpcHit)      &&
		 subsector == rpcHitData->subsector.at(iRpcHit)  &&
		 roll      == rpcHitData->roll.at(iRpcHit)       &&
		 ring      == rpcHitData->ring.at(iRpcHit)       &&
		 fabs(xLoc - rpcHitData->xLoc.at(iRpcHit)) < 0.01
		 )
		 
	      rpcHitData->muonId.at(iRpcHit) = muonData->nMuons; // CB rpc hit belongs to mu imu
	                                            // due to cleaning as in 2012 1 rpc hit belogns to 1 mu

	  }

	}

      }
      // at the end of the loop, write only the best rechit
      // if(globalPhiRCH > -100.) std::cout << "globalPhiRCH = " << globalPhiRCH 
      // << "globalEtaRCH = " << globalEtaRCH << std::endl;
      muonData -> rchCSCtype.push_back(globalTypeRCH);
      muonData -> rchPhi.push_back(globalPhiRCH);
      muonData -> rchEta.push_back(globalEtaRCH);
      // end GP
	
      if (isGL){
	// Filling calo energies and calo times
	if (imu->isEnergyValid()){
	  reco::MuonEnergy muon_energy;
	  muon_energy = imu->calEnergy();
	  muonData->calo_energy.push_back(muon_energy.tower);
	  muonData->calo_energy3x3.push_back(muon_energy.towerS9);
	  muonData->ecal_time.push_back(muon_energy.ecal_time);
	  muonData->ecal_terr.push_back(muon_energy.ecal_timeError);
	  muonData->hcal_time.push_back(muon_energy.hcal_time);
	  muonData->hcal_terr.push_back(muon_energy.hcal_timeError);
	} else {
	  muonData->calo_energy.push_back(-999999);	
	  muonData->calo_energy3x3.push_back(-999999);	
	  muonData->ecal_time.push_back(-999999);	
	  muonData->ecal_terr.push_back(-999999);	
	  muonData->hcal_time.push_back(-999999);	
	  muonData->hcal_terr.push_back(-999999);	
	}

	// Filling muon time data
	if (imu->isTimeValid()){
	  reco::MuonTime muon_time;
	  muon_time = imu->time();
	  muonData->time_dir.push_back(muon_time.direction()); 
	  muonData->time_inout.push_back(muon_time.timeAtIpInOut); 
	  muonData->time_inout_err.push_back(muon_time.timeAtIpInOutErr); 
	  muonData->time_outin.push_back(muon_time.timeAtIpOutIn); 
	  muonData->time_outin_err.push_back(muon_time.timeAtIpOutInErr); 
	} else {
	  muonData->time_dir.push_back(-999999);
	  muonData->time_inout.push_back(-999999);
	  muonData->time_inout_err.push_back(-999999);
	  muonData->time_outin.push_back(-999999);
	  muonData->time_outin_err.push_back(-999999);
	}

	// Use the track now, and make a transient track out of it (for extrapolations)
	reco::TrackRef glb_mu = imu->globalTrack();  
	reco::TransientTrack ttrack(*glb_mu,&*theBField,theTrackingGeometry);

	// Track quantities
	muonData->ch.push_back(glb_mu->charge());
	muonData->pt.push_back(glb_mu->pt());
	muonData->p.push_back(glb_mu->p());
	muonData->eta.push_back(glb_mu->eta());
	muonData->phi.push_back(glb_mu->phi());
	muonData->normchi2.push_back(glb_mu->normalizedChi2());
	muonData->validhits.push_back(glb_mu->numberOfValidHits()); 
	muonData->numberOfMatchedStations.push_back(imu->numberOfMatchedStations());
	muonData->numberOfValidMuonHits.push_back(glb_mu->hitPattern().numberOfValidMuonHits());

	// Extrapolation to IP
	if (ttrack.impactPointTSCP().isValid()){
	  muonData->imp_point_x.push_back(ttrack.impactPointTSCP().position().x());
	  muonData->imp_point_y.push_back(ttrack.impactPointTSCP().position().y());
	  muonData->imp_point_z.push_back(ttrack.impactPointTSCP().position().z());
	  muonData->imp_point_p.push_back(sqrt(ttrack.impactPointTSCP().position().mag()));
	  muonData->imp_point_pt.push_back(sqrt(ttrack.impactPointTSCP().position().perp2()));
	}else{
	  muonData->imp_point_x.push_back(-999999);
	  muonData->imp_point_y.push_back(-999999);
	  muonData->imp_point_z.push_back(-999999);
	  muonData->imp_point_p.push_back(-999999);
	  muonData->imp_point_pt.push_back(-999999);
	}

	// Extrapolation to HB
	TrajectoryStateOnSurface tsos;
	tsos = cylExtrapTrkSam(glb_mu, 235);
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double zz = tsos.globalPosition().z();
	  muonData->z_hb.push_back(zz);	
	  double rr = sqrt(xx*xx + yy*yy);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->phi_hb.push_back(acos(cosphi));
	  else
	    muonData->phi_hb.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->phi_hb.push_back(-999999);
	  muonData->z_hb.push_back(-999999);
	}

	// Extrapolation to HE+
	tsos = surfExtrapTrkSam(glb_mu, 479);
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->r_he_p.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->phi_he_p.push_back(acos(cosphi));
	  else
	    muonData->phi_he_p.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->r_he_p.push_back(-999999);
	  muonData->phi_he_p.push_back(-999999);
	}

	// Extrapolation to HE-
	tsos = surfExtrapTrkSam(glb_mu, -479);
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->r_he_n.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->phi_he_n.push_back(acos(cosphi));
	  else
	    muonData->phi_he_n.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->r_he_n.push_back(-999999);
	  muonData->phi_he_n.push_back(-999999);
	}

      } // end of IF IS GLOBAL

	// If global muon or tracker muon, fill the tracker track quantities
      if (isTR || isGL || isTRSA){

	// Take the tracker track and build a transient track out of it
	reco::TrackRef tr_mu = imu->innerTrack();  
	reco::TransientTrack ttrack(*tr_mu,&*theBField,theTrackingGeometry);
	// Fill track quantities
	muonData->tr_ch.push_back(tr_mu->charge());
	muonData->tr_pt.push_back(tr_mu->pt());
	muonData->tr_p.push_back(tr_mu->p());
	muonData->tr_eta.push_back(tr_mu->eta());
	muonData->tr_phi.push_back(tr_mu->phi());
	muonData->tr_normchi2.push_back(tr_mu->normalizedChi2());
	muonData->tr_validhits.push_back(tr_mu->numberOfValidHits()); 
	muonData->tr_validpixhits.push_back(tr_mu->hitPattern().numberOfValidPixelHits()); 
        // find d0 from vertex position  
        //////////// Beam spot //////////////
        edm::Handle<reco::BeamSpot> beamSpot;
        iEvent.getByLabel("offlineBeamSpot", beamSpot);
	muonData->tr_d0.push_back(tr_mu->dxy(beamSpot->position()));
       	
	
	// Extrapolation to the IP
	if (ttrack.impactPointTSCP().isValid()){
	  muonData->tr_imp_point_x.push_back(ttrack.impactPointTSCP().position().x());
	  muonData->tr_imp_point_y.push_back(ttrack.impactPointTSCP().position().y());
	  muonData->tr_imp_point_z.push_back(ttrack.impactPointTSCP().position().z());
	  muonData->tr_imp_point_p.push_back(sqrt(ttrack.impactPointTSCP().position().mag()));
	  muonData->tr_imp_point_pt.push_back(sqrt(ttrack.impactPointTSCP().position().perp2()));
	}
	else{
	  muonData->tr_imp_point_x.push_back(-999999);
	  muonData->tr_imp_point_y.push_back(-999999);
	  muonData->tr_imp_point_z.push_back(-999999);
	  muonData->tr_imp_point_p.push_back(-999999);
	  muonData->tr_imp_point_pt.push_back(-999999);
	}

	TrajectoryStateOnSurface tsos;
	tsos = cylExtrapTrkSam(tr_mu, 410);  // track at MB1 radius - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double zz = tsos.globalPosition().z();
	  muonData->tr_z_mb1.push_back(zz);	
	  double rr = sqrt(xx*xx + yy*yy);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->tr_phi_mb1.push_back(acos(cosphi));
	  else
	    muonData->tr_phi_mb1.push_back(2*pig-acos(cosphi));
	}
	else{
	  muonData->tr_z_mb1.push_back(-999999);
	  muonData->tr_phi_mb1.push_back(-999999);
	}

	tsos = cylExtrapTrkSam(tr_mu, 500);  // track at MB2 radius - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double zz = tsos.globalPosition().z();
	  muonData->tr_z_mb2.push_back(zz);	
	  double rr = sqrt(xx*xx + yy*yy);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->tr_phi_mb2.push_back(acos(cosphi));
	  else
	    muonData->tr_phi_mb2.push_back(2*pig-acos(cosphi));
	}
	else{
	  muonData->tr_z_mb2.push_back(-999999);
	  muonData->tr_phi_mb2.push_back(-999999);
	}
	
	tsos = surfExtrapTrkSam(tr_mu, 630);   // track at ME1+ plane - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->tr_r_me1_p.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->tr_phi_me1_p.push_back(acos(cosphi));
	  else
	    muonData->tr_phi_me1_p.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->tr_r_me1_p.push_back(-999999);
	  muonData->tr_phi_me1_p.push_back(-999999);
	}

	tsos = surfExtrapTrkSam(tr_mu, 790);   // track at ME2+ plane - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->tr_r_me2_p.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->tr_phi_me2_p.push_back(acos(cosphi));
	  else
	    muonData->tr_phi_me2_p.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->tr_r_me2_p.push_back(-999999);
	  muonData->tr_phi_me2_p.push_back(-999999);
	}

	tsos = surfExtrapTrkSam(tr_mu, -630);   // track at ME1- plane - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->tr_r_me1_n.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->tr_phi_me1_n.push_back(acos(cosphi));
	  else
	    muonData->tr_phi_me1_n.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->tr_r_me1_n.push_back(-999999);
	  muonData->tr_phi_me1_n.push_back(-999999);
	}

	tsos = surfExtrapTrkSam(tr_mu, -790);   // track at ME2- plane - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->tr_r_me2_n.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->tr_phi_me2_n.push_back(acos(cosphi));
	  else
	    muonData->tr_phi_me2_n.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->tr_r_me2_n.push_back(-999999);
	  muonData->tr_phi_me2_n.push_back(-999999);
	}


      } // end of IF IS TRACKER



	// If global muon or sa muon, fill the sa track quantities
      if (isGL || isSA || isTRSA){
	muonData->sa_nChambers.push_back(imu->numberOfChambers());
	muonData->sa_nMatches.push_back(imu->numberOfMatches());

	
	// Take the SA track and build a transient track out of it
	reco::TrackRef sa_mu = imu->outerTrack();  
	reco::TransientTrack ttrack(*sa_mu,&*theBField,theTrackingGeometry);

	// Extrapolation to IP
	if (ttrack.impactPointTSCP().isValid()){
	  muonData->sa_imp_point_x.push_back(ttrack.impactPointTSCP().position().x());
	  muonData->sa_imp_point_y.push_back(ttrack.impactPointTSCP().position().y());
	  muonData->sa_imp_point_z.push_back(ttrack.impactPointTSCP().position().z());
	  muonData->sa_imp_point_p.push_back(sqrt(ttrack.impactPointTSCP().position().mag()));
	  muonData->sa_imp_point_pt.push_back(sqrt(ttrack.impactPointTSCP().position().perp2()));
	} else {
	  muonData->sa_imp_point_x.push_back(-999999);
	  muonData->sa_imp_point_y.push_back(-999999);
	  muonData->sa_imp_point_z.push_back(-999999);
	  muonData->sa_imp_point_p.push_back(-999999);
	  muonData->sa_imp_point_pt.push_back(-999999);
	}

	// Extrapolation to MB2


	TrajectoryStateOnSurface tsos;
	tsos = cylExtrapTrkSam(sa_mu, 410);  // track at MB1 radius - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double zz = tsos.globalPosition().z();
	  muonData->sa_z_mb1.push_back(zz);	
	  double rr = sqrt(xx*xx + yy*yy);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_mb1.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_mb1.push_back(2*pig-acos(cosphi));
	}
	else{
	  muonData->sa_z_mb1.push_back(-999999);
	  muonData->sa_phi_mb1.push_back(-999999);
	}

	tsos = cylExtrapTrkSam(sa_mu, 500);  // track at MB2 radius - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double zz = tsos.globalPosition().z();
	  muonData->sa_z_mb2.push_back(zz);	
	  double rr = sqrt(xx*xx + yy*yy);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_mb2.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_mb2.push_back(2*pig-acos(cosphi));
	  double abspseta = -log( tan( atan(fabs(rr/zz))/2.0 ) );
	  if (zz>=0)
	    muonData->sa_pseta.push_back(abspseta);
	  else
	    muonData->sa_pseta.push_back(-abspseta);
	}
	else{
	  muonData->sa_z_mb2.push_back(-999999);
	  muonData->sa_phi_mb2.push_back(-999999);
	  muonData->sa_pseta.push_back(-999999);
	}
	
	tsos = surfExtrapTrkSam(sa_mu, 630);   // track at ME1+ plane - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->sa_r_me1_p.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_me1_p.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_me1_p.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->sa_r_me1_p.push_back(-999999);
	  muonData->sa_phi_me1_p.push_back(-999999);
	}

	// Extrapolation to ME2+
	tsos = surfExtrapTrkSam(sa_mu, 790);
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->sa_r_me2_p.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_me2_p.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_me2_p.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->sa_r_me2_p.push_back(-999999);
	  muonData->sa_phi_me2_p.push_back(-999999);
	}
	
	tsos = surfExtrapTrkSam(sa_mu, -630);   // track at ME1- plane - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->sa_r_me1_n.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_me1_n.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_me1_n.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->sa_r_me1_n.push_back(-999999);
	  muonData->sa_phi_me1_n.push_back(-999999);
	}

	// Extrapolation to ME2-
	tsos = surfExtrapTrkSam(sa_mu, -790); // track at ME2- disk - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->sa_r_me2_n.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_me2_n.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_me2_n.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->sa_r_me2_n.push_back(-999999);
	  muonData->sa_phi_me2_n.push_back(-999999);
	}

	// Extrapolation to HB
	tsos = cylExtrapTrkSam(sa_mu, 235);  // track at HB radius - extrapolation
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double zz = tsos.globalPosition().z();
	  muonData->sa_z_hb.push_back(zz);	
	  double rr = sqrt(xx*xx + yy*yy);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_hb.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_hb.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->sa_z_hb.push_back(-999999);
	  muonData->sa_phi_hb.push_back(-999999);	
	}

	// Extrapolation to HE+
	tsos = surfExtrapTrkSam(sa_mu, 479);
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->sa_r_he_p.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_he_p.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_he_p.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->sa_r_he_p.push_back(-999999);
	  muonData->sa_phi_he_p.push_back(-999999);
	}

	// Extrapolation to HE-
	tsos = surfExtrapTrkSam(sa_mu, -479);
	if (tsos.isValid()) {
	  double xx = tsos.globalPosition().x();
	  double yy = tsos.globalPosition().y();
	  double rr = sqrt(xx*xx + yy*yy);
	  muonData->sa_r_he_n.push_back(rr);
	  double cosphi = xx/rr;
	  if (yy>=0) 
	    muonData->sa_phi_he_n.push_back(acos(cosphi));
	  else
	    muonData->sa_phi_he_n.push_back(2*pig-acos(cosphi));	
	}
	else{
	  muonData->sa_r_he_n.push_back(-999999);
	  muonData->sa_phi_he_n.push_back(-999999);
	}

	//  Several track quantities
	muonData->sa_normchi2.push_back(sa_mu->normalizedChi2());
	muonData->sa_validhits.push_back(sa_mu->numberOfValidHits()); 
	muonData->sa_ch.push_back(sa_mu->charge());    
	muonData->sa_pt.push_back(sa_mu->pt());
	muonData->sa_p.push_back(sa_mu->p());
	muonData->sa_eta.push_back(sa_mu->eta());
	muonData->sa_phi.push_back(sa_mu->phi());
	muonData->sa_outer_pt.push_back( sqrt(sa_mu->outerMomentum().Perp2()));
	muonData->sa_inner_pt.push_back( sqrt(sa_mu->innerMomentum().Perp2()));
	muonData->sa_outer_eta.push_back(sa_mu->outerMomentum().Eta());
	muonData->sa_inner_eta.push_back(sa_mu->innerMomentum().Eta());
	muonData->sa_outer_phi.push_back(sa_mu->outerMomentum().Phi());
	muonData->sa_inner_phi.push_back(sa_mu->innerMomentum().Phi());
	muonData->sa_outer_x.push_back(sa_mu->outerPosition().x());
	muonData->sa_outer_y.push_back(sa_mu->outerPosition().y());
	muonData->sa_outer_z.push_back(sa_mu->outerPosition().z());
	muonData->sa_inner_x.push_back(sa_mu->innerPosition().x());
	muonData->sa_inner_y.push_back(sa_mu->innerPosition().y());
	muonData->sa_inner_z.push_back(sa_mu->innerPosition().z());

      } // end of IF IS STANDALONE

    }    // end of if type 012 found
  } // end of muon loop

  tree_->Fill();
    
}

// to get the track position info at a particular rho
TrajectoryStateOnSurface 
L1MuonRecoTreeProducer::cylExtrapTrkSam(reco::TrackRef track, double rho)
{
  Cylinder::PositionType pos(0, 0, 0);
  Cylinder::RotationType rot;
  Cylinder::CylinderPointer myCylinder = Cylinder::build(pos, rot, rho);

  FreeTrajectoryState recoStart = freeTrajStateMuon(track);
  TrajectoryStateOnSurface recoProp;
  recoProp = propagatorAlong->propagate(recoStart, *myCylinder);
  if (!recoProp.isValid()) {
    recoProp = propagatorOpposite->propagate(recoStart, *myCylinder);
  }
  return recoProp;
}

// to get track position at a particular (xy) plane given its z
TrajectoryStateOnSurface
L1MuonRecoTreeProducer::surfExtrapTrkSam(reco::TrackRef track, double z)
{
  Plane::PositionType pos(0, 0, z);
  Plane::RotationType rot;
  Plane::PlanePointer myPlane = Plane::build(pos, rot);

  FreeTrajectoryState recoStart = freeTrajStateMuon(track);
  TrajectoryStateOnSurface recoProp;
  recoProp = propagatorAlong->propagate(recoStart, *myPlane);
  if (!recoProp.isValid()) {
    recoProp = propagatorOpposite->propagate(recoStart, *myPlane);
  }
  return recoProp;
}



FreeTrajectoryState L1MuonRecoTreeProducer::freeTrajStateMuon(reco::TrackRef track)
{
  GlobalPoint  innerPoint(track->innerPosition().x(), track->innerPosition().y(),  track->innerPosition().z());
  GlobalVector innerVec  (track->innerMomentum().x(),  track->innerMomentum().y(),  track->innerMomentum().z());  
  
  FreeTrajectoryState recoStart(innerPoint, innerVec, track->charge(), &*theBField);
  
  return recoStart;
}

void L1MuonRecoTreeProducer::beginRun(const edm::Run &run, const edm::EventSetup &eventSetup) {
  // Prepare for trigger matching for each new run: 
  // Look up triggetIndices in the HLT config for the different paths
  if (triggerMatching_) {
    bool changed = true;
    if (!hltConfig_.init(run, eventSetup, triggerProcessLabel_, changed)) {
      // if you can't initialize hlt configuration, crash!
      std::cout << "Error: didn't find process" << triggerProcessLabel_ << std::endl;
      assert(false);
    }

    bool enableWildcard = true;
    for (size_t iTrig = 0; iTrig < triggerNames_.size(); ++iTrig) { 
      // prepare for regular expression (with wildcards) functionality:
      TString tNameTmp = TString(triggerNames_[iTrig]);
      TRegexp tNamePattern = TRegexp(tNameTmp, enableWildcard);
      int tIndex = -1;
      // find the trigger index:
      for (unsigned ipath = 0; ipath < hltConfig_.size(); ++ipath) {
        // use TString since it provides reg exp functionality:
        TString tmpName = TString(hltConfig_.triggerName(ipath));
        if (tmpName.Contains(tNamePattern)) {
          tIndex = int(ipath);
          triggerIndices_.push_back(tIndex);
        }
      }
      if (tIndex < 0) { // if can't find trigger path at all, give warning:
        std::cout << "Warning: Could not find trigger" << triggerNames_[iTrig] << std::endl;
        //assert(false);
      }
    } // end for triggerNames
    for (size_t iTrig = 0; iTrig < isoTriggerNames_.size(); ++iTrig) { 
      // prepare for regular expression functionality:
      TString tNameTmp = TString(isoTriggerNames_[iTrig]);
      TRegexp tNamePattern = TRegexp(tNameTmp, enableWildcard);
      int tIndex = -1;
      // find the trigger index:
      for (unsigned ipath = 0; ipath < hltConfig_.size(); ++ipath) {
        // use TString since it provides reg exp functionality:
        TString tmpName = TString(hltConfig_.triggerName(ipath));
        if (tmpName.Contains(tNamePattern)) {
          tIndex = int(ipath);
          isoTriggerIndices_.push_back(tIndex);
        }
      }
      if (tIndex < 0) { // if can't find trigger path at all, give warning:
        std::cout << "Warning: Could not find trigger" << isoTriggerNames_[iTrig] << std::endl;
        //assert(false);
      }
    } // end for isoTriggerNames
  } // end if (triggerMatching_)
}

void L1MuonRecoTreeProducer::endRun(const edm::Run &run, const edm::EventSetup &eventSetup) {
}
// ------------ method called once each job just before starting event loop  ------------
void 
L1MuonRecoTreeProducer::beginJob(void){
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1MuonRecoTreeProducer::endJob() 
{
 delete muon;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1MuonRecoTreeProducer);

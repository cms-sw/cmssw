// -*- C++ -*-
//
// Package:    MuonME0RecHits
// Class:      MuonME0RecHits
// 
/**\class MuonME0RecHits MuonME0RecHits.cc Validation/MuonME0RecHits/plugins/MuonME0RecHits.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author: Claudio Caputo, INFN Bari
//         Created:  Sun, 23 Mar 2014 17:28:48 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Data Format
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"
#include <DataFormats/GEMRecHit/interface/ME0Segment.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//
// class declaration
//
struct MyME0RecHit
{
    Int_t detId, particleType;
    Float_t x, y, xErr, yErr;
    Float_t xExt, yExt;
    Int_t region, ring, station, layer, chamber, roll;
    Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
    Int_t bx, clusterSize, firstClusterStrip;
    Float_t x_sim, y_sim;
    Float_t globalEta_sim, globalPhi_sim, globalX_sim, globalY_sim, globalZ_sim;
    Float_t pull;
};


struct MyME0SimHit
{
    Int_t eventNumber;
    Int_t detUnitId, particleType;
    Float_t x, y, energyLoss, pabs, timeOfFlight;
    Int_t region, ring, station, layer, chamber, roll;
    Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
    Int_t strip;
    Float_t Phi_0, DeltaPhi, R_0;
    Int_t countMatching;
};

struct MyME0Segment
{
    Int_t detId;
    Float_t localX, localY, localZ;
    Float_t dirTheta, dirPhi;
    Int_t numberRH, ndof;
    Float_t chi2;
};

struct MySimTrack
{
    Float_t pt, eta, phi;
    Char_t charge;
    Char_t endcap;
    Char_t gem_sh_layer1, gem_sh_layer2;
    Char_t gem_rh_layer1, gem_rh_layer2;
    Float_t gem_sh_eta, gem_sh_phi;
    Float_t gem_sh_x, gem_sh_y;
    Float_t gem_rh_eta, gem_rh_phi;
    Float_t gem_lx_even, gem_ly_even;
    Float_t gem_lx_odd, gem_ly_odd;
    Char_t has_gem_sh_l1, has_gem_sh_l2;
    Char_t has_gem_rh_l1, has_gem_rh_l2;
    Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho;
};





class MuonME0RecHits : public edm::EDAnalyzer {
   public:
      explicit MuonME0RecHits(const edm::ParameterSet&);
      ~MuonME0RecHits();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
    
      bool isSimTrackGood(const SimTrack &);
      bool isME0RecHitMatched(MyME0RecHit me0_recHit_, MyME0SimHit me0_sh);

      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
    bool debug_;
    std::string folderPath_;
    bool EffSaveRootFile_;
    std::string EffRootFileName_;
    
    DQMStore * dbe;
    std::map<std::string, MonitorElement*> meCollection;
    
    edm::Handle<ME0RecHitCollection> me0RecHits_;
    edm::Handle<ME0SegmentCollection> me0Segment_;
    edm::Handle<edm::PSimHitContainer> ME0Hits;
    edm::Handle<edm::SimTrackContainer> sim_tracks;
    edm::Handle<edm::SimVertexContainer> sim_vertices;
    edm::ESHandle<ME0Geometry> me0_geom;
    
    const ME0Geometry* me0_geometry_;
    
    MyME0SimHit me0_sh;
    MyME0RecHit me0_rh;
    MyME0Segment me0_seg;
    MyME0RecHit me0_rhFromSeg;
    MySimTrack track_;
    
    edm::ParameterSet cfg_;
    
    edm::InputTag simTrackInput_;
    edm::InputTag me0SimHitInput_;
    edm::InputTag me0RecHitInput_;
    edm::InputTag me0SegInput_;
    
    double simTrackMinPt_;
    double simTrackMaxPt_;
    double simTrackMinEta_;
    double simTrackMaxEta_;
    double simTrackOnlyMuon_;
    
    bool hasME0Geometry_;
    
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
MuonME0RecHits::MuonME0RecHits(const edm::ParameterSet& iConfig):
debug_(iConfig.getUntrackedParameter<bool>("debug")),
folderPath_(iConfig.getUntrackedParameter<std::string>("folderPath")),
EffSaveRootFile_(iConfig.getUntrackedParameter<bool>("EffSaveRootFile")),
EffRootFileName_(iConfig.getUntrackedParameter<std::string>("EffRootFileName"))
{
    
    dbe = edm::Service<DQMStore>().operator->();    
    
    cfg_ = iConfig.getParameter<edm::ParameterSet>("simTrackMatching");
    auto simTrack = cfg_.getParameter<edm::ParameterSet>("simTrack");
    simTrackInput_ = simTrack.getParameter<edm::InputTag>("input");
    simTrackMinPt_ = simTrack.getParameter<double>("minPt");
    simTrackMaxPt_ = simTrack.getParameter<double>("maxPt");
    simTrackMinEta_ = simTrack.getParameter<double>("minEta");
    simTrackMaxEta_ = simTrack.getParameter<double>("maxEta");
    simTrackOnlyMuon_ = simTrack.getParameter<bool>("onlyMuon");
    
    auto me0SimHit = cfg_.getParameter<edm::ParameterSet>("me0SimHit");
    me0SimHitInput_ = me0SimHit.getParameter<edm::InputTag>("input");
    
    auto me0RecHit = cfg_.getParameter<edm::ParameterSet>("me0RecHit");
    me0RecHitInput_ = me0RecHit.getParameter<edm::InputTag>("input");
    
    auto me0Seg = cfg_.getParameter<edm::ParameterSet>("me0Seg");
    me0SegInput_ = me0Seg.getParameter<edm::InputTag>("input");
    
    hasME0Geometry_=false;

}


MuonME0RecHits::~MuonME0RecHits()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
MuonME0RecHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
    iEvent.getByLabel(simTrackInput_, sim_tracks);
    iEvent.getByLabel(simTrackInput_, sim_vertices);
    iEvent.getByLabel(me0SimHitInput_, ME0Hits);
    iEvent.getByLabel(me0RecHitInput_, me0RecHits_);
    iEvent.getByLabel(me0SegInput_, me0Segment_);
    
    std::vector<int> trackIds;
    std::vector<int> trackType;
    const edm::SimTrackContainer & sim_trks = *sim_tracks.product();
    
    for (auto& t: sim_trks)
    {
        if (!isSimTrackGood(t)) continue;
        trackType.push_back(t.type());
        trackIds.push_back(t.trackId());
    }
    
    
    for (edm::PSimHitContainer::const_iterator itHit = ME0Hits->begin(); itHit != ME0Hits->end(); ++itHit)
    {
        if(abs(itHit->particleType()) != 13) continue;
        if(std::find(trackIds.begin(), trackIds.end(), itHit->trackId()) == trackIds.end()) continue;
        
        //std::cout<<"Size "<<trackIds.size()<<" id1 "<<trackIds[0]<<" type1 "<<trackType[0]<<" id2 "<<trackIds[1]<<" type2 "<<trackType[1]<<std::endl;
        
        me0_sh.eventNumber = iEvent.id().event();
        me0_sh.detUnitId = itHit->detUnitId();
        me0_sh.particleType = itHit->particleType();
        me0_sh.x = itHit->localPosition().x();
        me0_sh.y = itHit->localPosition().y();
        me0_sh.energyLoss = itHit->energyLoss();
        me0_sh.pabs = itHit->pabs();
        me0_sh.timeOfFlight = itHit->timeOfFlight();
        
        const ME0DetId id(itHit->detUnitId());
        
        me0_sh.region = id.region();
        me0_sh.ring = 0;
        me0_sh.station = 0;
        me0_sh.layer = id.layer();
        me0_sh.chamber = id.chamber();
        me0_sh.roll = id.roll();
        
        const LocalPoint p0(0., 0., 0.);
        const GlobalPoint Gp0(me0_geometry_->idToDet(itHit->detUnitId())->surface().toGlobal(p0));
        
        me0_sh.Phi_0 = Gp0.phi();
        me0_sh.R_0 = Gp0.perp();
        me0_sh.DeltaPhi = atan(-1*id.region()*pow(-1,id.chamber())*itHit->localPosition().x()/(Gp0.perp() + itHit->localPosition().y()));
        
        const LocalPoint hitLP(itHit->localPosition());
        const GlobalPoint hitGP(me0_geometry_->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP));
        me0_sh.globalR = hitGP.perp();
        me0_sh.globalEta = hitGP.eta();
        me0_sh.globalPhi = hitGP.phi();
        me0_sh.globalX = hitGP.x();
        me0_sh.globalY = hitGP.y();
        me0_sh.globalZ = hitGP.z();
        
        //  Now filling strip info using entry point rather than local position to be
        //  consistent with digi strips. To change back, just switch the comments - WHF
        //  me0_sh.strip=gem_geometry_->etaPartition(itHit->detUnitId())->strip(hitLP);
        const LocalPoint hitEP(itHit->entryPoint());
        me0_sh.strip = me0_geometry_->etaPartition(itHit->detUnitId())->strip(hitEP);
        
        int count = 0;
        //std::cout<<"SimHit: region "<<me0_sh.region<<" station "<<me0_sh.station<<" layer "<<me0_sh.layer<<" chamber "<<me0_sh.chamber<<" roll "<<me0_sh.roll<<" strip "<<me0_sh.strip<<" type "<<itHit->particleType()<<" id "<<itHit->trackId()<<" x: "<<me0_sh.x<<std::endl;
        
        for (ME0RecHitCollection::const_iterator recHit = me0RecHits_->begin(); recHit != me0RecHits_->end(); ++recHit)
        {
            
            me0_rh.x = recHit->localPosition().x();
            me0_rh.xErr = recHit->localPositionError().xx();
            me0_rh.y = recHit->localPosition().y();
            me0_rh.detId = (Short_t) (*recHit).me0Id();
            me0_rh.bx = 0;
            me0_rh.clusterSize = 0;
            me0_rh.firstClusterStrip = 0;
            
            ME0DetId id((*recHit).me0Id());
            
            me0_rh.region = (Short_t) id.region();
            me0_rh.ring = 0;
            me0_rh.station = 0;
            me0_rh.layer = (Short_t) id.layer();
            me0_rh.chamber = (Short_t) id.chamber();
            me0_rh.roll = (Short_t) id.roll();
            
            LocalPoint rhLP = recHit->localPosition();
            GlobalPoint rhGP = me0_geometry_->idToDet((*recHit).me0Id())->surface().toGlobal(rhLP);
            
            me0_rh.globalR = rhGP.perp();
            me0_rh.globalEta = rhGP.eta();
            me0_rh.globalPhi = rhGP.phi();
            me0_rh.globalX = rhGP.x();
            me0_rh.globalY = rhGP.y();
            me0_rh.globalZ = rhGP.z();
            
            me0_rh.x_sim = me0_sh.x;
            me0_rh.y_sim = me0_sh.y;
            me0_rh.globalEta_sim = me0_sh.globalEta;
            me0_rh.globalPhi_sim = me0_sh.globalPhi;
            me0_rh.globalX_sim = me0_sh.globalX;
            me0_rh.globalY_sim = me0_sh.globalY;
            me0_rh.globalZ_sim = me0_sh.globalZ;
            me0_rh.pull = (me0_sh.x - me0_rh.x) / sqrt(me0_rh.xErr);
            
            // abbreviations
            int re(me0_rh.region);
            int la(me0_rh.layer);
            
            if(me0_rh.bx != 0) continue;
            if(isME0RecHitMatched(me0_rh, me0_sh))
            {
                bool verbose(false);
                if (verbose){
                    std::cout<<"SimHit: region "<<me0_sh.region<<" station "<<me0_sh.station
                    <<" layer "<<me0_sh.layer<<" chamber "<<me0_sh.chamber<<" roll "
                    <<me0_sh.roll<<" strip "<<me0_sh.strip<<" type "<<itHit->particleType()
                    <<" id "<<itHit->trackId()<<" x: "<<me0_sh.x<<std::endl;
                    std::cout<<"RecHit: region "<<me0_rh.region<<" station "<<me0_rh.station
                    <<" layer "<<me0_rh.layer<<" chamber "<<me0_rh.chamber
                    <<" roll "<<me0_rh.roll<<" firstStrip "<<me0_rh.firstClusterStrip
                    <<" cls "<<me0_rh.clusterSize<<" bx "<<me0_rh.bx<<" x: "<<me0_rh.x
                    <<" sigma: "<<me0_rh.xErr<<std::endl;
                }
                
                /*---------- (x_sim - x_rec) -----------*/
                
                meCollection["recHitDX"]->Fill(me0_rh.x_sim-me0_rh.x);
                
                if(re==-1 && la==1) meCollection["recHitDX_rm1_l1"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==-1 && la==2) meCollection["recHitDX_rm1_l2"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==-1 && la==3) meCollection["recHitDX_rm1_l3"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==-1 && la==4) meCollection["recHitDX_rm1_l4"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==-1 && la==5) meCollection["recHitDX_rm1_l5"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==-1 && la==6) meCollection["recHitDX_rm1_l6"]->Fill(me0_rh.x_sim-me0_rh.x);
                
                if(re==1 && la==1) meCollection["recHitDX_rp1_l1"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==1 && la==2) meCollection["recHitDX_rp1_l2"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==1 && la==3) meCollection["recHitDX_rp1_l3"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==1 && la==4) meCollection["recHitDX_rp1_l4"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==1 && la==5) meCollection["recHitDX_rp1_l5"]->Fill(me0_rh.x_sim-me0_rh.x);
                if(re==1 && la==6) meCollection["recHitDX_rp1_l6"]->Fill(me0_rh.x_sim-me0_rh.x);
                
                /*---------- Pull -------------*/
                meCollection["recHitPullLocalX"]->Fill(me0_rh.pull);
                
                if(re==-1 && la==1) meCollection["recHitPullLocalX_rm1_l1"]->Fill(me0_rh.pull);
                if(re==-1 && la==2) meCollection["recHitPullLocalX_rm1_l2"]->Fill(me0_rh.pull);
                if(re==-1 && la==3) meCollection["recHitPullLocalX_rm1_l3"]->Fill(me0_rh.pull);
                if(re==-1 && la==4) meCollection["recHitPullLocalX_rm1_l4"]->Fill(me0_rh.pull);
                if(re==-1 && la==5) meCollection["recHitPullLocalX_rm1_l5"]->Fill(me0_rh.pull);
                if(re==-1 && la==6) meCollection["recHitPullLocalX_rm1_l6"]->Fill(me0_rh.pull);
                
                if(re==1 && la==1) meCollection["recHitPullLocalX_rp1_l1"]->Fill(me0_rh.pull);
                if(re==1 && la==2) meCollection["recHitPullLocalX_rp1_l2"]->Fill(me0_rh.pull);
                if(re==1 && la==3) meCollection["recHitPullLocalX_rp1_l3"]->Fill(me0_rh.pull);
                if(re==1 && la==4) meCollection["recHitPullLocalX_rp1_l4"]->Fill(me0_rh.pull);
                if(re==1 && la==5) meCollection["recHitPullLocalX_rp1_l5"]->Fill(me0_rh.pull);
                if(re==1 && la==6) meCollection["recHitPullLocalX_rp1_l6"]->Fill(me0_rh.pull);
                
                /*---------- Delta Phi ---------*/
                meCollection["recHitDPhi"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                
                if(re==-1 && la==1) meCollection["recHitDPhi_rm1_l1"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==-1 && la==2) meCollection["recHitDPhi_rm1_l2"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==-1 && la==3) meCollection["recHitDPhi_rm1_l3"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==-1 && la==4) meCollection["recHitDPhi_rm1_l4"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==-1 && la==5) meCollection["recHitDPhi_rm1_l5"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==-1 && la==6) meCollection["recHitDPhi_rm1_l6"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                
                if(re==1 && la==1) meCollection["recHitDPhi_rp1_l1"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==1 && la==2) meCollection["recHitDPhi_rp1_l2"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==1 && la==3) meCollection["recHitDPhi_rp1_l3"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==1 && la==4) meCollection["recHitDPhi_rp1_l4"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==1 && la==5) meCollection["recHitDPhi_rp1_l5"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                if(re==1 && la==6) meCollection["recHitDPhi_rp1_l6"]->Fill(me0_rh.globalPhi-me0_rh.globalPhi_sim);
                
                /*----------- xy Occupancy --------*/
                if(re==-1 && la==1) meCollection["localrh_xy_rm1_l1"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==-1 && la==2) meCollection["localrh_xy_rm1_l2"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==-1 && la==3) meCollection["localrh_xy_rm1_l3"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==-1 && la==4) meCollection["localrh_xy_rm1_l4"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==-1 && la==5) meCollection["localrh_xy_rm1_l5"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==-1 && la==6) meCollection["localrh_xy_rm1_l6"]->Fill(me0_rh.globalX,me0_rh.globalY);
                
                if(re==1 && la==1) meCollection["localrh_xy_rp1_l1"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==1 && la==2) meCollection["localrh_xy_rp1_l2"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==1 && la==3) meCollection["localrh_xy_rp1_l3"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==1 && la==4) meCollection["localrh_xy_rp1_l4"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==1 && la==5) meCollection["localrh_xy_rp1_l5"]->Fill(me0_rh.globalX,me0_rh.globalY);
                if(re==1 && la==6) meCollection["localrh_xy_rp1_l6"]->Fill(me0_rh.globalX,me0_rh.globalY);
                
                /*---------- zR Occupancy --------*/
                const double glb_R(sqrt(me0_rh.globalX*me0_rh.globalX+me0_rh.globalY*me0_rh.globalY));
                if(re==-1) meCollection["localrh_zr_rm1"]->Fill(me0_rh.globalZ,glb_R);
                if(re==1)  meCollection["localrh_zr_rp1"]->Fill(me0_rh.globalZ,glb_R);
                
                count++;
            }
        }
        me0_sh.countMatching = count;
    }
    
    
    for (auto me0s = me0Segment_->begin(); me0s != me0Segment_->end(); me0s++) {
        
        // The ME0 Ensamble DetId refers to layer = 1
        ME0DetId id = me0s->me0DetId();
        //std::cout <<" Original ME0DetID "<<id<<std::endl;
        auto roll = me0_geometry_->etaPartition(id);
        //std::cout <<"Global Segment Position "<< roll->toGlobal(me0s->localPosition())<<std::endl;
        auto segLP = me0s->localPosition();
        auto segLD = me0s->localDirection();
        //std::cout <<" Global Direction theta = "<<segLD.theta()<<" phi="<<segLD.phi()<<std::endl;
        auto me0rhs = me0s->specificRecHits();
        //std::cout <<"ME0 Ensamble Det Id "<<id<<" Number of RecHits "<<me0rhs.size()<<std::endl;
        
        me0_seg.detId = id;
        me0_seg.localX = segLP.x();
        me0_seg.localY = segLP.y();
        me0_seg.localZ = segLP.z();
        me0_seg.dirTheta = segLD.theta();
        me0_seg.dirPhi = segLD.phi();
        me0_seg.numberRH = me0rhs.size();
        me0_seg.chi2 = me0s->chi2();
        me0_seg.ndof = me0s->degreesOfFreedom();
        
        Double_t reducedChi2 = me0_seg.chi2/(Float_t)me0_seg.ndof;
        
        meCollection["segReducedChi2"]->Fill(reducedChi2);
        meCollection["segNumberRH"]->Fill(me0_seg.numberRH);

        for (auto rh = me0rhs.begin(); rh!= me0rhs.end(); rh++){
            
            auto me0id = rh->me0Id();
            auto rhr = me0_geometry_->etaPartition(me0id);
            auto rhLP = rh->localPosition();
            auto erhLEP = rh->localPositionError();
            auto rhGP = rhr->toGlobal(rhLP);
            auto rhLPSegm = roll->toLocal(rhGP);
            float xe = segLP.x()+segLD.x()*rhLPSegm.z()/segLD.z();
            float ye = segLP.y()+segLD.y()*rhLPSegm.z()/segLD.z();
            float ze = rhLPSegm.z();
            LocalPoint extrPoint(xe,ye,ze); // in segment rest frame
            auto extSegm = rhr->toLocal(roll->toGlobal(extrPoint)); // in layer restframe
            
            me0_rhFromSeg.detId = me0id;
            
            me0_rhFromSeg.region = me0id.region();
            me0_rhFromSeg.station = 0;
            me0_rhFromSeg.ring = 0;
            me0_rhFromSeg.layer = me0id.layer();
            me0_rhFromSeg.chamber = me0id.chamber();
            me0_rhFromSeg.roll = me0id.roll();
            
            me0_rhFromSeg.x = rhLP.x();
            me0_rhFromSeg.xErr = erhLEP.xx();
            me0_rhFromSeg.y = rhLP.y();
            me0_rhFromSeg.yErr = erhLEP.yy();
            
            me0_rhFromSeg.globalR = rhGP.perp();
            me0_rhFromSeg.globalX = rhGP.x();
            me0_rhFromSeg.globalY = rhGP.y();
            me0_rhFromSeg.globalZ = rhGP.z();
            me0_rhFromSeg.globalEta = rhGP.eta();
            me0_rhFromSeg.globalPhi = rhGP.phi();
            
            me0_rhFromSeg.xExt = extSegm.x();
            me0_rhFromSeg.yExt = extSegm.y();
            
	    Double_t pull_x = (me0_rhFromSeg.x - me0_rhFromSeg.xExt) / sqrt(me0_rhFromSeg.xErr);
	    Double_t pull_y = (me0_rhFromSeg.y - me0_rhFromSeg.yExt) / sqrt(me0_rhFromSeg.yErr);            
            // abbreviations
            int reS(me0_rhFromSeg.region);
            int laS(me0_rhFromSeg.layer);
            
            bool verbose(false);
            if (verbose)
                std::cout <<" ME0 Layer Id "<<rh->me0Id()<<" error on the local point "<< erhLEP
                <<"\n-> Ensamble Rest Frame RH local position "<<rhLPSegm<<" Segment extrapolation "<<extrPoint
                <<"\n-> Layer Rest Frame RH local position "<<rhLP<<" Segment extrapolation "<<extSegm<<std::endl;
            
           // me0_rhSeg_tree_->Fill();
            
            meCollection["globalEtaSpecRH"]->Fill(me0_rhFromSeg.globalEta);
            meCollection["globalPhiSpecRH"]->Fill(me0_rhFromSeg.globalPhi);
            
            
            //Occupancy
            if(reS==-1 && laS==1) meCollection["localrh_xy_specRH_rm1_l1"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==-1 && laS==2) meCollection["localrh_xy_specRH_rm1_l2"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==-1 && laS==3) meCollection["localrh_xy_specRH_rm1_l3"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==-1 && laS==4) meCollection["localrh_xy_specRH_rm1_l4"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==-1 && laS==5) meCollection["localrh_xy_specRH_rm1_l5"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==-1 && laS==6) meCollection["localrh_xy_specRH_rm1_l6"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            
            if(reS==1 && laS==1) meCollection["localrh_xy_specRH_rp1_l1"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==1 && laS==2) meCollection["localrh_xy_specRH_rp1_l2"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==1 && laS==3) meCollection["localrh_xy_specRH_rp1_l3"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==1 && laS==4) meCollection["localrh_xy_specRH_rp1_l4"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==1 && laS==5) meCollection["localrh_xy_specRH_rp1_l5"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            if(reS==1 && laS==6) meCollection["localrh_xy_specRH_rp1_l6"]->Fill(me0_rhFromSeg.globalX,me0_rhFromSeg.globalY);
            
            //-----zR Occupancy-----
            const double glb_R_specRH(sqrt(me0_rh.globalX*me0_rh.globalX+me0_rh.globalY*me0_rh.globalY));
            if(reS==-1) meCollection["localrh_zr_specRH_rm1"]->Fill(me0_rhFromSeg.globalZ,glb_R_specRH);
            if(reS==1)  meCollection["localrh_zr_specRH_rp1"]->Fill(me0_rhFromSeg.globalZ,glb_R_specRH);
            
            //Delta X
            if(reS==-1 && laS==1) meCollection["specRecHitDX_rm1_l1"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==-1 && laS==2) meCollection["specRecHitDX_rm1_l2"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==-1 && laS==3) meCollection["specRecHitDX_rm1_l3"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==-1 && laS==4) meCollection["specRecHitDX_rm1_l4"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==-1 && laS==5) meCollection["specRecHitDX_rm1_l5"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==-1 && laS==6) meCollection["specRecHitDX_rm1_l6"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            
            if(reS==1 && laS==1) meCollection["specRecHitDX_rp1_l1"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==1 && laS==2) meCollection["specRecHitDX_rp1_l2"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==1 && laS==3) meCollection["specRecHitDX_rp1_l3"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==1 && laS==4) meCollection["specRecHitDX_rp1_l4"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==1 && laS==5) meCollection["specRecHitDX_rp1_l5"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            if(reS==1 && laS==6) meCollection["specRecHitDX_rp1_l6"]->Fill(me0_rhFromSeg.x-me0_rhFromSeg.xExt);
            //Delta Y
            if(reS==-1 && laS==1) meCollection["specRecHitDY_rm1_l1"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==-1 && laS==2) meCollection["specRecHitDY_rm1_l2"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==-1 && laS==3) meCollection["specRecHitDY_rm1_l3"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==-1 && laS==4) meCollection["specRecHitDY_rm1_l4"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==-1 && laS==5) meCollection["specRecHitDY_rm1_l5"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==-1 && laS==6) meCollection["specRecHitDY_rm1_l6"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            
            if(reS==1 && laS==1) meCollection["specRecHitDY_rp1_l1"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==1 && laS==2) meCollection["specRecHitDY_rp1_l2"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==1 && laS==3) meCollection["specRecHitDY_rp1_l3"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==1 && laS==4) meCollection["specRecHitDY_rp1_l4"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==1 && laS==5) meCollection["specRecHitDY_rp1_l5"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            if(reS==1 && laS==6) meCollection["specRecHitDY_rp1_l6"]->Fill(me0_rhFromSeg.y-me0_rhFromSeg.yExt);
            
            //Pull X
            if(reS==-1 && laS==1) meCollection["specRecHitPullLocalX_rm1_l1"]->Fill(pull_x);
            if(reS==-1 && laS==2) meCollection["specRecHitPullLocalX_rm1_l2"]->Fill(pull_x);
            if(reS==-1 && laS==3) meCollection["specRecHitPullLocalX_rm1_l3"]->Fill(pull_x);
            if(reS==-1 && laS==4) meCollection["specRecHitPullLocalX_rm1_l4"]->Fill(pull_x);
            if(reS==-1 && laS==5) meCollection["specRecHitPullLocalX_rm1_l5"]->Fill(pull_x);
            if(reS==-1 && laS==6) meCollection["specRecHitPullLocalX_rm1_l6"]->Fill(pull_x);
            
            if(reS==1 && laS==1) meCollection["specRecHitPullLocalX_rp1_l1"]->Fill(pull_x);
            if(reS==1 && laS==2) meCollection["specRecHitPullLocalX_rp1_l2"]->Fill(pull_x);
            if(reS==1 && laS==3) meCollection["specRecHitPullLocalX_rp1_l3"]->Fill(pull_x);
            if(reS==1 && laS==4) meCollection["specRecHitPullLocalX_rp1_l4"]->Fill(pull_x);
            if(reS==1 && laS==5) meCollection["specRecHitPullLocalX_rp1_l5"]->Fill(pull_x);
            if(reS==1 && laS==6) meCollection["specRecHitPullLocalX_rp1_l6"]->Fill(pull_x);
            
            //Pull Y
            if(reS==-1 && laS==1) meCollection["specRecHitPullLocalY_rm1_l1"]->Fill(pull_y);
            if(reS==-1 && laS==2) meCollection["specRecHitPullLocalY_rm1_l2"]->Fill(pull_y);
            if(reS==-1 && laS==3) meCollection["specRecHitPullLocalY_rm1_l3"]->Fill(pull_y);
            if(reS==-1 && laS==4) meCollection["specRecHitPullLocalY_rm1_l4"]->Fill(pull_y);
            if(reS==-1 && laS==5) meCollection["specRecHitPullLocalY_rm1_l5"]->Fill(pull_y);
            if(reS==-1 && laS==6) meCollection["specRecHitPullLocalY_rm1_l6"]->Fill(pull_y);
            
            if(reS==1 && laS==1) meCollection["specRecHitPullLocalY_rp1_l1"]->Fill(pull_y);
            if(reS==1 && laS==2) meCollection["specRecHitPullLocalY_rp1_l2"]->Fill(pull_y);
            if(reS==1 && laS==3) meCollection["specRecHitPullLocalY_rp1_l3"]->Fill(pull_y);
            if(reS==1 && laS==4) meCollection["specRecHitPullLocalY_rp1_l4"]->Fill(pull_y);
            if(reS==1 && laS==5) meCollection["specRecHitPullLocalY_rp1_l5"]->Fill(pull_y);
            if(reS==1 && laS==6) meCollection["specRecHitPullLocalY_rp1_l6"]->Fill(pull_y);
        
        }
    }

}
    
    


// ------------ method called once each job just before starting event loop  ------------
void 
MuonME0RecHits::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonME0RecHits::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------

void 
MuonME0RecHits::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
    try {
        iSetup.get<MuonGeometryRecord>().get(me0_geom);
        me0_geometry_ = &*me0_geom;
        hasME0Geometry_ = true;
    } catch (edm::eventsetup::NoProxyException<ME0Geometry>& e) {
        hasME0Geometry_ = false;
        LogDebug("MuonRecHitAnalyzer") << "+++ Info: ME0 geometry is unavailable. +++\n";
    }

    if(debug_) std::cout<<"booking Global histograms with "<<folderPath_<<std::endl;
    std::string folder;
    folder = folderPath_;
    dbe->setCurrentFolder(folder);
    
    if(hasME0Geometry_){
        
        int num_region=2;
        int num_layer=6;
        std::string region[2] ={"m1", "p1"};
        std::string layer[6] = {"l1", "l2", "l3", "l4", "l5", "l6"};
        
        meCollection["recHitDX"]=dbe->book1D("recHitDX","x^{local}_{sim} - x^{local}_{rec}; x^{local}_{sim} - x^{local}_{rec} [cm]; entries",100,-1,+1);
        meCollection["recHitPullLocalX"]=dbe->book1D("recHitPullLocalX","(x^{local}_{sim} - x^{local}_{rec})/#sigma_{x}; (x^{local}_{sim} - x^{local}_{rec})/#sigma_{x}; entries",100,-5,+5);
        meCollection["recHitDPhi"]=dbe->book1D("recHitDPhi","#phi_{rec} - #phi_{sim}; #phi_{rec} - #phi_{sim} [rad]; entries",100,-0.005,+0.005);
        
        meCollection["localrh_zr_rm1"]=dbe->book2D("localrh_zr_rm1","ME0 RecHit occupancy: region m1;globalZ [cm];globalR [cm]",80,-555,-515,120,20,160);
        meCollection["localrh_zr_rp1"]=dbe->book2D("localrh_zr_rp1","ME0 RecHit occupancy: region p1;globalZ [cm];globalR [cm]",80,515,555,120,20,160);
        
        //-------ME0 Segments
        meCollection["segReducedChi2"]=dbe->book1D("segReducedChi2","#chi^{2}/ndof; #chi^{2}/ndof; # Segments",100,0,5);
        meCollection["segNumberRH"]=dbe->book1D("segNumberRH","Number of fitted RecHits; # RecHits; entries",11,-0.5,10.5);
        //---
        meCollection["globalEtaSpecRH"]=dbe->book1D("globalEtaSpecRH","Fitted RecHits Eta Distribution; #eta; entries",200,-4.0,4.0);
        meCollection["globalPhiSpecRH"]=dbe->book1D("globalPhiSpecRH","Fitted RecHits Phi Distribution; #phi; entries",18,-3.14,3.14);
        //---
        meCollection["localrh_zr_specRH_rm1"]=dbe->book2D("localrh_zr_specRH_rm1","ME0 Specific RecHit occupancy: region m1;globalZ [cm];globalR [cm]",80,-555,-515,120,20,160);
        meCollection["localrh_zr_specRH_rp1"]=dbe->book2D("localrh_zr_specRH_rp1","ME0 Specific RecHit occupancy: region p1;globalZ [cm];globalR [cm]",80,515,555,120,20,160);
        
        
        for(int k=0;k<num_region;k++){
            for (int j=0;j<num_layer;j++){
                
                meCollection["recHitDX_r"+region[k]+"_"+layer[j]]=dbe->book1D("recHitDX_r"+region[k]+"_"+layer[j],"x^{local}_{sim} - x^{local}_{rec} region "+region[k]+", layer "+std::to_string(j+1)+"; x^{local}_{sim} - x^{local}_{rec} [cm]; entries",100,-1,+1);
                
                meCollection["recHitPullLocalX_r"+region[k]+"_"+layer[j]]=dbe->book1D("recHitPullLocalX_r"+region[k]+"_"+layer[j],"(x^{local}_{sim} - x^{local}_{rec})/#sigma_{x} region "+region[k]+", layer "+std::to_string(j+1)+"; (x^{local}_{sim} - x^{local}_{rec})/#sigma_{x}; entries",100,-5,+5);
               
                meCollection["recHitDPhi_r"+region[k]+"_"+layer[j]]=dbe->book1D("recHitDPhi_r"+region[k]+"_"+layer[j],"#phi_{rec} - #phi_{sim} region "+region[k]+", layer "+std::to_string(j+1)+"; #phi_{rec} - #phi_{sim} [rad]; entries",100,-0.001,+0.001);
                
                meCollection["localrh_xy_r"+region[k]+"_"+layer[j]]=dbe->book2D("localrh_xy_r"+region[k]+"_"+layer[j],"ME0 RecHit occupancy: region "+region[k]+", layer "+std::to_string(j+1)+";globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
                
                //ME0 segmentes RecHits
                meCollection["localrh_xy_specRH_r"+region[k]+"_"+layer[j]]=dbe->book2D("localrh_xy_specRH_r"+region[k]+"_"+layer[j],"ME0 Specific RecHit occupancy: region "+region[k]+", layer "+std::to_string(j+1)+";globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
                
                meCollection["specRecHitDX_r"+region[k]+"_"+layer[j]]=dbe->book1D("specRecHitDX_r"+region[k]+"_"+layer[j],"x^{local}_{rec} - x^{local}_{ext} region "+region[k]+", layer "+std::to_string(j+1)+"; x^{local}_{rec} - x^{local}_{ext} [cm]; entries",100,-1,+1);
                meCollection["specRecHitDY_r"+region[k]+"_"+layer[j]]=dbe->book1D("specRecHitDY_r"+region[k]+"_"+layer[j],"y^{local}_{rec} - y^{local}_{ext} region "+region[k]+", layer "+std::to_string(j+1)+"; y^{local}_{rec} - y^{local}_{ext} [cm]; entries",100,-5,+5);
                meCollection["specRecHitPullLocalX_r"+region[k]+"_"+layer[j]]=dbe->book1D("specRecHitPullLocalX_r"+region[k]+"_"+layer[j],"(x^{local}_{rec} - x^{local}_{ext})/#sigma_{x} region "+region[k]+", layer "+std::to_string(j+1)+"; (x^{local}_{rec} - x^{local}_{ext})/#sigma_{x} [cm]; entries",100,-5,+5);
                meCollection["specRecHitPullLocalY_r"+region[k]+"_"+layer[j]]=dbe->book1D("specRecHitPullLocalY_r"+region[k]+"_"+layer[j],"(y^{local}_{rec} - y^{local}_{ext})/#sigma_{y} region "+region[k]+", layer "+std::to_string(j+1)+"; (y^{local}_{rec} - y^{local}_{ext})/#sigma_{y} [cm]; entries",100,-5,+5);
                
            }//Layers loop
        
        }//Stations loop
        
    }
}


// ------------ method called when ending the processing of a run  ------------

void 
MuonME0RecHits::endRun(edm::Run const&, edm::EventSetup const&)
{
    if (EffSaveRootFile_) dbe->save(EffRootFileName_);
}


// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
MuonME0RecHits::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
MuonME0RecHits::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

bool MuonME0RecHits::isME0RecHitMatched(MyME0RecHit me0_recHit_, MyME0SimHit me0_sh)
{
    
    Int_t me0_region = me0_recHit_.region;
    Int_t me0_layer = me0_recHit_.layer;
    Int_t me0_station = me0_recHit_.station;
    Int_t me0_chamber = me0_recHit_.chamber;
    Int_t me0_roll = me0_recHit_.roll;
    Int_t me0_firstStrip = me0_recHit_.firstClusterStrip;
    Int_t me0_cls = me0_recHit_.clusterSize;
    
    Int_t me0_sh_region = me0_sh.region;
    Int_t me0_sh_layer = me0_sh.layer;
    Int_t me0_sh_station = me0_sh.station;
    Int_t me0_sh_chamber = me0_sh.chamber;
    Int_t me0_sh_roll = me0_sh.roll;
    Int_t me0_sh_strip = me0_sh.strip;
    
    std::vector<int> stripsFired;
    for(int i = me0_firstStrip; i < (me0_firstStrip + me0_cls); i++){
        
        stripsFired.push_back(i);
        
    }
    
    bool cond1, cond2, cond3;
    
    if(me0_sh_region == me0_region && me0_sh_layer == me0_layer && me0_sh_station == me0_station) cond1 = true;
    else cond1 = false;
    if(me0_sh_chamber == me0_chamber && me0_sh_roll == me0_roll) cond2 = true;
    else cond2 = false;
    if(std::find(stripsFired.begin(), stripsFired.end(), (me0_sh_strip + 1)) != stripsFired.end()) cond3 = true;
    else cond3 = false;
    
    if(me0_cls == 0) cond3 = true;
    
    //std::cout<<"cond1: "<<cond1<<" cond2: "<<cond2<<" cond3: "<<cond3<<std::endl;
    return (cond1 & cond2 & cond3);
    
}


bool MuonME0RecHits::isSimTrackGood(const SimTrack &t)
{
    // SimTrack selection
    if (t.noVertex()) return false;
    if (t.noGenpart()) return false;
    // only muons
    if (std::abs(t.type()) != 13 and simTrackOnlyMuon_) return false;
    // pt selection
    if (t.momentum().pt() < simTrackMinPt_) return false;
    // eta selection
    const float eta(std::abs(t.momentum().eta()));
    if (eta > simTrackMaxEta_ || eta < simTrackMinEta_) return false;
    return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuonME0RecHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonME0RecHits);

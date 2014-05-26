// -*- C++ -*-
//
// Package:    MuonME0Digis
// Class:      MuonME0Digis
// 
/**\class MuonME0Digis MuonME0Digis.cc Validation/MuonME0Digis/plugins/MuonME0Digis.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Claudio Caputo, INFN Bari
//         Created:  Tue, 18 Mar 2014 21:24:33 GMT
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
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
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

struct MyME0Digi
{
    Int_t detId, particleType;
    Short_t region, ring, station, layer, chamber, roll;
    Short_t strip, bx;
    Float_t x, y;
    Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
    Float_t x_sim, y_sim;
    Float_t g_eta_sim, g_phi_sim, g_x_sim, g_y_sim, g_z_sim;
};

struct MySimTrack
{
    Float_t pt, eta, phi;
    Char_t charge;
    Char_t endcap;
    Char_t gem_sh_layer1, gem_sh_layer2;
    Char_t gem_dg_layer1, gem_dg_layer2;
    Char_t gem_pad_layer1, gem_pad_layer2;
    Float_t gem_sh_eta, gem_sh_phi;
    Float_t gem_sh_x, gem_sh_y;
    Float_t gem_dg_eta, gem_dg_phi;
    Float_t gem_pad_eta, gem_pad_phi;
    Float_t gem_lx_even, gem_ly_even;
    Float_t gem_lx_odd, gem_ly_odd;
    Char_t  has_gem_sh_l1, has_gem_sh_l2;
    Char_t  has_gem_dg_l1, has_gem_dg_l2;
    Char_t  has_gem_pad_l1, has_gem_pad_l2;
    Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho;
};


class MuonME0Digis : public edm::EDAnalyzer {
   public:
      explicit MuonME0Digis(const edm::ParameterSet&);
      ~MuonME0Digis();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
    
      bool isSimTrackGood(const SimTrack &);
      bool isME0DigiMatched(MyME0Digi me0_dg, MyME0SimHit me0_sh);

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
    
    edm::Handle<edm::PSimHitContainer> ME0Hits;
    edm::Handle<ME0DigiPreRecoCollection> me0_digis;
    edm::Handle<edm::SimTrackContainer> sim_tracks;
    edm::Handle<edm::SimVertexContainer> sim_vertices;
    edm::ESHandle<ME0Geometry> me0_geom;
    
    edm::ParameterSet cfg_;
    
    edm::InputTag me0SimHitInput_;
    edm::InputTag me0DigiInput_;
    edm::InputTag simTrackInput_;
    
    double simTrackMinPt_;
    double simTrackMaxPt_;
    double simTrackMinEta_;
    double simTrackMaxEta_;
    double simTrackOnlyMuon_;
    
    const ME0Geometry* me0_geometry_;
    
    MyME0Digi me0_digi_;
    MyME0SimHit me0_sh;
    MySimTrack track_;
    
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
MuonME0Digis::MuonME0Digis(const edm::ParameterSet& iConfig):
debug_(iConfig.getUntrackedParameter<bool>("debug")),
folderPath_(iConfig.getUntrackedParameter<std::string>("folderPath")),
EffSaveRootFile_(iConfig.getUntrackedParameter<bool>("EffSaveRootFile")),
EffRootFileName_(iConfig.getUntrackedParameter<std::string>("EffRootFileName"))
{
   //now do what ever initialization is needed
    
    //now do what ever initialization is needed
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
    
    auto me0Digi = cfg_.getParameter<edm::ParameterSet>("me0StripDigi");
    me0DigiInput_ = me0Digi.getParameter<edm::InputTag>("input");
    
    hasME0Geometry_=false;

}


MuonME0Digis::~MuonME0Digis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
MuonME0Digis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    
    iEvent.getByLabel(simTrackInput_, sim_tracks);
    iEvent.getByLabel(simTrackInput_, sim_vertices);
    iEvent.getByLabel(me0SimHitInput_, ME0Hits);
    iEvent.getByLabel(me0DigiInput_, me0_digis);
    
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
        
        for(ME0DigiPreRecoCollection::DigiRangeIterator cItr = me0_digis->begin(); cItr != me0_digis->end(); ++cItr){
            
            ME0DetId id = (*cItr).first;
            
            const GeomDet* gdet = me0_geom->idToDet(id);
            const BoundPlane & surface = gdet->surface();
            
            me0_digi_.detId = id();
            me0_digi_.region = (Short_t) id.region();
            me0_digi_.ring = 0;
            me0_digi_.station = 0;
            me0_digi_.layer = (Short_t) id.layer();
            me0_digi_.chamber = (Short_t) id.chamber();
            me0_digi_.roll = (Short_t) id.roll();
            
            ME0DigiPreRecoCollection::const_iterator digiItr;
            //loop over digis of given roll
            for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
            {
		if(abs(digiItr->pdgid()) != 13) continue;
		me0_digi_.particleType = digiItr->pdgid();
                me0_digi_.strip = 0;
                me0_digi_.bx = 0;
                
                me0_digi_.x = (Float_t) digiItr->x();
                me0_digi_.y = (Float_t) digiItr->y();
                
                LocalPoint lp(digiItr->x(), digiItr->y(), 0);
                
                GlobalPoint gp = surface.toGlobal(lp);
                me0_digi_.g_r = (Float_t) gp.perp();
                me0_digi_.g_eta = (Float_t) gp.eta();
                me0_digi_.g_phi = (Float_t) gp.phi();
                me0_digi_.g_x = (Float_t) gp.x();
                me0_digi_.g_y = (Float_t) gp.y();
                me0_digi_.g_z = (Float_t) gp.z();
                
                me0_digi_.x_sim = me0_sh.x;
                me0_digi_.y_sim = me0_sh.y;
                me0_digi_.g_eta_sim = me0_sh.globalEta;
                me0_digi_.g_phi_sim = me0_sh.globalPhi;
                me0_digi_.g_x_sim = me0_sh.globalX;
                me0_digi_.g_y_sim = me0_sh.globalY;
                me0_digi_.g_z_sim = me0_sh.globalZ;
                
                // abbreviations
                int re(me0_digi_.region);
                int la(me0_digi_.layer);
                
                if(me0_digi_.bx != 0) continue;
                if(isME0DigiMatched(me0_digi_, me0_sh)){
                    count++;
                    
                    /*------------zR Occupancy--------------*/
                    const double glb_R(sqrt(me0_digi_.g_x*me0_digi_.g_x+me0_digi_.g_y*me0_digi_.g_y));
                    if(re==-1) meCollection["strip_dg_zr_rm1"]->Fill(me0_digi_.g_z,glb_R);
                    if(re==1) meCollection["strip_dg_zr_rp1"]->Fill(me0_digi_.g_z,glb_R);
                    
                    /*-------------XY Occupancy---------------*/
                    if(re==-1 && la==1) meCollection["strip_dg_xy_rm1_l1"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==-1 && la==2) meCollection["strip_dg_xy_rm1_l2"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==-1 && la==3) meCollection["strip_dg_xy_rm1_l3"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==-1 && la==4) meCollection["strip_dg_xy_rm1_l4"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==-1 && la==5) meCollection["strip_dg_xy_rm1_l5"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==-1 && la==6) meCollection["strip_dg_xy_rm1_l6"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    
                    if(re==1 && la==1) meCollection["strip_dg_xy_rp1_l1"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==1 && la==2) meCollection["strip_dg_xy_rp1_l2"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==1 && la==3) meCollection["strip_dg_xy_rp1_l3"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==1 && la==4) meCollection["strip_dg_xy_rp1_l4"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==1 && la==5) meCollection["strip_dg_xy_rp1_l5"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    if(re==1 && la==6) meCollection["strip_dg_xy_rp1_l6"]->Fill(me0_sh.globalX,me0_sh.globalY);
                    
                    /*------------ (x_digi_sim - x_digi_rec) ------------*/
                    meCollection["digiDX"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    
                    if(re==-1 && la==1) meCollection["digiDX_rm1_l1"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==-1 && la==2) meCollection["digiDX_rm1_l2"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==-1 && la==3) meCollection["digiDX_rm1_l3"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==-1 && la==4) meCollection["digiDX_rm1_l4"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==-1 && la==5) meCollection["digiDX_rm1_l5"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==-1 && la==6) meCollection["digiDX_rm1_l6"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    
                    if(re==1 && la==1) meCollection["digiDX_rp1_l1"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==1 && la==2) meCollection["digiDX_rp1_l2"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==1 && la==3) meCollection["digiDX_rp1_l3"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==1 && la==4) meCollection["digiDX_rp1_l4"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==1 && la==5) meCollection["digiDX_rp1_l5"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                    if(re==1 && la==6) meCollection["digiDX_rp1_l6"]->Fill(me0_digi_.g_x_sim-me0_digi_.g_x);
                }
            }
            
        }
        me0_sh.countMatching = count;
    }

}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonME0Digis::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonME0Digis::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------

void MuonME0Digis::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
    try {
        iSetup.get<MuonGeometryRecord>().get(me0_geom);
        me0_geometry_ = &*me0_geom;
        hasME0Geometry_ = true;
    } catch (edm::eventsetup::NoProxyException<ME0Geometry>& e) {
        hasME0Geometry_ = false;
        LogDebug("MuonSimHitAnalyzer") << "+++ Info: ME0 geometry is unavailable. +++\n";
    }

    if(debug_) std::cout<<"booking Global histograms with "<<folderPath_<<std::endl;
    std::string folder;
    folder = folderPath_;
    dbe->setCurrentFolder(folder);
    
    if(hasME0Geometry_){
        
        int num_region=2;
        
        std::string region[2] ={"m1", "p1"};
        
        meCollection["strip_dg_zr_rm1"]=dbe->book2D("strip_dg_zr_rm1","Digi occupancy: region m1;globalZ [cm];globalR [cm]",80,-555,-515,120,20,160);
        meCollection["strip_dg_zr_rp1"]=dbe->book2D("strip_dg_zr_rp1","Digi occupancy: region p1;globalZ [cm];globalR [cm]",80,515,555,120,20,160);
        meCollection["digiDX"]=dbe->book1D("digiDX","x^{digi}_{sim} - x^{digi}_{rec}; x^{digi}_{sim} - x^{digi}_{rec} [cm]; entries",100,-10,+10);
        
        for(int k=0;k<num_region;k++){
            
            //std::cout<<"REGION!!!!!!   "<<region[k]<<std::endl;
            
            meCollection["strip_dg_xy_r"+region[k]+"_l1"]=dbe->book2D("strip_dg_xy_r"+region[k]+"_l1","Digi occupancy: region "+region[k]+", layer1;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["strip_dg_xy_r"+region[k]+"_l2"]=dbe->book2D("strip_dg_xy_r"+region[k]+"_l2","Digi occupancy: region "+region[k]+", layer2;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["strip_dg_xy_r"+region[k]+"_l3"]=dbe->book2D("strip_dg_xy_r"+region[k]+"_l3","Digi occupancy: region "+region[k]+", layer3;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["strip_dg_xy_r"+region[k]+"_l4"]=dbe->book2D("strip_dg_xy_r"+region[k]+"_l4","Digi occupancy: region "+region[k]+", layer4;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["strip_dg_xy_r"+region[k]+"_l5"]=dbe->book2D("strip_dg_xy_r"+region[k]+"_l5","Digi occupancy: region "+region[k]+", layer5;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["strip_dg_xy_r"+region[k]+"_l6"]=dbe->book2D("strip_dg_xy_r"+region[k]+"_l6","Digi occupancy: region "+region[k]+", layer6;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            
            meCollection["digiDX_r"+region[k]+"_l1"]=dbe->book1D("digiDX_r"+region[k]+"_l1","x^{digi}_{sim} - x^{digi}_{rec} region "+region[k]+", layer1; x^{digi}_{sim} - x^{digi}_{rec} [cm]; entries",100,-10,+10);
            meCollection["digiDX_r"+region[k]+"_l2"]=dbe->book1D("digiDX_r"+region[k]+"_l2","x^{digi}_{sim} - x^{digi}_{rec} region "+region[k]+", layer2; x^{digi}_{sim} - x^{digi}_{rec} [cm]; entries",100,-10,+10);
            meCollection["digiDX_r"+region[k]+"_l3"]=dbe->book1D("digiDX_r"+region[k]+"_l3","x^{digi}_{sim} - x^{digi}_{rec} region "+region[k]+", layer3; x^{digi}_{sim} - x^{digi}_{rec} [cm]; entries",100,-10,+10);
            meCollection["digiDX_r"+region[k]+"_l4"]=dbe->book1D("digiDX_r"+region[k]+"_l4","x^{digi}_{sim} - x^{digi}_{rec} region "+region[k]+", layer4; x^{digi}_{sim} - x^{digi}_{rec} [cm]; entries",100,-10,+10);
            meCollection["digiDX_r"+region[k]+"_l5"]=dbe->book1D("digiDX_r"+region[k]+"_l5","x^{digi}_{sim} - x^{digi}_{rec} region "+region[k]+", layer5; x^{digi}_{sim} - x^{digi}_{rec} [cm]; entries",100,-10,+10);
            meCollection["digiDX_r"+region[k]+"_l6"]=dbe->book1D("digiDX_r"+region[k]+"_l6","x^{digi}_{sim} - x^{digi}_{rec} region "+region[k]+", layer6; x^{digi}_{sim} - x^{digi}_{rec} [cm]; entries",100,-10,+10);
        }
    }
}


// ------------ method called when ending the processing of a run  ------------

void MuonME0Digis::endRun(edm::Run const&, edm::EventSetup const&)
{
    if (EffSaveRootFile_) dbe->save(EffRootFileName_);
}


// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
MuonME0Digis::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
MuonME0Digis::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

bool MuonME0Digis::isME0DigiMatched(MyME0Digi me0_dg, MyME0SimHit me0_sh)
{
    
    Int_t me0_region = me0_dg.region;
    Int_t me0_layer = me0_dg.layer;
    Int_t me0_station = me0_dg.station;
    Int_t me0_chamber = me0_dg.chamber;
    Int_t me0_roll = me0_dg.roll;
    //Int_t me0_strip = me0_dg.strip;
    
    Int_t me0_sh_region = me0_sh.region;
    Int_t me0_sh_layer = me0_sh.layer;
    Int_t me0_sh_station = me0_sh.station;
    Int_t me0_sh_chamber = me0_sh.chamber;
    Int_t me0_sh_roll = me0_sh.roll;
    //Int_t me0_sh_strip = me0_sh.strip;
    
    bool cond1, cond2;
    
    if(me0_sh_region == me0_region && me0_sh_layer == me0_layer && me0_sh_station == me0_station) cond1 = true;
    else cond1 = false;
    if(me0_sh_chamber == me0_chamber && me0_sh_roll == me0_roll) cond2 = true;
    else cond2 = false;
    
    return (cond1 & cond2);
    
}

bool MuonME0Digis::isSimTrackGood(const SimTrack &t)
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
MuonME0Digis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonME0Digis);

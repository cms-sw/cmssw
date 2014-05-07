// -*- C++ -*-
//
// Package:    MuonME0Hits
// Class:      MuonME0Hits
// 
/**\class MuonME0Hits MuonME0Hits.cc MuonME0Hits/MuonME0Hits/plugins/MuonME0Hits.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author: Claudio Caputo, INFN Bari  
//         Created:  Fri, 14 Mar 2014 09:44:58 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"


struct MyME0SimHit
{
    Int_t eventNumber;
    Int_t detUnitId, particleType;
    Float_t x, y, energyLoss, pabs, timeOfFlight;
    Int_t region, layer, chamber, roll;
    Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
    Int_t strip;
    Float_t Phi_0, DeltaPhi, R_0;
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

//
// class declaration
//

class MuonME0Hits : public edm::EDAnalyzer {
   public:
      explicit MuonME0Hits(const edm::ParameterSet&);
      ~MuonME0Hits();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
    
      bool isSimTrackGood(const SimTrack &);

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
    edm::Handle<edm::SimTrackContainer> sim_tracks;
    edm::Handle<edm::SimVertexContainer> sim_vertices;
    edm::ESHandle<ME0Geometry> me0_geom;
    
    const ME0Geometry* me0_geometry_;
    
    
    edm::ParameterSet cfg_;
    
    edm::InputTag me0SimHitInput_;
    edm::InputTag simTrackInput_;
    
    double simTrackMinPt_;
    double simTrackMaxPt_;
    double simTrackMinEta_;
    double simTrackMaxEta_;
    double simTrackOnlyMuon_;
    
    MyME0SimHit me0_sh;
    
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
MuonME0Hits::MuonME0Hits(const edm::ParameterSet& iConfig):
debug_(iConfig.getUntrackedParameter<bool>("debug")),
folderPath_(iConfig.getUntrackedParameter<std::string>("folderPath")),
EffSaveRootFile_(iConfig.getUntrackedParameter<bool>("EffSaveRootFile")),
EffRootFileName_(iConfig.getUntrackedParameter<std::string>("EffRootFileName"))
{
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
    
    hasME0Geometry_=false;
}


MuonME0Hits::~MuonME0Hits()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
MuonME0Hits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    
    iEvent.getByLabel(me0SimHitInput_, ME0Hits);
    iEvent.getByLabel(simTrackInput_, sim_tracks);
    
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
        //  me0_sh.strip=me0_geometry_->etaPartition(itHit->detUnitId())->strip(hitLP);
        const LocalPoint hitEP(itHit->entryPoint());
        me0_sh.strip=me0_geometry_->etaPartition(itHit->detUnitId())->strip(hitEP);
        
        // abbreviations
        int re(me0_sh.region);
        int la(me0_sh.layer);
    
        /*-------------XY Occupancy---------------*/
        if(re==-1 && la==1) meCollection["sh_me0_xy_rm1_l1"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==-1 && la==2) meCollection["sh_me0_xy_rm1_l2"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==-1 && la==3) meCollection["sh_me0_xy_rm1_l3"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==-1 && la==4) meCollection["sh_me0_xy_rm1_l4"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==-1 && la==5) meCollection["sh_me0_xy_rm1_l5"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==-1 && la==6) meCollection["sh_me0_xy_rm1_l6"]->Fill(me0_sh.globalX,me0_sh.globalY);
        
        if(re==1 && la==1) meCollection["sh_me0_xy_rp1_l1"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==1 && la==2) meCollection["sh_me0_xy_rp1_l2"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==1 && la==3) meCollection["sh_me0_xy_rp1_l3"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==1 && la==4) meCollection["sh_me0_xy_rp1_l4"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==1 && la==5) meCollection["sh_me0_xy_rp1_l5"]->Fill(me0_sh.globalX,me0_sh.globalY);
        if(re==1 && la==6) meCollection["sh_me0_xy_rp1_l6"]->Fill(me0_sh.globalX,me0_sh.globalY);
        
        /*------------zR Occupancy--------------*/
        const double glb_R(sqrt(me0_sh.globalX*me0_sh.globalX+me0_sh.globalY*me0_sh.globalY));
        if(re==-1) meCollection["sh_me0_zr_rm1"]->Fill(me0_sh.globalZ,glb_R);
        if(re==1) meCollection["sh_me0_zr_rp1"]->Fill(me0_sh.globalZ,glb_R);
        
        /*-----------Time of Flight-------------*/
        if(re==-1 && la==1) meCollection["sh_me0_tof_rm1_l1"]->Fill(me0_sh.timeOfFlight);
        if(re==-1 && la==2) meCollection["sh_me0_tof_rm1_l2"]->Fill(me0_sh.timeOfFlight);
        if(re==-1 && la==3) meCollection["sh_me0_tof_rm1_l3"]->Fill(me0_sh.timeOfFlight);
        if(re==-1 && la==4) meCollection["sh_me0_tof_rm1_l4"]->Fill(me0_sh.timeOfFlight);
        if(re==-1 && la==5) meCollection["sh_me0_tof_rm1_l5"]->Fill(me0_sh.timeOfFlight);
        if(re==-1 && la==6) meCollection["sh_me0_tof_rm1_l6"]->Fill(me0_sh.timeOfFlight);
        
        if(re==1 && la==1) meCollection["sh_me0_tof_rp1_l1"]->Fill(me0_sh.timeOfFlight);
        if(re==1 && la==2) meCollection["sh_me0_tof_rp1_l2"]->Fill(me0_sh.timeOfFlight);
        if(re==1 && la==3) meCollection["sh_me0_tof_rp1_l3"]->Fill(me0_sh.timeOfFlight);
        if(re==1 && la==4) meCollection["sh_me0_tof_rp1_l4"]->Fill(me0_sh.timeOfFlight);
        if(re==1 && la==5) meCollection["sh_me0_tof_rp1_l5"]->Fill(me0_sh.timeOfFlight);
        if(re==1 && la==6) meCollection["sh_me0_tof_rp1_l6"]->Fill(me0_sh.timeOfFlight);
    }


}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonME0Hits::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonME0Hits::endJob() 
{
     dbe = 0;
}

// ------------ method called when starting to processes a run  ------------

void 
MuonME0Hits::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
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
        
        //int num_region=me0_geometry_->regions().size();   Non sono definiti le regions()   80,515,555,120,20,280
        int num_region=2;
        
        std::string region[2] ={"m1", "p1"};
        
        meCollection["sh_me0_zr_rm1"]=dbe->book2D("sh_me0_zr_rm1","SimHit occupancy: region m1;globalZ [cm];globalR [cm]",80,-555,-515,120,20,160);
        meCollection["sh_me0_zr_rp1"]=dbe->book2D("sh_me0_zr_rp1","SimHit occupancy: region p1;globalZ [cm];globalR [cm]",80,515,555,120,20,160);
        
        for(int k=0;k<num_region;k++){
            meCollection["sh_me0_xy_r"+region[k]+"_l1"]=dbe->book2D("sh_me0_xy_r"+region[k]+"_l1","SimHit occupancy: region "+region[k]+", layer1;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["sh_me0_xy_r"+region[k]+"_l2"]=dbe->book2D("sh_me0_xy_r"+region[k]+"_l2","SimHit occupancy: region "+region[k]+", layer2;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["sh_me0_xy_r"+region[k]+"_l3"]=dbe->book2D("sh_me0_xy_r"+region[k]+"_l3","SimHit occupancy: region "+region[k]+", layer3;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["sh_me0_xy_r"+region[k]+"_l4"]=dbe->book2D("sh_me0_xy_r"+region[k]+"_l4","SimHit occupancy: region "+region[k]+", layer4;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["sh_me0_xy_r"+region[k]+"_l5"]=dbe->book2D("sh_me0_xy_r"+region[k]+"_l5","SimHit occupancy: region "+region[k]+", layer5;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            meCollection["sh_me0_xy_r"+region[k]+"_l6"]=dbe->book2D("sh_me0_xy_r"+region[k]+"_l6","SimHit occupancy: region "+region[k]+", layer6;globalX [cm];globalY [cm]",120,-280,280,120,-280,280);
            
            meCollection["sh_me0_tof_r"+region[k]+"_l1"]=dbe->book1D("sh_me0_tof_r"+region[k]+"_l1", "SimHit TOF: region "+region[k]+", layer1;Time of flight [ns];entries",40,15,22);
            meCollection["sh_me0_tof_r"+region[k]+"_l2"]=dbe->book1D("sh_me0_tof_r"+region[k]+"_l2", "SimHit TOF: region "+region[k]+", layer2;Time of flight [ns];entries",40,15,22);
            meCollection["sh_me0_tof_r"+region[k]+"_l3"]=dbe->book1D("sh_me0_tof_r"+region[k]+"_l3", "SimHit TOF: region "+region[k]+", layer3;Time of flight [ns];entries",40,15,22);
            meCollection["sh_me0_tof_r"+region[k]+"_l4"]=dbe->book1D("sh_me0_tof_r"+region[k]+"_l4", "SimHit TOF: region "+region[k]+", layer4;Time of flight [ns];entries",40,15,22);
            meCollection["sh_me0_tof_r"+region[k]+"_l5"]=dbe->book1D("sh_me0_tof_r"+region[k]+"_l5", "SimHit TOF: region "+region[k]+", layer5;Time of flight [ns];entries",40,15,22);
            meCollection["sh_me0_tof_r"+region[k]+"_l6"]=dbe->book1D("sh_me0_tof_r"+region[k]+"_l6", "SimHit TOF: region "+region[k]+", layer6;Time of flight [ns];entries",40,15,22);
        }
    }
    
}


// ------------ method called when ending the processing of a run  ------------

void 
MuonME0Hits::endRun(edm::Run const&, edm::EventSetup const&)
{
    if (EffSaveRootFile_) dbe->save(EffRootFileName_);
}


// ------------ mathod that selects good sim tracks ------------
bool MuonME0Hits::isSimTrackGood(const SimTrack &t)
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


// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
MuonME0Hits::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
MuonME0Hits::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuonME0Hits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonME0Hits);

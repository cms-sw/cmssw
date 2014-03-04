#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>

GEMHitsValidation::GEMHitsValidation(DQMStore* dbe, const edm::InputTag & inputTag)
:  GEMBaseValidation(dbe, inputTag)
{}

void GEMHitsValidation::bookHisto() {
  LogDebug("MuonGEMHitsValidation")<<"Info : Loading Geometry information\n";

  dbe_->setCurrentFolder("MuonGEMHitsV/GEMHitsTask");


  Int_t nregion  = theGEMGeometry->regions().size();
  Int_t nstation = theGEMGeometry->regions()[0]->stations().size() ;

  npart    = theGEMGeometry->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions().size();


  LogDebug("MuonGEMHitsValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;
  LogDebug("MuonGEMHitsValidation")<<"+++ Info : # of stations : "<<nstation<<std::endl;
  LogDebug("MuonGEMHitsValidation")<<"+++ Info : # of eta partition : "<< npart <<std::endl;


  std::vector< std::string > region;
  std::vector< std::string > station;

  if ( nregion == 2) { region.push_back("-1"); region.push_back("1"); }
  else LogDebug("MuonGEMHitsValidation")<<"+++ Error : # of regions is not 2!\n";

  if ( nstation == 1 ) { station.push_back("1"); } 
  else if ( nstation == 3 ) { station.push_back("1"); station.push_back("2"); station.push_back("3"); } 
  else LogDebug("MuonGEMHitsValidation")<<"+++ Error : # of stations is not 1 or 3.\n"; 

  std::string layer[2]= { "1","2" } ; 
  std::string has_muon[3]= { "_Muon","_noMuon","_All"} ;

  LogDebug("MuonGEMHitsValidation")<<"+++ Info : finish to get geometry information from ES.\n";



  for( int i=0 ; i <3 ; i++) {
    gem_sh_zr_rm1[i] =  dbe_->book2D("gem_sh_zr_rm1"+has_muon[i], "SimHit occupancy: region-1; globalZ [cm] ; globalR [cm] ", 200,-573,-564,110,130,240);
    gem_sh_zr_rp1[i] =  dbe_->book2D("gem_sh_zr_rp1"+has_muon[i], "SimHit occupancy: region 1; globalZ [cm] ; globalR [cm] ", 200, 564, 573,110,130,240);
 
    gem_sh_tof_rm1_l1[i] = dbe_->book1D("gem_sh_tof_rm1_l1"+has_muon[i], "SimHit TOF : region-1, layer 1 ; Time of flight [ns] ; entries", 40,18,22);
    gem_sh_tof_rm1_l2[i] = dbe_->book1D("gem_sh_tof_rm1_l2"+has_muon[i], "SimHit TOF : region-1, layer 2 ; Time of flight [ns] ; entries", 40,18,22);
    gem_sh_tof_rp1_l1[i] = dbe_->book1D("gem_sh_tof_rp1_l1"+has_muon[i], "SimHit TOF : region 1, layer 1 ; Time of flight [ns] ; entries", 40,18,22);
    gem_sh_tof_rp1_l2[i] = dbe_->book1D("gem_sh_tof_rp1_l2"+has_muon[i], "SimHit TOF : region 1, layer 2 ; Time of flight [ns] ; entries", 40,18,22);
 
    gem_sh_pabs[i]  = dbe_->book1D("gem_sh_pabs"+has_muon[i], "SimHit absolute momentum; Momentum [GeV/c]; entries", 200,0.,200.);
 
    gem_sh_pdgid[i] = dbe_->book1D("gem_sh_pdgid"+has_muon[i], "SimHit PDG ID ; PDG ID ; entries", 200,-100.,100.);
 
    gem_sh_global_eta[i] = dbe_->book1D("gem_sh_global_eta"+has_muon[i],"SimHit occupancy in eta partitions; occupancy in #eta partition; entries",4*npart,1.,1+4*npart);
 
    gem_sh_energyloss[i] = dbe_->book1D("gem_sh_energyloss"+has_muon[i],"SimHit energy loss;Energy loss [eV];entries",60,0.,6000.);

  }

  for( unsigned int region_num = 0 ; region_num < region.size() ; region_num++ ) {
    for( unsigned int station_num = 0 ; station_num < station.size() ; station_num++) {
      for( unsigned int layer_num = 0 ; layer_num < 2 ; layer_num++) {
        for( unsigned int sel = 0 ; sel < 3 ; sel++) {
          std::string hist_name  = std::string("gem_sh_xy_r")+region[region_num]+"_st"+station[station_num]+"_l"+layer[layer_num]+has_muon[sel];
          std::string hist_label = has_muon[sel]+" SimHit occupancy : region"+region[region_num]+" station "+station[station_num]+" layer "+layer[layer_num]+" ; globalX [cm]; globalY[cm]";
          gem_sh_xy[region_num][station_num][layer_num][sel] = dbe_->book2D( hist_name.c_str(), hist_label.c_str(), 100, -260,260,100,-260,260);
        }   
      }
    }
  }
}


GEMHitsValidation::~GEMHitsValidation() {
}


void GEMHitsValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{

  edm::Handle<edm::PSimHitContainer> GEMHits;
  e.getByLabel(theInputTag, GEMHits);
  if (!GEMHits.isValid()) {
    edm::LogError("GEMHitsValidation") << "Cannot get GEMHits by label "
                                       << theInputTag.encode();
  }

  for (auto hits=GEMHits->begin(); hits!=GEMHits->end(); hits++) {
    Int_t particleType = hits->particleType();
    Float_t energyLoss = hits->energyLoss();
    Float_t pabs = hits->pabs();
    Float_t timeOfFlight = hits->timeOfFlight();
    
    const GEMDetId id(hits->detUnitId());
    
    Int_t region = id.region();
    Int_t station = id.station();
    Int_t layer = id.layer();
    Int_t roll = id.roll();


    const LocalPoint p0(0., 0., 0.);
    const GlobalPoint Gp0(theGEMGeometry->idToDet(hits->detUnitId())->surface().toGlobal(p0));
    const LocalPoint hitLP(hits->localPosition());
    const GlobalPoint hitGP(theGEMGeometry->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
    Float_t g_r = hitGP.perp();
    Float_t g_x = hitGP.x();
    Float_t g_y = hitGP.y();
    Float_t g_z = hitGP.z();

    const LocalPoint hitEP(hits->entryPoint());

      // fill hist
      int muonSel=999;
      int all = 2 ; 
      if ( TMath::Abs(particleType)== 13 ) { muonSel = 0 ; }
      else { muonSel=1; }
      // First, fill variable has no condition.
      gem_sh_pabs[all]->Fill(pabs);
      gem_sh_pdgid[all]->Fill(particleType);    
      gem_sh_energyloss[all]->Fill( energyLoss*1.e9);

      gem_sh_pabs[muonSel]->Fill(pabs);
      gem_sh_pdgid[muonSel]->Fill(particleType);    
      gem_sh_energyloss[muonSel]->Fill( energyLoss*1.e9);

      gem_sh_xy[(int)(region/2.+0.5)][station-1][layer-1][all]->Fill(g_x,g_y);
      gem_sh_xy[(int)(region/2.+0.5)][station-1][layer-1][muonSel]->Fill(g_x,g_y);
      if ( region== -1 ) {
         gem_sh_zr_rm1[all]->Fill(g_z,g_r);
         gem_sh_zr_rm1[muonSel]->Fill(g_z,g_r);
	if ( layer == 1 ) {
          gem_sh_tof_rm1_l1[all]->Fill(timeOfFlight);
          gem_sh_global_eta[all]->Fill( roll+ 0 + 0);    // roll + layer + region
          gem_sh_tof_rm1_l1[muonSel]->Fill(timeOfFlight);
          gem_sh_global_eta[muonSel]->Fill( roll+ 0 + 0);    // roll + layer + region
        }
        else if ( layer ==2 ) {
          gem_sh_tof_rm1_l2[all]->Fill(timeOfFlight);
          gem_sh_global_eta[all]->Fill( roll+ npart + 0);
          gem_sh_tof_rm1_l2[muonSel]->Fill(timeOfFlight);
          gem_sh_global_eta[muonSel]->Fill( roll+ npart + 0);
        }
        else {
          LogDebug("MuonGEMHitsValidation")<<"+++ Error : layer : "<<layer<<std::endl;
	}
      }
      else if ( region == 1 ) {
        gem_sh_zr_rp1[all]->Fill(g_z,g_r);
        gem_sh_zr_rp1[muonSel]->Fill(g_z,g_r);
        if ( layer == 1 ) {
          gem_sh_tof_rp1_l1[all]->Fill(timeOfFlight);
          gem_sh_global_eta[all]->Fill( roll+ 0 + 2*npart );
          gem_sh_tof_rp1_l1[muonSel]->Fill(timeOfFlight);
          gem_sh_global_eta[muonSel]->Fill( roll+ 0 + 2*npart );
        }
        else if ( layer == 2 ) {
          gem_sh_tof_rp1_l2[all]->Fill(timeOfFlight);
          gem_sh_global_eta[all]->Fill( roll+ npart + 2*npart );
          gem_sh_tof_rp1_l2[muonSel]->Fill(timeOfFlight);
          gem_sh_global_eta[muonSel]->Fill( roll+ npart + 2*npart );
        }
        else {
          LogDebug("MuonGEMHitsValidation")<<"+++ Error : layer : "<<layer<<std::endl;
        }
      }
      else {
        LogDebug("MuonGEMHitsValidation")<<"+++ Error : region : "<<region<<std::endl;
      }
   }
}

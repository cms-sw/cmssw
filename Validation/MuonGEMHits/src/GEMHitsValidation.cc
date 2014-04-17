#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>

GEMHitsValidation::GEMHitsValidation(DQMStore* dbe, const edm::InputTag & inputTag)
:  GEMBaseValidation(dbe, inputTag)
{}

void GEMHitsValidation::bookHisto(const GEMGeometry* gem_geo) {
  theGEMGeometry = gem_geo;
  LogDebug("MuonGEMHitsValidation")<<"Info : Loading Geometry information\n";
  dbe_->setCurrentFolder("MuonGEMHitsV/GEMHitsTask");

  Int_t nregion  = theGEMGeometry->regions().size();
  Int_t nstation = theGEMGeometry->regions()[0]->stations().size() ;
  npart    = theGEMGeometry->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions().size();
  edm::LogInfo("MuonGEMHitsValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;
  edm::LogInfo("MuonGEMHitsValidation")<<"+++ Info : # of stations : "<<nstation<<std::endl;
  edm::LogInfo("MuonGEMHitsValidation")<<"+++ Info : # of eta partition : "<< npart <<std::endl;

  std::vector< std::string > region;
  std::vector< std::string > station;

  if ( nregion == 2) { region.push_back("-1"); region.push_back("1"); }
  else edm::LogWarning("MuonGEMHitsValidation")<<"+++ Error : # of regions is not 2!\n";

  if ( nstation == 1 ) { station.push_back("1"); } 
  else if ( nstation == 3 ) { station.push_back("1"); station.push_back("2"); station.push_back("3"); } 
  else edm::LogWarning("MuonGEMHitsValidation")<<"+++ Error : # of stations is not 1 or 3.\n"; 

  std::string layer[2]= { "1","2" } ;
  std::string cham_loc[2] = {"even","odd"}; 

  LogDebug("MuonGEMHitsValidation")<<"+++ Info : finish to get geometry information from ES.\n";

  for( unsigned int region_num = 0 ; region_num < region.size() ; region_num++ ) {
    for( unsigned int station_num = 0 ; station_num < station.size() ; station_num++) {
      for( unsigned int layer_num = 0 ; layer_num < 2 ; layer_num++) {
        std::string hist_name_for_zr  = std::string("gem_sh_zr_r")+region[region_num]+"_st"+station[station_num]+"_l"+layer[layer_num]+"_";
        std::string hist_label_for_zr = "SimHit occupancy : region"+region[region_num]+" station "+station[station_num]+" layer "+layer[layer_num]+" "+" ; globalZ [cm]; globalR[cm]";
        if ( region_num == 0 ) gem_sh_zr[region_num][station_num][layer_num] = dbe_->book2D( hist_name_for_zr.c_str(), hist_label_for_zr.c_str(), 200, -573,-564,110,130,240);
        else gem_sh_zr[region_num][station_num][layer_num] = dbe_->book2D( hist_name_for_zr.c_str(), hist_label_for_zr.c_str(), 200, 564,573,110,130,240);

        for( unsigned int cham_loc_num = 0 ; cham_loc_num < 2 ; cham_loc_num++) {
          std::string hist_name_for_xy  = std::string("gem_sh_xy_r")+region[region_num]+"_st"+station[station_num]+"_l"+layer[layer_num]+"_"+cham_loc[cham_loc_num];
          std::string hist_name_for_tof  = std::string("gem_sh_tof_r")+region[region_num]+"_st"+station[station_num]+"_l"+layer[layer_num]+"_"+cham_loc[cham_loc_num];
          std::string hist_name_for_eloss  = std::string("gem_sh_energyloss_r")+region[region_num]+"_st"+station[station_num]+"_l"+layer[layer_num]+"_"+cham_loc[cham_loc_num];
          std::string hist_label_for_xy = "SimHit occupancy : region"+region[region_num]+" station "+station[station_num]+" layer "+layer[layer_num]+" "+cham_loc[cham_loc_num]+" ; globalX [cm]; globalY[cm]";
          std::string hist_label_for_tof = "SimHit TOF : region"+region[region_num]+" station "+station[station_num]+" layer "+layer[layer_num]+" "+cham_loc[cham_loc_num]+" ; Time of flight [ns] ; entries";
          std::string hist_label_for_eloss = "SimHit energy loss : region"+region[region_num]+" station "+station[station_num]+" layer "+layer[layer_num]+" "+cham_loc[cham_loc_num]+" ; Energy loss [eV] ; entries";
          gem_sh_xy[region_num][station_num][layer_num][cham_loc_num] = dbe_->book2D( hist_name_for_xy.c_str(), hist_label_for_xy.c_str(), 100, -260,260,100,-260,260);
          gem_sh_tof[region_num][station_num][layer_num][cham_loc_num] = dbe_->book1D( hist_name_for_tof.c_str(), hist_label_for_tof.c_str(), 40,18,22);
          gem_sh_eloss[region_num][station_num][layer_num][cham_loc_num] = dbe_->book1D( hist_name_for_eloss.c_str(), hist_label_for_eloss.c_str(), 60,0.,6000.);
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
    return ;
  }

  for (auto hits=GEMHits->begin(); hits!=GEMHits->end(); hits++) {
    Float_t energyLoss = hits->energyLoss();
    Float_t timeOfFlight = hits->timeOfFlight();
    
    const GEMDetId id(hits->detUnitId());
    Int_t region = id.region();
    Int_t station = id.station();
    Int_t layer = id.layer();
    Int_t even_odd = id.chamber()%2;
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
    // First, fill variable has no condition.
    gem_sh_zr[(int)(region/2.+0.5)][station-1][layer-1]->Fill(g_z,g_r);
    gem_sh_xy[(int)(region/2.+0.5)][station-1][layer-1][even_odd]->Fill(g_x,g_y);
    gem_sh_tof[(int)(region/2.+0.5)][station-1][layer-1][even_odd]->Fill(timeOfFlight);
    gem_sh_eloss[(int)(region/2.+0.5)][station-1][layer-1][even_odd]->Fill(energyLoss);
   }
}

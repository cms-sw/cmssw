#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <TMath.h>

GEMHitsValidation::GEMHitsValidation(const edm::ParameterSet& cfg):  GEMBaseValidation(cfg)
{
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
   detailPlot_ = cfg.getParameter<bool>("detailPlot");
}

void GEMHitsValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
   edm::ESHandle<GEMGeometry> hGeom;
   iSetup.get<MuonGeometryRecord>().get(hGeom);
   const GEMGeometry* GEMGeometry_ =( &*hGeom);
  
  LogDebug("MuonGEMHitsValidation")<<"Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask");

  unsigned int nregion  = GEMGeometry_->regions().size();
  unsigned int nstation = GEMGeometry_->regions()[0]->stations().size() ;
   

  npart    = GEMGeometry_->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions().size();
  edm::LogInfo("MuonGEMHitsValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;
  edm::LogInfo("MuonGEMHitsValidation")<<"+++ Info : # of stations : "<<nstation<<std::endl;
  edm::LogInfo("MuonGEMHitsValidation")<<"+++ Info : # of eta partition : "<< npart <<std::endl;

  LogDebug("MuonGEMHitsValidation")<<"+++ Info : finish to get geometry information from ES.\n";

  for( auto& region : GEMGeometry_->regions() ){
    int region_num = region->region();
    TString title_suffix = TString::Format(" at Region%d", region_num);
    TString histname_suffix = TString::Format("_r%d", region_num);
    TString simpleZR_title = TString::Format("ZR Occupancy%s; |Z|(cm); R(cm)", title_suffix.Data());
    TString simpleZR_histname = TString::Format("hit_simple_zr%s", histname_suffix.Data());
    Hit_simple_zr[simpleZR_histname.Hash()] = ibooker.book2D(simpleZR_histname, simpleZR_title, 100, 550, 820, 100, 100, 300);
    for( auto& station : region->stations() ){
      if( station->station()==2 ) continue;
      int station_num = ( station->station()==1 ) ? 1 : 2;
      // TOF and Energy loss part are indepent from Region.
      if ( region_num == 1 ) {
        // Labeling TOF and Energy loss
        TString hist_name_for_tofMu  = TString::Format("gem_sh_simple_tofMuon_st%d",station_num);
        TString hist_name_for_elossMu  = TString("gem_sh_simple_energylossMuon_st%d",+station_num);
        TString hist_label_for_tofMu = TString("SimHit TOF(Muon only) station : station %d ; Time of flight [ns] ; entries",station_num);
        TString hist_label_for_elossMu = TString("SimHit energy loss(Muon only) : station %d ; Energy loss [eV] ; entries",station_num);
        // Booking
        double tof_min, tof_max;
        if( station_num == 0 ) { tof_min = 18; tof_max = 22; }
        else  { tof_min = 26; tof_max = 30; }
        gem_sh_simple_tofMu[ hist_name_for_tofMu.Hash()] = ibooker.book1D( hist_name_for_tofMu.Data(), hist_label_for_tofMu.Data(), 40,tof_min,tof_max);
        gem_sh_simple_elossMu[ hist_name_for_elossMu.Hash()] = ibooker.book1D( hist_name_for_elossMu.Data(), hist_label_for_elossMu.Data(), 60,0.,6000.);
      }

      TString title_suffix2 = title_suffix + TString::Format("  Station%d", station_num);
      TString histname_suffix2 = histname_suffix + TString::Format("_st%d", station_num);
      TString dcEta_title = TString::Format("Occupancy for detector component %s;;#eta-partition",title_suffix2.Data());
      TString dcEta_histname = TString::Format("hit_dcEta%s",histname_suffix2.Data());

      int nXbins = station->rings()[0]->nSuperChambers()*2;
      int nRoll1 = station->rings()[0]->superChambers()[0]->chambers()[0]->etaPartitions().size();
      int nRoll2 = station->rings()[0]->superChambers()[0]->chambers()[1]->etaPartitions().size();
      int nYbins = ( nRoll1 > nRoll2 ) ? nRoll1 : nRoll2;
  
      Hit_dcEta[ dcEta_histname.Hash() ] = ibooker.book2D(dcEta_histname, dcEta_title, nXbins, 0, nXbins, nYbins, 1, nYbins+1);
    
      int idx = 0;
      for(unsigned int sCh = 1; sCh <= station->superChambers().size(); sCh++){
        for(unsigned int Ch = 1; Ch <= 2; Ch++){
          idx++;
          TString label = TString::Format("ch%d_la%d", sCh, Ch);
          Hit_dcEta[ dcEta_histname.Hash() ]->setBinLabel(idx, label.Data());
        }
      }
    }
  }


  if( detailPlot_ ){
    for( auto& region : GEMGeometry_->regions() ){
      for( auto& station : region->stations() ){    
        for( auto& ring : station->rings()){
          GEMDetId id;
          if ( ring->station() == 2 ) continue;

          if ( ring->ring() != 1 ) break ; // Only Ring1 is interesting.
          std::stringstream hist_name;
          hist_name<<"gem_sh_xy_r"<<region->region()<<"_st"<<stationLabel[station->station()-1]<<"_";
          std::stringstream hist_title;
          hist_title<<"Simhit Global XY Plots at "<<id<<" on ";
          MonitorElement* temp = ibooker.book2D( (hist_name.str()+"even").c_str(), (hist_title.str()+"even").c_str(),nBinXY_,-360,360,nBinXY_,-360,360);
          if ( temp != nullptr ) {
            LogDebug("MuonGEMHitsValidation")<<"ME can be acquired!";
          }
          else {
            LogDebug("MuonGEMHitsValidation")<<"ME can not be acquired!";
            return ;
          }
          gem_sh_xy_st_ch.insert( std::map<std::string, MonitorElement*>::value_type( hist_name.str()+"even", temp));
          MonitorElement* temp2 = ibooker.book2D( (hist_name.str()+"odd").c_str(), (hist_title.str()+"odd").c_str(),nBinXY_,-360,360,nBinXY_,-360,360);
          if ( temp2 != nullptr ) {
            LogDebug("MuonGEMHitsValidation")<<"ME can be acquired!";
          }
          else {
            LogDebug("MuonGEMHitsValidation")<<"ME can not be acquired!";
            return ;
          }
          gem_sh_xy_st_ch.insert( std::map<std::string, MonitorElement*>::value_type( hist_name.str()+"odd", temp2));
    //std::cout<<hist_name.str()<<std::endl;
        }
      }
    }

    for( unsigned int region_num = 0 ; region_num < nregion ; region_num++ ) {
      for( unsigned int station_num = 0 ; station_num < nstation ; station_num++) {
        for( unsigned int layer_num = 0 ; layer_num < 2 ; layer_num++) {
          if ( station_num ==1 ) continue;
          gem_sh_zr[region_num][station_num][layer_num] = BookHistZR(ibooker,"gem_sh","SimHit",region_num,station_num,layer_num);
          gem_sh_xy[region_num][station_num][layer_num] = BookHistXY(ibooker,"gem_sh","SimHit",region_num,station_num,layer_num);
          std::string hist_name_for_tof  = std::string("gem_sh_tof_r")+regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
          std::string hist_name_for_tofMu  = std::string("gem_sh_tofMuon_r")+regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
          std::string hist_name_for_eloss  = std::string("gem_sh_energyloss_r")+regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
          std::string hist_name_for_elossMu  = std::string("gem_sh_energylossMuon_r")+regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
          std::string hist_label_for_xy = "SimHit occupancy : region"+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" ; globalX [cm]; globalY[cm]";
          std::string hist_label_for_tof = "SimHit TOF : region"+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; Time of flight [ns] ; entries";
          std::string hist_label_for_tofMu = "SimHit TOF(Muon only) : region"+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; Time of flight [ns] ; entries";
          std::string hist_label_for_eloss = "SimHit energy loss : region"+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; Energy loss [eV] ; entries";
          std::string hist_label_for_elossMu = "SimHit energy loss(Muon only) : region"+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; Energy loss [eV] ; entries";
        
          double tof_min, tof_max;
          if( station_num == 0 ) { tof_min = 18; tof_max = 22; }
          else  { tof_min = 26; tof_max = 30; }
          gem_sh_tof[region_num][station_num][layer_num] = ibooker.book1D( hist_name_for_tof.c_str(), hist_label_for_tof.c_str(), 40,tof_min,tof_max);
          gem_sh_tofMu[region_num][station_num][layer_num] = ibooker.book1D( hist_name_for_tofMu.c_str(), hist_label_for_tofMu.c_str(), 40,tof_min,tof_max);
          gem_sh_eloss[region_num][station_num][layer_num] = ibooker.book1D( hist_name_for_eloss.c_str(), hist_label_for_eloss.c_str(), 60,0.,6000.);
          gem_sh_elossMu[region_num][station_num][layer_num] = ibooker.book1D( hist_name_for_elossMu.c_str(), hist_label_for_elossMu.c_str(), 60,0.,6000.);
        }
      }
    }
  }
}


GEMHitsValidation::~GEMHitsValidation() {
}


void GEMHitsValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry* GEMGeometry_ =( &*hGeom);
  edm::Handle<edm::PSimHitContainer> GEMHits;
  e.getByToken(InputTagToken_, GEMHits);
  if (!GEMHits.isValid()) {
    edm::LogError("GEMHitsValidation") << "Cannot get GEMHits by Token simInputTagToken";
    return ;
  }

  Float_t timeOfFlightMuon = 0.;
  Float_t energyLossMuon = 0;

  for (auto hits=GEMHits->begin(); hits!=GEMHits->end(); hits++) {

    const GEMDetId id(hits->detUnitId());
    Int_t region = (Int_t) id.region();
    Int_t station = (Int_t) id.station();
    Int_t layer = (Int_t) id.layer();
    Int_t chamber = (Int_t) id.chamber();
    Int_t nroll = (Int_t) id.roll();

    if ( station == 2 ) continue;


    //Int_t even_odd = id.chamber()%2;
    if ( GEMGeometry_->idToDet(hits->detUnitId()) == nullptr) {
      std::cout<<"simHit did not matched with GEMGeometry."<<std::endl;
      continue;
    }
    //const LocalPoint p0(0., 0., 0.);
    //const GlobalPoint Gp0(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(p0));
    const LocalPoint hitLP(hits->localPosition());
    
    const GlobalPoint hitGP(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
    Float_t g_r = hitGP.perp();
    Float_t g_x = hitGP.x();
    Float_t g_y = hitGP.y();
    Float_t g_z = hitGP.z();
    Float_t energyLoss = hits->energyLoss();
    Float_t timeOfFlight = hits->timeOfFlight();

//    int region_num = 0;
//    if( region == -1 ) region_num = 0;
//    else if ( region == 1 ) region_num = 1;
    int layer_num = layer-1;
    int binX = (chamber-1)*2+layer_num;
    int binY = nroll;
//    int station_num = station-1;
    if ( station == 2 ) continue;
    if ( station == 3 ) station = 2;

    //const LocalPoint hitEP(hits->entryPoint());
    
    TString histname_suffix = TString::Format("_r%d", region);
    TString simple_zr_histname = TString::Format("hit_simple_zr%s",histname_suffix.Data());
    Hit_simple_zr[ simple_zr_histname.Hash() ] ->Fill(fabs(g_z), g_r);

    histname_suffix = TString::Format("_r%d_st%d", region, station);
    TString dcEta_histname = TString::Format("hit_dcEta%s", histname_suffix.Data());
    Hit_dcEta[ dcEta_histname.Hash() ]->Fill(binX, binY);

    TString tofMu = TString::Format("gem_sh_simple_tofMuon_st%d",station);
    TString elossMu = TString::Format("gem_sh_simple_elossMuon_st%d",station);
    gem_sh_simple_tofMu[ tofMu.Hash() ]->Fill( timeOfFlight );
    gem_sh_simple_elossMu[ elossMu.Hash() ]->Fill( energyLoss );

    if( detailPlot_ ){
//move these here in order to use a GEMDetId - layers, station...
      if (abs(hits-> particleType()) == 13){
        timeOfFlightMuon = hits->timeOfFlight();
        energyLossMuon = hits->energyLoss();
//fill histos for Muons only
        gem_sh_tofMu[(int)(region/2.+0.5)][station-1][layer-1]->Fill(timeOfFlightMuon);
        gem_sh_elossMu[(int)(region/2.+0.5)][station-1][layer-1]->Fill(energyLossMuon*1.e9);
      }
    
      // fill hist
      // First, fill variable has no condition.
      gem_sh_zr[(int)(region/2.+0.5)][station-1][layer-1]->Fill(g_z,g_r);
      gem_sh_xy[(int)(region/2.+0.5)][station-1][layer-1]->Fill(g_x,g_y);
      gem_sh_tof[(int)(region/2.+0.5)][station-1][layer-1]->Fill(timeOfFlight);
      gem_sh_eloss[(int)(region/2.+0.5)][station-1][layer-1]->Fill(energyLoss*1.e9);

      std::string chamber ="";
      if ( id.chamber() %2 == 1 ) chamber = "odd";
      else  chamber = "even";
      std::stringstream hist_name;
      hist_name<<"gem_sh_xy_r"<<id.region()<<"_st"<<stationLabel[id.station()-1]<<"_"<<chamber;
      //std::cout<<hist_name.str()<<std::endl;
      gem_sh_xy_st_ch[hist_name.str()]->Fill( g_x, g_y); 
    }
  }
}


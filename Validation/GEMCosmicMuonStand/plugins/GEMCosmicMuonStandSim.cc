#include "GEMCosmicMuonStandSim.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace std;

GEMCosmicMuonStandSim::GEMCosmicMuonStandSim(const edm::ParameterSet& cfg)
{
  simHitToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simHitToken"));
  recHitToken_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitToken"));
}

MonitorElement* GEMCosmicMuonStandSim::BookHist1D( DQMStore::IBooker &ibooker, const char* name, const char* label, unsigned int row, unsigned int coll, unsigned int layer_num, unsigned int vfat_num, const unsigned int Nbin, const Float_t xMin, const Float_t xMax)
{
  string hist_name,hist_label;
  return ibooker.book1D( hist_name, hist_label,Nbin,xMin,xMax ); 
}

MonitorElement* GEMCosmicMuonStandSim::BookHist1D( DQMStore::IBooker &ibooker, const char* name, const char* label, const unsigned int Nbin, const Float_t xMin, const Float_t xMax)
{
  return ibooker.book1D( name, label,Nbin,xMin,xMax );
}

void GEMCosmicMuonStandSim::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup )
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  LogDebug("GEMCosmicMuonStandSim")<<"Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("GEMCosmicMuonStandSim");
  LogDebug("GEMCosmicMuonStandSim")<<"ibooker set current folder\n";

  // for( auto& region : GEMGeometry_->regions() ){
  //   int re = region->region();
  // }
  
  LogDebug("GEMCosmicMuonStandSim")<<"Booking End.\n";
}

void GEMCosmicMuonStandSim::analyze(const edm::Event& e,const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  edm::Handle<GEMRecHitCollection> gemRecHits;
  e.getByToken( recHitToken_, gemRecHits);
  edm::Handle<edm::PSimHitContainer> gemSimHits;
  e.getByToken( simHitToken_, gemSimHits);
  
  if (!gemRecHits.isValid()) {
    edm::LogError("GEMCosmicMuonStandSim") << "Cannot get strips by Token RecHits Token.\n";
    return ;
  }
  
  // for (edm::PSimHitContainer::const_iterator hits = gemSimHits->begin(); hits!=gemSimHits->end(); ++hits) {
  //   const GEMDetId id(hits->detUnitId());

  //   Int_t sh_region = id.region();
  //   //Int_t sh_ring = id.ring();
  //   Int_t sh_roll = id.roll();
  //   Int_t sh_station = id.station();
  //   Int_t sh_layer = id.layer();
  //   Int_t sh_chamber = id.chamber();

  //   if ( GEMGeometry_->idToDet(hits->detUnitId()) == nullptr) {
  //     std::cout<<"simHit did not matched with GEMGeometry."<<std::endl;
  //     continue;
  //   }

  //   if (!(abs(hits-> particleType()) == 13)) continue;

  //   //const LocalPoint p0(0., 0., 0.);
  //   //const GlobalPoint Gp0(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(p0));
  //   const LocalPoint hitLP(hits->localPosition());

  //   const LocalPoint hitEP(hits->entryPoint());
  //   Int_t sh_strip = GEMGeometry_->etaPartition(hits->detUnitId())->strip(hitEP);

  //   //const GlobalPoint hitGP(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
  //   //Float_t sh_l_r = hitLP.perp();
  //   Float_t sh_l_x = hitLP.x();
  //   Float_t sh_l_y = hitLP.y();
  //   //Float_t sh_l_z = hitLP.z();

  //   for (GEMRecHitCollection::const_iterator recHit = gemRecHits->begin(); recHit != gemRecHits->end(); ++recHit){
  //     Float_t  rh_l_x = recHit->localPosition().x();
  //     Float_t  rh_l_xErr = recHit->localPositionError().xx();
  //     Float_t  rh_l_y = recHit->localPosition().y();
  //     Float_t  rh_l_yErr = recHit->localPositionError().yy();
  //     //Int_t  detId = (Short_t) (*recHit).gemId();
  //     //Int_t  bx = recHit->BunchX();
  //     Int_t  clusterSize = recHit->clusterSize();
  //     Int_t  firstClusterStrip = recHit->firstClusterStrip();

  //     GEMDetId id((*recHit).gemId());

  //     Short_t rh_region = (Short_t) id.region();
  //     //Int_t rh_ring = (Short_t) id.ring();
  //     Short_t rh_station = (Short_t) id.station();
  //     Short_t rh_layer = (Short_t) id.layer();
  //     Short_t rh_chamber = (Short_t) id.chamber();
  //     Short_t rh_roll = (Short_t) id.roll();

  //     LocalPoint recHitLP = recHit->localPosition();
  //     if ( GEMGeometry_->idToDet((*recHit).gemId()) == nullptr) {
  //       std::cout<<"This gem recHit did not matched with GEMGeometry."<<std::endl;
  //       continue;
  //     }
  //     GlobalPoint recHitGP = GEMGeometry_->idToDet((*recHit).gemId())->surface().toGlobal(recHitLP);

  //     Float_t     rh_g_R = recHitGP.perp();
  //     //Float_t rh_g_Eta = recHitGP.eta();
  //     //Float_t rh_g_Phi = recHitGP.phi();
  //     Float_t     rh_g_X = recHitGP.x();
  //     Float_t     rh_g_Y = recHitGP.y();
  //     Float_t     rh_g_Z = recHitGP.z();
  //     Float_t   rh_pullX = (Float_t)(rh_l_x - sh_l_x)/(rh_l_xErr);
  //     Float_t   rh_pullY = (Float_t)(rh_l_y - sh_l_y)/(rh_l_yErr);

  //     std::vector<int> stripsFired;
  //     for(int i = firstClusterStrip; i < (firstClusterStrip + clusterSize); i++){
  //       stripsFired.push_back(i);
  //     }

  //     const bool cond1( sh_region == rh_region and sh_layer == rh_layer and sh_station == rh_station);
  //     const bool cond2(sh_chamber == rh_chamber and sh_roll == rh_roll);
  //     const bool cond3(std::find(stripsFired.begin(), stripsFired.end(), (sh_strip + 1)) != stripsFired.end());

  //     if(cond1 and cond2 and cond3){
  //       LogDebug("GEMCosmicMuonStandSim")<< " Region : " << rh_region << "\t Station : " << rh_station
  //         << "\t Layer : "<< rh_layer << "\n Radius: " << rh_g_R << "\t X : " << rh_g_X << "\t Y : "<< rh_g_Y << "\t Z : " << rh_g_Z << std::endl;	

  //       // int region_num=0 ;
  //       // if ( rh_region ==-1 ) region_num = 0 ;
  //       // else if ( rh_region==1) region_num = 1;
  //       int layer_num = rh_layer-1;
  //       int binX = (rh_chamber-1)*2+layer_num;
  //       int binY = rh_roll;
  //       int station_num = rh_station -1;

  //       // Fill normal plots.
  //       TString histname_suffix = TString::Format("_r%d",rh_region);
  //       TString simple_zr_histname = TString::Format("rh_simple_zr%s",histname_suffix.Data());
  //       LogDebug("GEMCosmicMuonStandSim")<< " simpleZR!\n";
  //       recHits_simple_zr[simple_zr_histname.Hash()]->Fill( fabs(rh_g_Z), rh_g_R);

  //       histname_suffix = TString::Format("_r%d_st%d",rh_region, rh_station);
  //       TString dcEta_histname = TString::Format("rh_dcEta%s",histname_suffix.Data());
  //       LogDebug("GEMCosmicMuonStandSim")<< " dcEta\n";
  //       recHits_dcEta[dcEta_histname.Hash()]->Fill( binX, binY);

  //       gem_cls_tot->Fill(clusterSize);
  //       gem_region_pullX[0]->Fill(rh_pullX);
  //       gem_region_pullY[0]->Fill(rh_pullY);
  //       LogDebug("GEMCosmicMuonStandSim")<< " Begin detailPlot!\n";

  //     }
  //   } //End loop on RecHits
  // } //End loop on SimHits

}

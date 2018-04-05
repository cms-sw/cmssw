#include "GEMCosmicMuonStandEfficiency.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <iostream>

using namespace std;

GEMCosmicMuonStandEfficiency::GEMCosmicMuonStandEfficiency(const edm::ParameterSet& cfg)
{
  insideOutTracks_ = consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("insideOutTracks"));
  outsideInTracks_ = consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("outsideInTracks"));
}

MonitorElement* GEMCosmicMuonStandEfficiency::BookHist1D( DQMStore::IBooker &ibooker, const char* name, const char* label, unsigned int row, unsigned int coll, unsigned int layer_num, unsigned int vfat_num, const unsigned int Nbin, const Float_t xMin, const Float_t xMax)
{
  string hist_name = name+row+coll+layer_num+vfat_num;
  string hist_label;
  // hist_name.Format("{} {} {} {} {}", name, row, coll, layer_num, vfat_num);
  // hist_label.Format("{} {} {} {} {}", label, row, coll, layer_num, vfat_num);
  return ibooker.book1D( hist_name, hist_label,Nbin,xMin,xMax ); 
}

MonitorElement* GEMCosmicMuonStandEfficiency::BookHist1D( DQMStore::IBooker &ibooker, const char* name, const char* label, const unsigned int Nbin, const Float_t xMin, const Float_t xMax)
{
  return ibooker.book1D( name, label,Nbin,xMin,xMax );
}

void GEMCosmicMuonStandEfficiency::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup )
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  LogDebug("GEMCosmicMuonStandEfficiency")<<"Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("GEMCosmicMuonStandEfficiency");
  LogDebug("GEMCosmicMuonStandEfficiency")<<"ibooker set current folder\n";

  // for( auto& region : GEMGeometry_->regions() ){
  //   int re = region->region();
  // }
  for(int i = 0; i < 30; i++)
  {
    std::string temp1 = "vfatValidChamber" + to_string(i+1); 
    gem_vfat_eff[i] = BookHist1D(ibooker, temp1.c_str(), temp1.c_str(), 24, -0.5, 23.5);
    std::string temp2 = "vfatInValidChamber" + to_string(i+1);
    gem_vfat_tot[i] = BookHist1D(ibooker, temp2.c_str(), temp2.c_str(), 24, -0.5, 23.5);
  }
  gem_vfat_total_eff = BookHist1D(ibooker, "vfatTotChambers", "vfatTotChambers", 24, -0.5, 23.5);
  

  isuperChamber = BookHist1D(ibooker, "superChamber", "superChamber", 15, 0.5, 15.5);
  ilayers = BookHist1D(ibooker, "layers", "layers", 2, 0.5, 2.5);
  ichamber = BookHist1D(ibooker, "chamber", "chamber", 30, 0.5, 30.5);
  iroll = BookHist1D(ibooker, "roll", "roll", 8, 0.5, 8.5);
  ipartition = BookHist1D(ibooker, "partition", "partition", 3, -0.5, 2.5);
  ichi2 = BookHist1D(ibooker, "chi2", "chi2", 100, 0, 10);

  LogDebug("GEMCosmicMuonStandEfficiency")<<"Booking End.\n";
}

void GEMCosmicMuonStandEfficiency::analyze(const edm::Event& e,const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  edm::Handle<reco::TrackCollection> insideOutTracks;
  e.getByToken( insideOutTracks_, insideOutTracks);

  edm::Handle<reco::TrackCollection> outsideInTracks;
  e.getByToken( outsideInTracks_, outsideInTracks);
//Test Lines
  
  for (std::vector<reco::Track>::const_iterator track = insideOutTracks->begin(); track != insideOutTracks->end(); ++track)
  {
    cout << "track->recHitsSize() "<< track->recHitsSize() <<endl;
    
    ichi2->Fill(track->chi2());
    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      auto rawId = (*recHit)->rawId();
      auto etaPartition = GEMGeometry_->etaPartition(rawId);
      auto superChamber = GEMGeometry_->superChamber(rawId);

      int chamber = superChamber->id().chamber();
      int layer = etaPartition->id().layer();
      int roll = etaPartition->id().roll();
      int nStrips = etaPartition->nstrips();
      float strip = etaPartition->strip((*recHit)->localPosition());

      isuperChamber->Fill(chamber/2);
      ilayers->Fill(layer);
      ichamber->Fill(chamber + layer - 1);
      iroll->Fill(roll);
      ipartition->Fill(int(strip*3/nStrips));
      
      int iChamber = chamber+layer-2;
      int vfat = (roll-1)+int(strip/nStrips*3)*8;
      
      gem_vfat_total_eff->Fill(vfat);
      if((*recHit)->isValid()) gem_vfat_eff[iChamber]->Fill(vfat);
      else gem_vfat_tot[iChamber]->Fill(vfat);
      
    }
 
 }
  for (std::vector<reco::Track>::const_iterator track = outsideInTracks->begin(); track != outsideInTracks->end(); ++track)
  {
    ichi2->Fill(track->chi2());

//// Below lines have error
    cout << "track->recHitsSize() "<< track->recHitsSize() <<endl;
    // auto seed = track->seedRef();
    // auto seedhit = seed->recHits().first;
    // cout << "first  gp "<< GEMDetId(seedhit->rawId()) <<endl;
    // seedhit++;
    // cout << "second gp "<< GEMDetId(seedhit->rawId()) <<endl;
    
    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      auto rawId = (*recHit)->rawId();

      auto etaPartition = GEMGeometry_->etaPartition(rawId);
      auto superChamber = GEMGeometry_->superChamber(rawId);

      int chamber = superChamber->id().chamber();
      int layer = etaPartition->id().layer();
      int roll = etaPartition->id().roll();
      int nStrips = etaPartition->nstrips();
      float strip = etaPartition->strip((*recHit)->localPosition());
 
      isuperChamber->Fill(chamber/2);
      ilayers->Fill(layer);
      ichamber->Fill(chamber + layer - 1);
      iroll->Fill(roll);
      ipartition->Fill(int(strip*3/nStrips));
      
      int iChamber = chamber+layer-2;
      int vfat = (roll-1)+int(strip/nStrips*3)*8;
      
      if((*recHit)->isValid()) gem_vfat_eff[iChamber]->Fill(vfat);
      else gem_vfat_tot[iChamber]->Fill(vfat);
       
    }
  }
}

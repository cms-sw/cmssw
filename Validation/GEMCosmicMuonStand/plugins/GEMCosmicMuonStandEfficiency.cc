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
  seedInside_ = consumes<vector<TrajectorySeed>>(cfg.getParameter<edm::InputTag>("insideOutTracks"));
  seedOutside_ = consumes<vector<TrajectorySeed>>(cfg.getParameter<edm::InputTag>("outsideInTracks"));
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
  for(int i = 0; i < 15; i++)
  {
    std::string temp1;
    for(int j = 0; j < 2; j++)
    {
      temp1 = "vfat_Valid_Hit_Chamber" + to_string(2*i+1) + "_layer" + to_string(j+1) + "_inside";
      gem_vfat_eff[i][j][0] = BookHist1D(ibooker, temp1.c_str(), temp1.c_str(), 24, -0.5, 23.5);
      temp1 = "vfat_Valid_Hit_Chamber" + to_string(2*i+1) + "_layer" + to_string(j+1) + "_outside"; 
      gem_vfat_eff[i][j][1] = BookHist1D(ibooker, temp1.c_str(), temp1.c_str(), 24, -0.5, 23.5);
      temp1 = "vfat_Total_Hit_Chamber" + to_string(2*i+1) + "_layer" + to_string(j+1) +"_inside";
      gem_vfat_tot[i][j][0] = BookHist1D(ibooker, temp1.c_str(), temp1.c_str(), 24, -0.5, 23.5);
      temp1 = "vfat_Total_Hit_Chamber" + to_string(2*i+1) + "_layer" + to_string(j+1) + "_outside";
      gem_vfat_tot[i][j][1] = BookHist1D(ibooker, temp1.c_str(), temp1.c_str(), 24, -0.5, 23.5);
    }
  }

  insideChi2 = BookHist1D(ibooker, "insideChi2", "insideChi2", 1000, 0, 100);
  outsideChi2 = BookHist1D(ibooker, "outsideChi2", "outsideChi2", 1000, 0, 100);

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

  edm::Handle<vector<TrajectorySeed>> seedInside;
  e.getByToken( seedInside_, seedInside);

  edm::Handle<vector<TrajectorySeed>> seedOutside;
  e.getByToken( seedOutside_, seedOutside);


  // Analysis inside out tracks 
  for (std::vector<reco::Track>::const_iterator track = insideOutTracks->begin(); track != insideOutTracks->end(); ++track)
  {
    vector<TrajectorySeed>::const_iterator seeds = seedInside->begin();
    auto seed = ((*seeds).recHits()).first;
    GEMDetId firstHit(seed->rawId());
    seed++;
    GEMDetId secondHit(seed->rawId());

    //if((firstHit.chamber()-1)/10 != (secondHit.chamber()-1)/10) continue;
    if(track->normalizedChi2() > 1.) continue;
    
    insideChi2->Fill(track->normalizedChi2());

    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      GEMDetId gemId((*recHit)->rawId());
      if(gemId.chamber() == firstHit.chamber() and gemId.layer() == firstHit.layer()) continue;
      if(gemId.chamber() == secondHit.chamber() and gemId.layer() == secondHit.layer()) continue;
      
      auto etaPartition = GEMGeometry_->etaPartition(gemId);
      int chamber = etaPartition->id().chamber();
      int layer = etaPartition->id().layer();
      int roll = etaPartition->id().roll();
      int nStrips = etaPartition->nstrips();
      float strip = etaPartition->strip((*recHit)->localPosition());

      int idxChamber = (chamber-1)/2;
      int idxLayer = layer-1;
      int vfat = (8-roll)+int(strip/nStrips*3)*8;
      
      if((*recHit)->isValid()) gem_vfat_eff[idxChamber][idxLayer][0]->Fill(vfat);
      gem_vfat_tot[idxChamber][idxLayer][0]->Fill(vfat);
    }
  }
 


  // Analysis outside in tracks 
  for (std::vector<reco::Track>::const_iterator track = outsideInTracks->begin(); track != outsideInTracks->end(); ++track)
  {
    vector<TrajectorySeed>::const_iterator seeds = seedOutside->begin();
    auto seed = ((*seeds).recHits()).first;
    GEMDetId firstHit(seed->rawId());
    seed++;
    GEMDetId secondHit(seed->rawId());

    //if((firstHit.chamber()-1)/10 != (secondHit.chamber()-1)/10) continue;
    if(track->normalizedChi2() > 1.) continue;

    outsideChi2->Fill(track->normalizedChi2());

    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      GEMDetId gemId((*recHit)->rawId());
      if(gemId.chamber() == firstHit.chamber() and gemId.layer() == firstHit.layer()) continue;
      if(gemId.chamber() == secondHit.chamber() and gemId.layer() == secondHit.layer()) continue;
      
      auto etaPartition = GEMGeometry_->etaPartition(gemId);
      int chamber = etaPartition->id().chamber();
      int layer = etaPartition->id().layer();
      int roll = etaPartition->id().roll();
      int nStrips = etaPartition->nstrips();
      float strip = etaPartition->strip((*recHit)->localPosition());

      int idxChamber = (chamber-1)/2;
      int idxLayer = layer-1;
      int vfat = (8-roll)+int(strip/nStrips*3)*8;
      
      if((*recHit)->isValid()) gem_vfat_eff[idxChamber][idxLayer][1]->Fill(vfat);
      gem_vfat_tot[idxChamber][idxLayer][1]->Fill(vfat);
    }
  }
}

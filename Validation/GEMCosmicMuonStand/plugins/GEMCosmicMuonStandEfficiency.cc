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
    for(int j = 0; j < 2; j++)
    {
      std::string temp1 = "vfat_Valid_Hit_Chamber" + to_string(2*i+1) + "_layer" + to_string(j+1); 
      gem_vfat_eff[i][j] = BookHist1D(ibooker, temp1.c_str(), temp1.c_str(), 24, -0.5, 23.5);
      std::string temp2 = "vfat_Total_Hit_Chamber" + to_string(2*i+1) + "_layer" + to_string(j+1);
      gem_vfat_tot[i][j] = BookHist1D(ibooker, temp2.c_str(), temp2.c_str(), 24, -0.5, 23.5);
    }
  }
  gem_vfat_total_eff = BookHist1D(ibooker, "vfatTotHit", "vfatTotHit", 24, -0.5, 23.5);
  

  isuperChamber = BookHist1D(ibooker, "superChamber", "superChamber", 30, 0.5, 30.5);
  ilayers = BookHist1D(ibooker, "layers", "layers", 2, 0.5, 2.5);
  ichamber = BookHist1D(ibooker, "chamber", "chamber", 30, 0.5, 30.5);
  iroll = BookHist1D(ibooker, "roll", "roll", 8, 0.5, 8.5);
  ipartition = BookHist1D(ibooker, "partition", "partition", 3, -0.5, 2.5);

  insideCount = BookHist1D(ibooker, "insideOutRecHits", "insideOutRecHits", 10, -0.5, 9.5);
  outsideCount = BookHist1D(ibooker, "outsideInRecHits", "outsideInRecHits", 10, -0.5, 9.5);

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
<<<<<<< HEAD
    auto secondHit = seed->rawId();

    int count = 0;

    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      count++;
      auto rawId = (*recHit)->rawId();
      //if(rawId == firstHit or rawId == secondHit) continue;
      auto etaPartition = GEMGeometry_->etaPartition(rawId);
      auto superChamber = GEMGeometry_->superChamber(rawId);
=======
    GEMDetId secondHit(seed->rawId());
    
    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      GEMDetId gemId((*recHit)->rawId());
      if(gemId == firstHit or gemId == secondHit) continue;
      
      auto etaPartition = GEMGeometry_->etaPartition(gemId);
>>>>>>> 034ef76b6ecca7a5e6cce28b490792a582d5ea81

      int chamber = etaPartition->id().chamber();
      int layer = etaPartition->id().layer();
      int roll = etaPartition->id().roll();
      int nStrips = etaPartition->nstrips();
      float strip = etaPartition->strip((*recHit)->localPosition());

      if(chamber == 2 or chamber == 9) continue;
      else if(chamber == 12 or chamber == 19) continue;
      else if(chamber == 22 or chamber == 29) continue;
      
      isuperChamber->Fill(chamber);
      ilayers->Fill(layer);
      ichamber->Fill(chamber + layer - 1);
      iroll->Fill(roll);
      ipartition->Fill(int(strip*3/nStrips));
      
      int idxChamber = (chamber-1)/2;
      int idxLayer = layer-1;
      int vfat = (roll-1)+int(strip/nStrips*3)*8;
      
      gem_vfat_total_eff->Fill(vfat);
      if((*recHit)->isValid()) gem_vfat_eff[idxChamber][idxLayer]->Fill(vfat);
      gem_vfat_tot[idxChamber][idxLayer]->Fill(vfat);
    }
    insideCount->Fill(count);
  }


  // Analysis outside in tracks 
  for (std::vector<reco::Track>::const_iterator track = outsideInTracks->begin(); track != outsideInTracks->end(); ++track)
  {
    vector<TrajectorySeed>::const_iterator seeds = seedOutside->begin();
    auto seed = ((*seeds).recHits()).first;
    GEMDetId firstHit(seed->rawId());
    seed++;
<<<<<<< HEAD
    auto secondHit = seed->rawId();
    int count = 0;
    
    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      count++;
      auto rawId = (*recHit)->rawId();
      //if(rawId == firstHit or rawId == secondHit) continue;
      auto etaPartition = GEMGeometry_->etaPartition(rawId);
      auto superChamber = GEMGeometry_->superChamber(rawId);
=======
    GEMDetId secondHit(seed->rawId());
    
    for (trackingRecHit_iterator recHit = track->recHitsBegin(); recHit != track->recHitsEnd(); ++recHit)
    {
      GEMDetId gemId((*recHit)->rawId());
      if (gemId == firstHit or gemId == secondHit) continue;
      if (gemId.chamber() < firstHit.chamber() and gemId.chamber() > secondHit.chamber())
	continue;
      
      cout <<"gemId "<< GEMDetId(gemId)<< endl;
      auto etaPartition = GEMGeometry_->etaPartition(gemId);
>>>>>>> 034ef76b6ecca7a5e6cce28b490792a582d5ea81

      int chamber = etaPartition->id().chamber();
      int layer = etaPartition->id().layer();
      int roll = etaPartition->id().roll();
      int nStrips = etaPartition->nstrips();
      float strip = etaPartition->strip((*recHit)->localPosition());

      if(chamber == 1 or chamber == 10) continue;
      else if(chamber == 11 or chamber == 20) continue;
      else if(chamber == 21 or chamber == 30) continue;
 
      isuperChamber->Fill(chamber);
      ilayers->Fill(layer);
      ichamber->Fill(chamber + layer - 1);
      iroll->Fill(roll);
      ipartition->Fill(int(strip*3/nStrips));
      
      int idxChamber = (chamber-1)/2;
      int idxLayer = layer-1;
      int vfat = (roll-1)+int(strip/nStrips*3)*8;
      
<<<<<<< HEAD
      gem_vfat_total_eff->Fill(vfat);
      if((*recHit)->isValid()) gem_vfat_eff[idxChamber][idxLayer]->Fill(vfat);
      gem_vfat_tot[idxChamber][idxLayer]->Fill(vfat);
=======
      if((*recHit)->isValid()) gem_vfat_eff[iChamber]->Fill(vfat);
      else gem_vfat_tot[iChamber]->Fill(vfat);
      
>>>>>>> 034ef76b6ecca7a5e6cce28b490792a582d5ea81
    }
    outsideCount->Fill(count);
  }
}

// -*- C++ -*-
//
// Package:    GEMDigiAnalyzer
// Class:      GEMDigiAnalyzer
// 
/**\class GEMDigiAnalyzer GEMDigiAnalyzer.cc MyAnalyzers/GEMDigiAnalyzer/src/GEMDigiAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// $Id: GEMDigiAnalyzer.cc,v 1.9 2013/04/23 07:40:17 dildick Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TTree.h"
#include "TFile.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"


///Data Format
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Validation/MuonGEMDigis/interface/SimTrackMatchManager.h"
#include "Validation/MuonGEMDigis/interface/GEMDigiAnalyzer.h"



//
// constructors and destructor
//
GEMDigiAnalyzer::GEMDigiAnalyzer(const edm::ParameterSet& ps)
  : cfg_(ps.getParameterSet("simTrackMatching"))
  , simInputLabel_(ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits"))
  , minPt_(ps.getUntrackedParameter<double>("minPt", 5.))
  , verbose_(ps.getUntrackedParameter<int>("verbose", 0))
{
  bookGEMDigiTree();
  bookGEMCSCPadDigiTree();
  bookGEMCSCCoPadDigiTree();
  bookSimTracksTree();
}

GEMDigiAnalyzer::~GEMDigiAnalyzer() 
{
}

// ------------ method called when starting to processes a run  ------------
void GEMDigiAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(gem_geo_);
  gem_geometry_ = &*gem_geo_;

  const auto top_chamber = static_cast<const GEMEtaPartition*>(gem_geometry_->idToDetUnit(GEMDetId(1,1,1,1,1,1)));
   // TODO: it's really bad to hardcode max partition number!
  const auto bottom_chamber = static_cast<const GEMEtaPartition*>(gem_geometry_->idToDetUnit(GEMDetId(1,1,1,1,1,6)));
  const float top_half_striplength = top_chamber->specs()->specificTopology().stripLength()/2.;
  const float bottom_half_striplength = bottom_chamber->specs()->specificTopology().stripLength()/2.;
  const LocalPoint lp_top(0., top_half_striplength, 0.);
  const LocalPoint lp_bottom(0., -bottom_half_striplength, 0.);
  const GlobalPoint gp_top = top_chamber->toGlobal(lp_top);
  const GlobalPoint gp_bottom = bottom_chamber->toGlobal(lp_bottom);

  radiusCenter_ = (gp_bottom.perp() + gp_top.perp())/2.;
  chamberHeight_ = gp_top.perp() - gp_bottom.perp();

  using namespace std;
  cout<<"half top "<<top_half_striplength<<" bot "<<lp_bottom<<endl;
  cout<<"r  top "<<gp_top.perp()<<" bot "<<gp_bottom.perp()<<endl;
  LocalPoint p0(0.,0.,0.);
  cout<<"r0 top "<<top_chamber->toGlobal(p0).perp()<<" bot "<< bottom_chamber->toGlobal(p0).perp()<<endl;
  cout<<"rch "<<radiusCenter_<<" hch "<<chamberHeight_<<endl;

  buildLUT();
}

void GEMDigiAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iEvent.getByLabel(edm::InputTag("simMuonGEMDigis"), gem_digis);
  analyzeGEM();
  
  iEvent.getByLabel(edm::InputTag("simMuonGEMCSCPadDigis"), gemcscpad_digis);
  analyzeGEMCSCPad();  
  
  iEvent.getByLabel(edm::InputTag("simMuonGEMCSCPadDigis","Coincidence"), gemcsccopad_digis);
  analyzeGEMCSCCoPad();  

  iEvent.getByLabel(simInputLabel_, sim_tracks);

  iEvent.getByLabel(simInputLabel_, sim_vertices);

  analyzeTracks(cfg_,iEvent,iSetup);  
}

void GEMDigiAnalyzer::bookGEMDigiTree()
{
  edm::Service<TFileService> fs;
  gem_tree_ = fs->make<TTree>("GEMDigiTree", "GEMDigiTree");
  gem_tree_->Branch("detId", &gem_digi_.detId);
  gem_tree_->Branch("region", &gem_digi_.region);
  gem_tree_->Branch("ring", &gem_digi_.ring);
  gem_tree_->Branch("station", &gem_digi_.station);
  gem_tree_->Branch("layer", &gem_digi_.layer);
  gem_tree_->Branch("chamber", &gem_digi_.chamber);
  gem_tree_->Branch("roll", &gem_digi_.roll);
  gem_tree_->Branch("strip", &gem_digi_.strip);
  gem_tree_->Branch("bx", &gem_digi_.bx);
  gem_tree_->Branch("x", &gem_digi_.x);
  gem_tree_->Branch("y", &gem_digi_.y);
  gem_tree_->Branch("g_r", &gem_digi_.g_r);
  gem_tree_->Branch("g_eta", &gem_digi_.g_eta);
  gem_tree_->Branch("g_phi", &gem_digi_.g_phi);
  gem_tree_->Branch("g_x", &gem_digi_.g_x);
  gem_tree_->Branch("g_y", &gem_digi_.g_y);
  gem_tree_->Branch("g_z", &gem_digi_.g_z);
}

void GEMDigiAnalyzer::bookGEMCSCPadDigiTree()
{
  edm::Service<TFileService> fs;
  gemcscpad_tree_ = fs->make<TTree>("GEMCSCPadDigiTree", "GEMCSCPadDigiTree");
  gemcscpad_tree_->Branch("detId", &gemcscpad_digi_.detId);
  gemcscpad_tree_->Branch("region", &gemcscpad_digi_.region);
  gemcscpad_tree_->Branch("ring", &gemcscpad_digi_.ring);
  gemcscpad_tree_->Branch("station", &gemcscpad_digi_.station);
  gemcscpad_tree_->Branch("layer", &gemcscpad_digi_.layer);
  gemcscpad_tree_->Branch("chamber", &gemcscpad_digi_.chamber);
  gemcscpad_tree_->Branch("roll", &gemcscpad_digi_.roll);
  gemcscpad_tree_->Branch("pad", &gemcscpad_digi_.pad);
  gemcscpad_tree_->Branch("bx", &gemcscpad_digi_.bx);
  gemcscpad_tree_->Branch("x", &gemcscpad_digi_.x);
  gemcscpad_tree_->Branch("y", &gemcscpad_digi_.y);
  gemcscpad_tree_->Branch("g_r", &gemcscpad_digi_.g_r);
  gemcscpad_tree_->Branch("g_eta", &gemcscpad_digi_.g_eta);
  gemcscpad_tree_->Branch("g_phi", &gemcscpad_digi_.g_phi);
  gemcscpad_tree_->Branch("g_x", &gemcscpad_digi_.g_x);
  gemcscpad_tree_->Branch("g_y", &gemcscpad_digi_.g_y);
  gemcscpad_tree_->Branch("g_z", &gemcscpad_digi_.g_z);
}

void GEMDigiAnalyzer::bookGEMCSCCoPadDigiTree()
{
  edm::Service<TFileService> fs;
  gemcsccopad_tree_ = fs->make<TTree>("GEMCSCCoPadDigiTree", "GEMCSCCoPadDigiTree");
  gemcsccopad_tree_->Branch("detId", &gemcsccopad_digi_.detId);
  gemcsccopad_tree_->Branch("region", &gemcsccopad_digi_.region);
  gemcsccopad_tree_->Branch("ring", &gemcsccopad_digi_.ring);
  gemcsccopad_tree_->Branch("station", &gemcsccopad_digi_.station);
  gemcsccopad_tree_->Branch("layer", &gemcsccopad_digi_.layer);
  gemcsccopad_tree_->Branch("chamber", &gemcsccopad_digi_.chamber);
  gemcsccopad_tree_->Branch("roll", &gemcsccopad_digi_.roll);
  gemcsccopad_tree_->Branch("pad", &gemcsccopad_digi_.pad);
  gemcsccopad_tree_->Branch("bx", &gemcsccopad_digi_.bx);
  gemcsccopad_tree_->Branch("x", &gemcsccopad_digi_.x);
  gemcsccopad_tree_->Branch("y", &gemcsccopad_digi_.y);
  gemcsccopad_tree_->Branch("g_r", &gemcsccopad_digi_.g_r);
  gemcsccopad_tree_->Branch("g_eta", &gemcsccopad_digi_.g_eta);
  gemcsccopad_tree_->Branch("g_phi", &gemcsccopad_digi_.g_phi);
  gemcsccopad_tree_->Branch("g_x", &gemcsccopad_digi_.g_x);
  gemcsccopad_tree_->Branch("g_y", &gemcsccopad_digi_.g_y);
  gemcsccopad_tree_->Branch("g_z", &gemcsccopad_digi_.g_z);
}

 void GEMDigiAnalyzer::bookSimTracksTree()
 {
   edm::Service< TFileService > fs;
   track_tree_ = fs->make<TTree>("TrackTree", "TrackTree");
   track_tree_->Branch("pt", &track_.pt);
   track_tree_->Branch("eta", &track_.eta);
   track_tree_->Branch("phi", &track_.phi);
   track_tree_->Branch("charge", &track_.charge);
   track_tree_->Branch("endcap", &track_.endcap);
   track_tree_->Branch("gem_sh_layer1", &track_.gem_sh_layer1);
   track_tree_->Branch("gem_sh_layer2", &track_.gem_sh_layer2);
   track_tree_->Branch("gem_dg_layer1", &track_.gem_dg_layer1);
   track_tree_->Branch("gem_dg_layer2", &track_.gem_dg_layer2);
   track_tree_->Branch("gem_pad_layer1", &track_.gem_pad_layer1);
   track_tree_->Branch("gem_pad_layer2", &track_.gem_pad_layer2);
   track_tree_->Branch("gem_sh_eta", &track_.gem_sh_eta);
   track_tree_->Branch("gem_sh_phi", &track_.gem_sh_phi);
   track_tree_->Branch("gem_sh_x", &track_.gem_sh_x);
   track_tree_->Branch("gem_sh_y", &track_.gem_sh_y);
   track_tree_->Branch("gem_dg_eta", &track_.gem_dg_eta);
   track_tree_->Branch("gem_dg_phi", &track_.gem_dg_phi);
   track_tree_->Branch("gem_pad_eta", &track_.gem_pad_eta);
   track_tree_->Branch("gem_pad_phi", &track_.gem_pad_phi);
   track_tree_->Branch("gem_lx_even",&track_.gem_lx_even);
   track_tree_->Branch("gem_ly_even",&track_.gem_ly_even);
   track_tree_->Branch("gem_lx_odd",&track_.gem_lx_odd);
   track_tree_->Branch("gem_ly_odd",&track_.gem_ly_odd);
   track_tree_->Branch("has_gem_sh_l1",&track_.has_gem_sh_l1);
   track_tree_->Branch("has_gem_sh_l2",&track_.has_gem_sh_l2);
   track_tree_->Branch("has_gem_dg_l1",&track_.has_gem_dg_l1);
   track_tree_->Branch("has_gem_dg_l2",&track_.has_gem_dg_l2);
   track_tree_->Branch("has_gem_pad_l1",&track_.has_gem_pad_l1);
   track_tree_->Branch("has_gem_pad_l2",&track_.has_gem_pad_l2);
 }

// ------------ method called for each event  ------------
void GEMDigiAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void GEMDigiAnalyzer::endJob() 
{
}

// ======= GEM ========
void GEMDigiAnalyzer::analyzeGEM()
{
  //Loop over GEM digi collection
  for(GEMDigiCollection::DigiRangeIterator cItr = gem_digis->begin(); cItr != gem_digis->end(); ++cItr)
  {
    GEMDetId id = (*cItr).first; 

    const GeomDet* gdet = gem_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = gem_geo_->etaPartition(id);

    gem_digi_.detId = id();
    gem_digi_.region = (Short_t) id.region();
    gem_digi_.ring = (Short_t) id.ring();
    gem_digi_.station = (Short_t) id.station();
    gem_digi_.layer = (Short_t) id.layer();
    gem_digi_.chamber = (Short_t) id.chamber();
    gem_digi_.roll = (Short_t) id.roll();

    GEMDigiCollection::const_iterator digiItr; 
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      gem_digi_.strip = (Short_t) digiItr->strip();
      gem_digi_.bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());
      gem_digi_.x = (Float_t) lp.x();
      gem_digi_.y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      gem_digi_.g_r = (Float_t) gp.perp();
      gem_digi_.g_eta = (Float_t) gp.eta();
      gem_digi_.g_phi = (Float_t) gp.phi();
      gem_digi_.g_x = (Float_t) gp.x();
      gem_digi_.g_y = (Float_t) gp.y();
      gem_digi_.g_z = (Float_t) gp.z();

      // fill TTree
      gem_tree_->Fill();
    }
  }
}


// ======= GEMCSCPad ========
void GEMDigiAnalyzer::analyzeGEMCSCPad()
{
  //Loop over GEMCSCPad digi collection
  for(GEMCSCPadDigiCollection::DigiRangeIterator cItr = gemcscpad_digis->begin(); cItr != gemcscpad_digis->end(); ++cItr)
    {
    GEMDetId id = (*cItr).first; 

    const GeomDet* gdet = gem_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = gem_geo_->etaPartition(id);

    gemcscpad_digi_.detId = id();
    gemcscpad_digi_.region = (Short_t) id.region();
    gemcscpad_digi_.ring = (Short_t) id.ring();
    gemcscpad_digi_.station = (Short_t) id.station();
    gemcscpad_digi_.layer = (Short_t) id.layer();
    gemcscpad_digi_.chamber = (Short_t) id.chamber();
    gemcscpad_digi_.roll = (Short_t) id.roll();

    GEMCSCPadDigiCollection::const_iterator digiItr; 
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      gemcscpad_digi_.pad = (Short_t) digiItr->pad();
      gemcscpad_digi_.bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfPad(digiItr->pad());
      gemcscpad_digi_.x = (Float_t) lp.x();
      gemcscpad_digi_.y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      gemcscpad_digi_.g_r = (Float_t) gp.perp();
      gemcscpad_digi_.g_eta = (Float_t) gp.eta();
      gemcscpad_digi_.g_phi = (Float_t) gp.phi();
      gemcscpad_digi_.g_x = (Float_t) gp.x();
      gemcscpad_digi_.g_y = (Float_t) gp.y();
      gemcscpad_digi_.g_z = (Float_t) gp.z();

      // fill TTree
      gemcscpad_tree_->Fill();
    }
  }
}


// ======= GEMCSCCoPad ========
void GEMDigiAnalyzer::analyzeGEMCSCCoPad()
{
  //Loop over GEMCSCPad digi collection
  for(GEMCSCPadDigiCollection::DigiRangeIterator cItr = gemcsccopad_digis->begin(); cItr != gemcsccopad_digis->end(); ++cItr)
  {
    GEMDetId id = (*cItr).first; 

    const GeomDet* gdet = gem_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = gem_geo_->etaPartition(id);

    gemcsccopad_digi_.detId = id();
    gemcsccopad_digi_.region = (Short_t) id.region();
    gemcsccopad_digi_.ring = (Short_t) id.ring();
    gemcsccopad_digi_.station = (Short_t) id.station();
    gemcsccopad_digi_.layer = (Short_t) id.layer();
    gemcsccopad_digi_.chamber = (Short_t) id.chamber();
    gemcsccopad_digi_.roll = (Short_t) id.roll();

    GEMCSCPadDigiCollection::const_iterator digiItr; 
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      gemcsccopad_digi_.pad = (Short_t) digiItr->pad();
      gemcsccopad_digi_.bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfPad(digiItr->pad());
      gemcsccopad_digi_.x = (Float_t) lp.x();
      gemcsccopad_digi_.y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      gemcsccopad_digi_.g_r = (Float_t) gp.perp();
      gemcsccopad_digi_.g_eta = (Float_t) gp.eta();
      gemcsccopad_digi_.g_phi = (Float_t) gp.phi();
      gemcsccopad_digi_.g_x = (Float_t) gp.x();
      gemcsccopad_digi_.g_y = (Float_t) gp.y();
      gemcsccopad_digi_.g_z = (Float_t) gp.z();

      // fill TTree
      gemcsccopad_tree_->Fill();
    }
  }
}

bool GEMDigiAnalyzer::isSimTrackGood(const SimTrack &t)
{
  // SimTrack selection
  if (t.noVertex()) return false;
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < minPt_) return false;
  float eta = std::abs(t.momentum().eta());
  if (eta > 2.18 || eta < 1.55) return false; // no GEMs could be in such eta
  return true;
}

// ======= GEM Matching ========
void GEMDigiAnalyzer::analyzeTracks(edm::ParameterSet cfg_, const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer & sim_trks = *sim_tracks.product();

  for (auto& t: sim_trks)
  {
    if (!isSimTrackGood(t)) continue;
    
    // match hits and digis to this SimTrack
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup);
    
    const SimHitMatcher&  match_sh = match.simhits();
    const GEMDigiMatcher& match_gd = match.gemDigis();
    //    const SimTrack &t = match_sh.trk();
    
    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    track_.charge = t.charge();
    track_.endcap = (track_.eta > 0.) ? 1 : -1;
    track_.gem_sh_layer1 = 0; 
    track_.gem_sh_layer2 = 0; 
    track_.gem_dg_layer1 = 0; 
    track_.gem_dg_layer2 = 0; 
    track_.gem_pad_layer1 = 0; 
    track_.gem_pad_layer2 = 0; 
    track_.gem_sh_eta = -9.;
    track_.gem_sh_phi = -9.;
    track_.gem_sh_x = -999;
    track_.gem_sh_y = -999;
    track_.gem_dg_eta = -9.;
    track_.gem_dg_phi = -9.;
    track_.gem_pad_eta = -9.;
    track_.gem_pad_phi = -9.;
    track_.gem_trk_rho = -999.;
    track_.gem_lx_even = -999.;
    track_.gem_ly_even = -999.;
    track_.gem_lx_odd = -999.;
    track_.gem_ly_odd = -999.;
    track_.has_gem_sh_l1 = 0;
    track_.has_gem_sh_l2 = 0;
    track_.has_gem_dg_l1 = 0;
    track_.has_gem_dg_l2 = 0;
    track_.has_gem_pad_l1 = 0;
    track_.has_gem_pad_l2 = 0;
    
    // ** GEM SimHits ** //    
    auto gem_sh_ids_sch = match_sh.superChamberIdsGEM();
    for(auto d: gem_sh_ids_sch)
    {
      auto gem_simhits = match_sh.hitsInSuperChamber(d);
      auto gem_simhits_gp = match_sh.simHitsMeanPosition(gem_simhits);

      track_.gem_sh_eta = gem_simhits_gp.eta();
      track_.gem_sh_phi = gem_simhits_gp.phi();
      track_.gem_sh_x = gem_simhits_gp.x();
      track_.gem_sh_y = gem_simhits_gp.y();
    }

    // Calculation of the localXY efficiency
    GlobalPoint gp_track(match_sh.propagatedPositionGEM());
    track_.gem_trk_eta = gp_track.eta();
    track_.gem_trk_phi = gp_track.phi();
    track_.gem_trk_rho = gp_track.perp();
    std::cout << "track eta phi rho = " << track_.gem_trk_eta << " " << track_.gem_trk_phi << " " << track_.gem_trk_rho << std::endl;
    
    float track_angle = gp_track.phi().degrees();
    if (track_angle < 0.) track_angle += 360.;
    std::cout << "track angle = " << track_angle << std::endl;
    const int track_region = (gp_track.z() > 0 ? 1 : -1);
    
    // closest chambers in phi
    const auto mypair = getClosestChambers(track_region, track_angle);
    
    // chambers
    GEMDetId detId_first(mypair.first);
    GEMDetId detId_second(mypair.second);

    // assignment of local even and odd chambers (there is always an even and an odd chamber)
    bool firstIsOdd = detId_first.chamber() & 1;
    
    GEMDetId detId_even_L1(firstIsOdd ? detId_second : detId_first);
    GEMDetId detId_odd_L1(firstIsOdd ? detId_first  : detId_second);

    auto even_partition = gem_geometry_->idToDetUnit(detId_even_L1)->surface();
    auto odd_partition  = gem_geometry_->idToDetUnit(detId_odd_L1)->surface();

    // global positions of partitions' centers
    LocalPoint p0(0.,0.,0.);
    GlobalPoint gp_even_partition = even_partition.toGlobal(p0);
    GlobalPoint gp_odd_partition = odd_partition.toGlobal(p0);
    
    LocalPoint lp_track_even_partition = even_partition.toLocal(gp_track);
    LocalPoint lp_track_odd_partition = odd_partition.toLocal(gp_track);

    // track chamber local x is the same as track partition local x
    track_.gem_lx_even = lp_track_even_partition.x();
    track_.gem_lx_odd = lp_track_odd_partition.x();

    // track chamber local y is the same as track partition local y
    // corrected for partition's local y WRT chamber
    track_.gem_ly_even = lp_track_even_partition.y() + (gp_even_partition.perp() - radiusCenter_);
    track_.gem_ly_odd = lp_track_odd_partition.y() + (gp_odd_partition.perp() - radiusCenter_);

    std::cout << track_.gem_lx_even << " " << track_.gem_ly_even << std::endl;
    std::cout << track_.gem_lx_odd << " " << track_.gem_ly_odd << std::endl;


    auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      GEMDetId id(d);
      bool odd(id.chamber() & 1);
      
      if (id.layer() == 1)
      {
	if (odd) track_.gem_sh_layer1 |= 1;
	else     track_.gem_sh_layer1 |= 2;
      }
      else if (id.layer() == 2)
      {
	if (odd) track_.gem_sh_layer2 |= 1;
	else     track_.gem_sh_layer2 |= 2;
      }
    }

    // ** GEM Digis, Pads and CoPads ** //    
    auto gem_dg_ids_sch = match_gd.superChamberIds();
    for(auto d: gem_dg_ids_sch)
    {
      auto gem_digis = match_gd.digisInSuperChamber(d);
      auto gem_dg_gp = match_gd.digisMeanPosition(gem_digis);

      track_.gem_dg_eta = gem_dg_gp.eta();
      track_.gem_dg_phi = gem_dg_gp.phi();	      

      auto gem_pads = match_gd.padsInSuperChamber(d);
      auto gem_pad_gp = match_gd.digisMeanPosition(gem_pads);	  
      
      track_.gem_pad_eta = gem_pad_gp.eta();
      track_.gem_pad_phi = gem_pad_gp.phi();	      
    }

    auto gem_dg_ids_ch = match_gd.chamberIds();
    for(auto d: gem_dg_ids_ch)
    {
      GEMDetId id(d);
      bool odd(id.chamber() & 1);
      
      if (id.layer() == 1)
      {
	if (odd)
	{
	  track_.gem_dg_layer1 |= 1;
	  track_.gem_pad_layer1 |= 1;
	}
	else     
	{
	  track_.gem_dg_layer1 |= 2;
	  track_.gem_pad_layer1 |= 2;
	}
      }
      else if (id.layer() == 2)
      {
	if (odd)
	{
	  track_.gem_dg_layer2 |= 1;
	  track_.gem_pad_layer2 |= 1;
	}
	else     
	{
	  track_.gem_dg_layer2 |= 2;
	  track_.gem_pad_layer2 |= 2;
	}
      }
    }

    // Construct Chamber DetIds from the "projected" ids:
    GEMDetId id_ch_even_L1(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 1, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L1(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 1, detId_odd_L1.chamber(), 0);
    GEMDetId id_ch_even_L2(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 2, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L2(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 2, detId_odd_L1.chamber(), 0);

    // check if track has sh 
    if(gem_sh_ids_ch.count(id_ch_even_L1)!=0) track_.has_gem_sh_l1 |= 2;
    if(gem_sh_ids_ch.count(id_ch_odd_L1)!=0)  track_.has_gem_sh_l1 |= 1;
    if(gem_sh_ids_ch.count(id_ch_even_L2)!=0) track_.has_gem_sh_l2 |= 2;
    if(gem_sh_ids_ch.count(id_ch_odd_L2)!=0)  track_.has_gem_sh_l2 |= 1;

    // check if track has dg
    if(gem_dg_ids_ch.count(id_ch_even_L1)!=0){
      track_.has_gem_dg_l1 |= 2;
      track_.has_gem_pad_l1 |= 2;
    }
    if(gem_dg_ids_ch.count(id_ch_odd_L1)!=0){
      track_.has_gem_dg_l1 |= 1; 
      track_.has_gem_pad_l1 |= 1;
    }
    if(gem_dg_ids_ch.count(id_ch_even_L2)!=0){
      track_.has_gem_dg_l2 |= 2; 
      track_.has_gem_pad_l2 |= 2;
    }
    if(gem_dg_ids_ch.count(id_ch_odd_L2)!=0){
      track_.has_gem_dg_l2 |= 1; 
      track_.has_gem_pad_l2 |= 1;
    }

    track_tree_->Fill();
  } // track loop
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GEMDigiAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void GEMDigiAnalyzer::buildLUT()
{
  std::vector<int> pos_ids;
  pos_ids.push_back(GEMDetId(1,1,1,1,36,1).rawId());

  std::vector<int> neg_ids;
  neg_ids.push_back(GEMDetId(-1,1,1,1,36,1).rawId());

  // VK: I would really suggest getting phis from GEMGeometry
  
  std::vector<float> phis;
  phis.push_back(0.);
  for(int i=1; i<37; ++i)
  {
    pos_ids.push_back(GEMDetId(1,1,1,1,i,1).rawId());
    neg_ids.push_back(GEMDetId(-1,1,1,1,i,1).rawId());
    phis.push_back(i*10.);
  }
  positiveLUT_ = std::make_pair(phis,pos_ids);
  negativeLUT_ = std::make_pair(phis,neg_ids);
}

std::pair<int,int>
GEMDigiAnalyzer::getClosestChambers(int region, float phi)
{
  auto& phis(positiveLUT_.first);
  auto upper = std::upper_bound(phis.begin(), phis.end(), phi);
  std::cout << "lower = " << upper - phis.begin()  << std::endl;
  std::cout << "upper = " << upper - phis.begin() + 1 << std::endl;
  auto& LUT = (region == 1 ? positiveLUT_.second : negativeLUT_.second);
  return std::make_pair(LUT.at(upper - phis.begin()), (LUT.at((upper - phis.begin() + 1)%36)));
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMDigiAnalyzer);

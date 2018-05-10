#include "GEMCosmicMuonStandSim.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"

#include "TMath.h"

#include <vector>
#include <iostream>
#include <iterator> // std::distance
#include <algorithm> // std::find, std::transform, std::replace_if, std::min
#include <numeric> // std::iota
#include <string>
#include <cmath> // std::remainder, std::fabs

using namespace std;

GEMCosmicMuonStandSim::GEMCosmicMuonStandSim(const edm::ParameterSet& cfg)
{
  sim_hit_token_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simHitToken"));
  rec_hit_token_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitToken"));
}


GEMCosmicMuonStandSim::~GEMCosmicMuonStandSim() {}




TString GEMCosmicMuonStandSim::ConvertTitleToName(TString const& title)
{
  //
  std::string tmp_name = title.Data();
  std::replace_if(tmp_name.begin(), tmp_name.end(), ::ispunct, '_');
  std::replace_if(tmp_name.begin(), tmp_name.end(), ::isspace, '_');
  std::transform(tmp_name.begin(), tmp_name.end(), tmp_name.begin(), ::tolower);
  //
  TString name = tmp_name;
  name.ReplaceAll("__", "_");
  name = name.Strip(TString::EStripType::kBoth, '_');
  //
  return name;
}


MonitorElement* GEMCosmicMuonStandSim::BookHist1D(DQMStore::IBooker &ibooker,
                                                  TString title,
                                                  Int_t nchX, Double_t lowX, Double_t highX)
{
  TString name = ConvertTitleToName(title);
  return ibooker.book1D(name, title, nchX, lowX, highX);
}


MonitorElement* GEMCosmicMuonStandSim::BookHist2D(DQMStore::IBooker &ibooker,
                                                TString title,
                                                Int_t nchX, Double_t lowX, Double_t highX,
                                                Int_t nchY, Double_t lowY, Double_t highY)
{
  TString name = ConvertTitleToName(title);
  return ibooker.book2D(name, title, nchX, lowX, highX, nchY, lowY, highY);
}


void GEMCosmicMuonStandSim::bookHistograms(DQMStore::IBooker & ibooker,
                                           edm::Run const & Run,
                                           edm::EventSetup const & iSetup) {
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);

  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  LogDebug("GEMCosmicMuonStandSim") << "Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("GEMCosmicMuonStandSim");

  LogDebug("GEMCosmicMuonStandSim") << "ibooker set current folder\n";

  me_vfat_total_ = BookHist2D(
    ibooker, "The Number Of Total Events",
    15, 0, 14 + 1,
    3*8, 1, 3*8 + 1); // 24 = kNumRollId * kNumVFATId

  me_vfat_passed_ = BookHist2D(
    ibooker, "The Number Of Passed Events",
    15, 0, 14 + 1,
    3*8, 1, 3*8 + 1);

  me_vfat_occupancy_ = BookHist2D(
    ibooker, "Occupancy",
    15, 0, 14 + 1,
    3*8, 1, 3*8 + 1);

  me_vfat_passed_ = BookHist2D(
    ibooker, "The Number Of Passed Events", 15, 0, 15, 24, 1, 24 + 1);

  for(Int_t chamber_id = 1; chamber_id <= 29; chamber_id += 2) {
    Int_t index = GetChamberIndex(chamber_id);

    me_vfat_total_per_chamber_[index] = BookHist2D(ibooker,
      TString::Format("The Number Of Total Events (Chamber %d)", chamber_id),
      kNumVFATId_, kMinVFATId_, kMaxVFATId_ + 1,
      kNumRollId_, kMinRollId_, kMaxRollId_ + 1);

    me_vfat_passed_per_chamber_[index] = BookHist2D(ibooker,
      TString::Format("The Number Of Passed Events (Chamber %d)", chamber_id),
      kNumVFATId_, kMinVFATId_, kMaxVFATId_ + 1,
      kNumRollId_, kMinRollId_, kMaxRollId_ + 1);

    me_vfat_occupancy_per_chamber_[index] = BookHist2D(ibooker,
      TString::Format("Occupancy (Chamber %d)", chamber_id),
      kNumVFATId_, kMinVFATId_, kMaxVFATId_ + 1,
      kNumRollId_, kMinRollId_, kMaxRollId_ + 1);
  }

  me_residual_local_x_ = BookHist1D(ibooker, "The Residuals of The Local X", 100, -1, 1);
  me_residual_local_y_ = BookHist1D(ibooker, "The Residuals of The Local Y", 100, -10, 10);
  me_residual_local_phi_ = BookHist1D(ibooker, "The Residuals of The Local Phi", 100, -0.015, 0.015);

  me_error_x_ = BookHist1D(ibooker, "Local Postion Error X", 100, 0, 0.3);
  me_error_y_ = BookHist1D(ibooker, "Local Postion Error Y", 100, 0, 100);

  me_pull_local_x_ = BookHist1D(ibooker, "The Pulls of The Local X", 100, -30, 30);
  me_pull_local_y_ = BookHist1D(ibooker, "The Pulls of The Local Y", 100, -0.6, 0.6);

  me_cls_ = BookHist1D(ibooker, "The Cluster Size of RecHit", 10, 1, 10+1);
  me_cls_vs_chamber_ = BookHist2D(ibooker, "CLS vs Chamber", 15, 1, 15+1, 10, 1, 10+1);
  me_num_clusters_ = BookHist1D(ibooker, "The Number of Clusters", 8, 1, 8+1);
  me_num_clusters_vs_chamber_ = BookHist2D(ibooker, "Number of Cluster vs Chamber", 15, 1, 15+1, 10, 1, 10+1);

  me_num_sim_hits_ = BookHist1D(ibooker, "The Number of SimHits", 25, 1, 25+1);
  me_num_rec_hits_ = BookHist1D(ibooker, "The Number of RecHits", 25, 1, 25+1);

  me_sim_hit_local_phi_ = BookHist1D(ibooker, "Local Phi of SimHits", 25, -1 * TMath::Pi() / 18, TMath::Pi() / 18);
  me_rec_hit_local_phi_ = BookHist1D(ibooker, "Local Phi of RecHits", 25, -1 * TMath::Pi() / 18, TMath::Pi() / 18);

  // For debug
  me_mat_chamber_ = BookHist1D(ibooker, "Matching Case - Chamber Id", kNumChamberId_, kMinChamberId_, kMaxChamberId_+1);
  me_mis_chamber_ = BookHist1D(ibooker, "Mismatching Case - Chamber Id", 5, 1, 5+1);

  me_sim_strip_ = BookHist1D(ibooker, "Sim Fired Strip", 385, 0 , 384 + 1); // 0 ~ 384
  me_rec_first_strip_ = BookHist1D(ibooker, "Rec First Fired Strip", 385, 0 , 384 + 1);
  // me_strip_diff_ = BookHist1D(ibooker, "Difference between RecStrip and SimStrip", 101, -50.5, 50.5);

  LogDebug("GEMCosmicMuonStandSim")<<"Booking End.\n";
}


Int_t GEMCosmicMuonStandSim::GetVFATId(Float_t x, const GEMEtaPartition* roll) {
  Int_t nstrips = roll->nstrips();
  Float_t x_min = roll->centreOfStrip(1).x(); // - strip width
  Float_t x_max = roll->centreOfStrip(nstrips).x(); // + strip width

  Float_t x0 = std::min(x_min, x_max);

  // 3.0 means the number of phi-segmentations  in the eta partition.
  Float_t width = std::fabs(x_max - x_min) / kNumVFATId_;

  if (x < x0 + width)        return 1;
  else if (x < x0 + 2 * width) return 2;
  else if (x < x0 + 3 * width) return 3;
  else                       return -1;
}


void GEMCosmicMuonStandSim::analyze(const edm::Event& e,
                                    const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  edm::Handle<GEMRecHitCollection> gem_rec_hits;
  e.getByToken(rec_hit_token_, gem_rec_hits);
  edm::Handle<edm::PSimHitContainer> gem_sim_hits;
  e.getByToken(sim_hit_token_, gem_sim_hits);
  
  if (not gem_rec_hits.isValid()) {
    edm::LogError("GEMCosmicMuonStandSim") << "Cannot get strips by Token RecHits Token.\n";
    return ;
  }

  // if( isMC) 
  Int_t num_sim_hits = gem_sim_hits->size();
  if(num_sim_hits == 0) return ;
  Int_t num_rec_hits = std::distance(gem_rec_hits->begin(), gem_rec_hits->end());
  // if(num_rec_hits == 0) return ;

  me_num_sim_hits_->Fill(num_sim_hits);
  me_num_rec_hits_->Fill(num_rec_hits);

  for(edm::PSimHitContainer::const_iterator sim_hit = gem_sim_hits->begin(); sim_hit != gem_sim_hits->end(); ++sim_hit)
  {
    Local3DPoint sim_hit_lp = sim_hit->localPosition();
    GEMDetId sim_det_id(sim_hit->detUnitId());

    // XXX +1 ?
    Int_t sim_fired_strip = GEMGeometry_->etaPartition(sim_det_id)->strip(sim_hit_lp) + 1;

    const GEMEtaPartition* kSimRoll = GEMGeometry_->etaPartition(sim_det_id);
    Int_t sim_vfat_id = GetVFATId(sim_hit_lp.x(), kSimRoll);
    Int_t sim_chamber_idx = GetChamberIndex(sim_det_id.chamber()); // 0 ~ 14

    // Int_t y_overall_vfat = 3 * sim_det_id.roll() + sim_vfat_id - 3; // 1 ~ 24
    Int_t y_overall_vfat_plot = GetOverallVFATPlotY(sim_det_id.roll(), sim_vfat_id);
    me_vfat_total_->Fill(sim_chamber_idx, y_overall_vfat_plot);
    me_vfat_total_per_chamber_[sim_chamber_idx]->Fill(sim_vfat_id, sim_det_id.roll());

    GEMRecHitCollection::range range = gem_rec_hits->get(sim_det_id);
    Int_t num_clusters = std::distance(range.first, range.second);
    me_num_clusters_->Fill(num_clusters);

    // candidates of strip difference in the mismatching case
    //Int_t mis_strip_diff = 999;

    for(GEMRecHitCollection::const_iterator rec_hit = range.first; rec_hit != range.second; ++rec_hit)
    {
      Int_t rec_cls = rec_hit->clusterSize();

      // Checkt whether a sim. fired strip is in a rec. cluster strips.
      Bool_t is_matched_hit;
      if ( rec_cls == 1 ) {
        is_matched_hit = sim_fired_strip == rec_hit->firstClusterStrip();

        me_sim_strip_->Fill(sim_fired_strip);
        me_rec_first_strip_->Fill(sim_fired_strip);
      }
      else {
        Int_t rec_first_fired_strip = rec_hit->firstClusterStrip();
        Int_t rec_last_fired_strip = rec_first_fired_strip + rec_cls - 1;
        is_matched_hit = (sim_fired_strip >= rec_first_fired_strip) and (sim_fired_strip <= rec_last_fired_strip);

        me_sim_strip_->Fill(sim_fired_strip);
        me_rec_first_strip_->Fill(rec_first_fired_strip);
      }

      // If SimHit and RecHit are matched,
      // then monitor elements fill value.
      if( is_matched_hit )
      {
        LocalPoint rec_hit_lp = rec_hit->localPosition();
        GEMDetId rec_det_id = rec_hit->gemId();
        // GlobalPoint rec_hitGP = GEMGeometry_->idToDet(rec_det_id)->surface().toGlobal(rec_hit_lp);

        const GEMEtaPartition* kRecRoll = GEMGeometry_->etaPartition(rec_det_id);
        
        Float_t sim_hit_local_phi = GetLocalPhi(kRecRoll->strip(sim_hit_lp));
        Float_t rec_hit_local_phi = GetLocalPhi(kRecRoll->strip(rec_hit_lp));
        Float_t residual_local_phi = rec_hit_local_phi - sim_hit_local_phi;

        // Int_t recVFATId = GetVFATId(rec_hit_lp.x(), kRecRoll);
        // Int_t recChamberIdx = GetChamberIndex(rec_det_id.chamber());

        me_vfat_passed_->Fill(sim_chamber_idx, y_overall_vfat_plot);
        me_vfat_passed_per_chamber_[sim_chamber_idx]->Fill(sim_vfat_id, sim_det_id.roll());

        Float_t residual_local_x = rec_hit_lp.x() - sim_hit_lp.x();
        Float_t residual_local_y = rec_hit_lp.y() - sim_hit_lp.y();
        Float_t error_x = rec_hit->localPositionError().xx();
        Float_t error_y = rec_hit->localPositionError().yy();
        Float_t pull_x = residual_local_x / error_x;
        Float_t pull_y = residual_local_y / error_y;

        me_residual_local_x_->Fill(residual_local_x);
        me_residual_local_y_->Fill(residual_local_y);
        me_residual_local_phi_->Fill(residual_local_phi);
        me_error_x_->Fill(error_x);
        me_error_y_->Fill(error_y);
        me_pull_local_x_->Fill(pull_x);
        me_pull_local_y_->Fill(pull_y);

        me_cls_->Fill(rec_cls);
        me_cls_vs_chamber_->Fill(sim_chamber_idx, rec_cls);
        me_num_clusters_vs_chamber_->Fill(sim_chamber_idx, num_clusters);
        me_sim_hit_local_phi_->Fill(sim_hit_local_phi);
        me_rec_hit_local_phi_->Fill(rec_hit_local_phi);
        break;
      }
      else
      {
        continue;
        // Int_t rec_first_fired_strip = rec_hit->firstClusterStrip();
        // Int_t rec_last_fired_strip = rec_first_fired_strip + rec_cls - 1;
        // Int_t first_diff = rec_first_fired_strip - sim_fired_strip;
        // Int_t last_diff = rec_last_fired_strip - sim_fired_strip;
        // Int_t strip_diff = std::abs(first_diff) < std::abs(last_diff) ? first_diff : last_diff;

        // if( std::abs(strip_diff) < std::abs(mis_strip_diff) )
        //   mis_strip_diff = strip_diff; 
      }
    } // rechit loop end
  } // simhit loop end


  // TODO occupancy
  for(GEMRecHitCollection::const_iterator rec_hit = gem_rec_hits->begin(); rec_hit != gem_rec_hits->end(); ++rec_hit)
  {
    LocalPoint rec_hit_lp = rec_hit->localPosition();
    GEMDetId rec_det_id = rec_hit->gemId();

    const GEMEtaPartition* kRecRoll = GEMGeometry_->etaPartition(rec_det_id);

    Int_t rec_chamber_idx = GetChamberIndex(rec_det_id.chamber());
    Int_t rec_vfat_id = GetVFATId(rec_hit_lp.x(), kRecRoll);

    me_vfat_occupancy_per_chamber_[rec_chamber_idx]->Fill(rec_vfat_id, rec_det_id.roll());

    Int_t y_overall_vfat_plot = GetOverallVFATPlotY(rec_det_id.roll(), rec_vfat_id);
    me_vfat_occupancy_->Fill(rec_chamber_idx, y_overall_vfat_plot);

  }


} // analyze end

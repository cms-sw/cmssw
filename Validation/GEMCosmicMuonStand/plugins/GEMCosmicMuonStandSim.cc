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

  me_vfat_total_ = BookHist2D(ibooker, "The Number Of Total Events", 15, 0, 15, 24, 1, 24 + 1);
  me_vfat_passed_ = BookHist2D(ibooker, "The Number Of Passed Events", 15, 0, 15, 24, 1, 24 + 1);


  for(Int_t chamber_id = 1; chamber_id <= 29; chamber_id += 2) {
    Int_t index = GetChamberIndex(chamber_id);

    me_vfat_total_per_chamber_[index] = BookHist2D(ibooker,
      TString::Format("The Number Of Total Events (Chamber %d)", chamber_id),
      kMaxVFATId_, kMinVFATId_, kMaxVFATId_ + 1,
      kMaxRollId_, kMinVFATId_, kMaxVFATId_ + 1);

    me_vfat_passed_per_chamber_[index] = BookHist2D(ibooker,
      TString::Format("The Number Of Passed Events (Chamber %d)", chamber_id),
      kMaxVFATId_, kMinVFATId_, kMaxVFATId_ + 1,
      kMaxRollId_, kMinVFATId_, kMaxVFATId_ + 1);

    me_vfat_occupancy_per_chamber_[index] = BookHist2D(ibooker,
      TString::Format("Occupancy (Chamber %d)", chamber_id),
      kMaxVFATId_, kMinVFATId_, kMaxVFATId_ + 1,
      kMaxRollId_, kMinVFATId_, kMaxVFATId_ + 1);
  }

  me_residual_local_x_ = BookHist1D(ibooker, "The Residuals of The Local X", 100, -1, 1);
  me_residual_local_y_ = BookHist1D(ibooker, "The Residuals of The Local Y", 100, -10, 10);
  me_residual_local_phi_ = BookHist1D(ibooker, "The Residuals of The Local Phi", 100, 1.5, 1.5);

  me_error_x_ = BookHist1D(ibooker, "Local Postion Error X", 100, 0, 0.3);
  me_error_y_ = BookHist1D(ibooker, "Local Postion Error Y", 100, 0, 100);

  me_pull_local_x_ = BookHist1D(ibooker, "The Pulls of The Local X", 100, -10, 10);
  me_pull_local_y_ = BookHist1D(ibooker, "The Pulls of The Local Y", 100, -0.6, 0.6);

  me_cls_ = BookHist1D(ibooker, "The Cluster Size of RecHit", 10, 0, 10);
  me_cls_vs_chamber_ = BookHist2D(ibooker, "CLS vs Chamber", 15, 0, 15, 10, 0, 10);
  me_num_clusters_vs_chamber_ = BookHist2D(ibooker, "Number of Cluster vs Chamber", 15, 0, 15, 10, 0, 10);
  meNumClusters_ = BookHist1D(ibooker, "The Number of Clusters", 8, 1, 9);
  me_num_sim_hits_ = BookHist1D(ibooker, "The Number of SimHits", 25, 0, 26);
  me_num_rec_hits_ = BookHist1D(ibooker, "The Number of RecHits", 25, 0, 26);
  me_sim_hit_bare_local_phi_ = BookHist1D(ibooker, "Bare Local Phi of SimHits", 100, -3*TMath::Pi(), 3*TMath::Pi());
  me_sim_hit_local_phi_ = BookHist1D(ibooker, "Local Phi of SimHits", 100, -3*TMath::Pi(), 3*TMath::Pi());
  me_rec_hit_local_phi_ = BookHist1D(ibooker, "Local Phi of RecHits", 100, -1 * TMath::Pi() / 18, TMath::Pi() / 18);

  // For debug
  me_mat_chamber_ = BookHist1D(ibooker, "Matching Case - Chamber Id", 15, 0, 30);
  me_mis_chamber_ = BookHist1D(ibooker, "Mismatching Case - Chamber Id", 15, 0, 30);

  LogDebug("GEMCosmicMuonStandSim")<<"Booking End.\n";
}


Int_t GEMCosmicMuonStandSim::GetVFATId(Float_t x, const GEMEtaPartition* roll) {
  Int_t nstrips = roll->nstrips();
  Float_t x_min = roll->centreOfStrip(1).x(); // - strip width
  Float_t x_max = roll->centreOfStrip(nstrips).x(); // + strip width

  Float_t x0 = std::min(x_min, x_max);

  // 3.0 means the number of phi-segmentations  in the eta partition.
  Float_t width = std::fabs(x_max - x_min) / 3.0;

  if (x < x0 + width)        return 1;
  else if (x < x0 + 2*width) return 2;
  else if (x < x0 + 3*width) return 3;
  else                       return -1;
}


void GEMCosmicMuonStandSim::analyze(const edm::Event& e,
                                    const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  edm::Handle<GEMRecHitCollection> gemRecHits;
  e.getByToken( rec_hit_token_, gemRecHits);
  edm::Handle<edm::PSimHitContainer> gemSimHits;
  e.getByToken( sim_hit_token_, gemSimHits);
  
  if (not gemRecHits.isValid()) {
    edm::LogError("GEMCosmicMuonStandSim") << "Cannot get strips by Token RecHits Token.\n";
    return ;
  }

  // if( isMC) 
  Int_t num_sim_hits = gemSimHits->size();
  if(num_sim_hits == 0) return ;
  Int_t num_rec_hits = std::distance(gemRecHits->begin(), gemRecHits->end());
  if(num_rec_hits == 0) return ;

  me_num_sim_hits_->Fill(num_sim_hits);
  me_num_rec_hits_->Fill(num_rec_hits);

  for(edm::PSimHitContainer::const_iterator sim_hit = gemSimHits->begin(); sim_hit != gemSimHits->end(); ++sim_hit)
  {
    Local3DPoint sim_hit_lp = sim_hit->localPosition();
    GEMDetId sim_det_id(sim_hit->detUnitId());
    // XXX +1 ?
    Int_t sim_fired_strip = GEMGeometry_->etaPartition(sim_det_id)->strip(sim_hit->entryPoint()) + 1;

    const GEMEtaPartition* kSimRoll = GEMGeometry_->etaPartition(sim_det_id);
    Int_t sim_vfat_id = GetVFATId(sim_hit_lp.x(), kSimRoll);
    Int_t sim_chamber_idx = GetChamberIndex(sim_det_id.chamber());

    Int_t y_overall_vfat = 3 * sim_det_id.roll() + sim_vfat_id - 3; // 1 ~ 24
    me_vfat_total_->Fill(sim_chamber_idx, y_overall_vfat);
    me_vfat_total_per_chamber_[sim_chamber_idx]->Fill(sim_vfat_id, sim_det_id.roll());

    GEMRecHitCollection::range range = gemRecHits->get(sim_det_id);
    Int_t num_clusters = std::distance(range.first, range.second);
    meNumClusters_->Fill(num_clusters);

    Int_t num_matched = 0;
    for(GEMRecHitCollection::const_iterator rec_hit = range.first; rec_hit != range.second; ++rec_hit)
    {
      Int_t rec_cls = rec_hit->clusterSize();

      Bool_t is_matched;
      if ( rec_cls == 1 ) {
        is_matched = sim_fired_strip == rec_hit->firstClusterStrip();
      }
      else {
        Int_t rec_first_fired_strip = rec_hit->firstClusterStrip();
        Int_t rec_last_fired_strip = rec_first_fired_strip + rec_cls - 1;
        is_matched = (sim_fired_strip >= rec_first_fired_strip) and (sim_fired_strip <= rec_last_fired_strip);
      }

      if( is_matched )
      {
        num_matched++;

        LocalPoint rec_hit_lp = rec_hit->localPosition();
        GEMDetId rec_det_id = rec_hit->gemId();
        // GlobalPoint rec_hitGP = GEMGeometry_->idToDet(rec_det_id)->surface().toGlobal(rec_hit_lp);

        const GEMEtaPartition* kRecRoll = GEMGeometry_->etaPartition(rec_det_id);
        
        Float_t sim_hit_local_phi = GetLocalPhi(kRecRoll->strip(sim_hit_lp));
        Float_t rec_hit_local_phi = GetLocalPhi(kRecRoll->strip(rec_hit_lp));
        Float_t residual_local_phi = rec_hit_local_phi - sim_hit_local_phi;

        // Int_t recVFATId = GetVFATId(rec_hit_lp.x(), kRecRoll);
        // Int_t recChamberIdx = GetChamberIndex(rec_det_id.chamber());

        me_vfat_passed_->Fill(sim_chamber_idx, y_overall_vfat);
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
        me_sim_hit_bare_local_phi_->Fill(sim_hit_lp.phi());
        me_sim_hit_local_phi_->Fill(sim_hit_local_phi);
        me_rec_hit_local_phi_->Fill(rec_hit_local_phi);
      }
      else {
        // TODO
        continue;
      }
    } // RECHIT LOOP END
  } // SIMHIT LOOP END
}

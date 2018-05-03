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
  simHitToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simHitToken"));
  recHitToken_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitToken"));


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

  meVFATTotal_ = BookHist2D(ibooker, "The Number Of Total Events", 15, 0, 15, 24, 1, 24 + 1);
  meVFATPassed_ = BookHist2D(ibooker, "The Number Of Passed Events", 15, 0, 15, 24, 1, 24 + 1);

  for(Int_t chamber_id = 1; chamber_id <= 29; chamber_id += 2) {
    Int_t index = GetChamberIndex(chamber_id);

    meVFATTotalPerChamber_[index] = BookHist2D(ibooker,
      TString::Format("The Number Of Total Events (Chamber %d)", chamber_id),
      kNumPhiSegments_, 1, kNumPhiSegments_ + 1,
      kNumEtaSegments_, 1, kNumEtaSegments_ + 1);

    meVFATPassedPerChamber_[index] = BookHist2D(ibooker,
      TString::Format("The Number Of Passed Events (Chamber %d)", chamber_id),
      kNumPhiSegments_, 1, kNumPhiSegments_ + 1,
      kNumEtaSegments_, 1, kNumEtaSegments_ + 1);

    meVFATOccupancyPerChamber_[index] = BookHist2D(ibooker,
      TString::Format("Occupancy (Chamber %d)", chamber_id),
      kNumPhiSegments_, 1, kNumPhiSegments_ + 1,
      kNumEtaSegments_, 1, kNumEtaSegments_ + 1);
  }

  meResidualLocalX_ = BookHist1D(ibooker, "The Residuals of The Local X", 100, -1, 1);
  meResidualLocalY_ = BookHist1D(ibooker, "The Residuals of The Local Y", 100, -10, 10);
  meResidualLocalPhi_ = BookHist1D(ibooker, "The Residuals of The Local Phi", 100, 1.5, 1.5);

  meLocalPosErrorX_ = BookHist1D(ibooker, "Local Postion Error X", 100, 0, 0.3);
  meLocalPosErrorY_ = BookHist1D(ibooker, "Local Postion Error Y", 100, 0, 100);

  mePullLocalX_ = BookHist1D(ibooker, "The Pulls of The Local X", 100, -10, 10);
  mePullLocalY_ = BookHist1D(ibooker, "The Pulls of The Local Y", 100, -0.6, 0.6);

  meCLS_ = BookHist1D(ibooker, "The Cluster Size of RecHit", 10, 0, 10);
  meCLSvsChamber_ = BookHist2D(ibooker, "CLS vs Chamber", 15, 0, 15, 10, 0, 10);
  meNumClustersvsChamber_ = BookHist2D(ibooker, "Number of Cluster vs Chamber", 15, 0, 15, 10, 0, 10);
  meNumClusters_ = BookHist1D(ibooker, "The Number of Clusters", 8, 1, 9);
  meNumSimHits_ = BookHist1D(ibooker, "The Number of SimHits", 25, 0, 26);
  meNumRecHits_ = BookHist1D(ibooker, "The Number of RecHits", 25, 0, 26);
  meSimHitBareLocalPhi_ = BookHist1D(ibooker, "Bare Local Phi of SimHits", 100, -3*TMath::Pi(), 3*TMath::Pi());
  meSimHitLocalPhi_ = BookHist1D(ibooker, "Local Phi of SimHits", 100, -3*TMath::Pi(), 3*TMath::Pi());
  meRecHitLocalPhi_ = BookHist1D(ibooker, "Local Phi of RecHits", 100, -1 * TMath::Pi() / 18, TMath::Pi() / 18);


  /************
   * 
   ****************************/
  meMatChamber_ = BookHist1D(ibooker, "Matching Case - Chamber Id", 15, 0, 30);
  meMisChamber_ = BookHist1D(ibooker, "Mismatching Case - Chamber Id", 15, 0, 30);

  LogDebug("GEMCosmicMuonStandSim")<<"Booking End.\n";
}


Int_t GEMCosmicMuonStandSim::GetVFATId(Float_t x, const GEMEtaPartition* roll) {
  /* ambig pt in boundaries.
   */
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
  e.getByToken( recHitToken_, gemRecHits);
  edm::Handle<edm::PSimHitContainer> gemSimHits;
  e.getByToken( simHitToken_, gemSimHits);
  
  if (not gemRecHits.isValid()) {
    edm::LogError("GEMCosmicMuonStandSim") << "Cannot get strips by Token RecHits Token.\n";
    return ;
  }

  // if( isMC) 
  Int_t numSimHits = gemSimHits->size();
  if(numSimHits == 0) return ;
  Int_t numRecHits = std::distance(gemRecHits->begin(), gemRecHits->end());
  if(numRecHits == 0) return ;

  meNumSimHits_->Fill(numSimHits);
  meNumRecHits_->Fill(numRecHits);

  for(edm::PSimHitContainer::const_iterator simHit = gemSimHits->begin(); simHit != gemSimHits->end(); ++simHit)
  {
    Local3DPoint simHitLP = simHit->localPosition();
    GEMDetId simDetId(simHit->detUnitId());

    // XXX +1 ?
    Int_t simFiredStrip = GEMGeometry_->etaPartition(simDetId)->strip(simHit->entryPoint()) + 1;

    const GEMEtaPartition* kSimRoll = GEMGeometry_->etaPartition(simDetId);
    Int_t simVFATId = GetVFATId(simHitLP.x(), kSimRoll);
    Int_t simChamberIdx = GetChamberIndex(simDetId.chamber());

    Int_t y_overall_vfat = 3 * simDetId.roll() + simVFATId - 3; // 1 ~ 24
    meVFATTotal_->Fill(simChamberIdx, y_overall_vfat);
    meVFATTotalPerChamber_[simChamberIdx]->Fill(simVFATId, simDetId.roll());

    GEMRecHitCollection::range range = gemRecHits->get(simDetId);

    Int_t numClusters = std::distance(range.first, range.second);
    meNumClusters_->Fill(numClusters);
    // if(numClusters == 0) continue;

    Bool_t isMatching = false;
    for(GEMRecHitCollection::const_iterator recHit = range.first; recHit != range.second; ++recHit)
    {
      Int_t recFirstFiredStrip = recHit->firstClusterStrip();
      Int_t recCLS = recHit->clusterSize();
      Int_t recLastFiredStrip = recFirstFiredStrip + recCLS - 1;

      isMatching = (simFiredStrip >= recFirstFiredStrip) and (simFiredStrip <= recLastFiredStrip);
      if( isMatching )
      {
        LocalPoint recHitLP = recHit->localPosition();
        GEMDetId recDetId = recHit->gemId();
        // GlobalPoint recHitGP = GEMGeometry_->idToDet(recDetId)->surface().toGlobal(recHitLP);

        const GEMEtaPartition* kRecRoll = GEMGeometry_->etaPartition(recDetId);
        
        Float_t sim_hit_local_phi = GetLocalPhi(kRecRoll->strip(simHitLP));
        Float_t rec_hit_local_phi = GetLocalPhi(kRecRoll->strip(recHitLP));
        Float_t residual_local_phi = rec_hit_local_phi - sim_hit_local_phi;

        // Int_t recVFATId = GetVFATId(recHitLP.x(), kRecRoll);
        // Int_t recChamberIdx = GetChamberIndex(recDetId.chamber());
        meVFATPassed_->Fill(simChamberIdx, y_overall_vfat);
        meVFATPassedPerChamber_[simChamberIdx]->Fill(simVFATId, simDetId.roll());

        Float_t residual_local_x = recHitLP.x() - simHitLP.x();
        Float_t residual_local_y = recHitLP.y() - simHitLP.y();
        Float_t error_x = recHit->localPositionError().xx();
        Float_t error_y = recHit->localPositionError().yy();
        Float_t pull_x = residual_local_x / error_x;
        Float_t pull_y = residual_local_y / error_y;

        meResidualLocalX_->Fill(residual_local_x);
        meResidualLocalY_->Fill(residual_local_y);
        meResidualLocalPhi_->Fill(residual_local_phi);
        meLocalPosErrorX_->Fill(error_x);
        meLocalPosErrorY_->Fill(error_y);
        mePullLocalX_->Fill(pull_x);
        mePullLocalY_->Fill(pull_y);

        meCLS_->Fill(recCLS);
        meCLSvsChamber_->Fill(simChamberIdx, recCLS);
        meNumClustersvsChamber_->Fill(simChamberIdx, numClusters);
        meSimHitBareLocalPhi_->Fill(simHitLP.phi());
        meSimHitLocalPhi_->Fill(sim_hit_local_phi);
        meRecHitLocalPhi_->Fill(rec_hit_local_phi);
        break;
      } // MATCING IF STATEMENT END

      if(isMatching)
      {
        meMatChamber_->Fill(simDetId.chamber());
      }
      else
      {
        meMisChamber_->Fill(simDetId.chamber());
      }


    } // RECHIT LOOP END
  } // SIMHIT LOOP END


 
 }

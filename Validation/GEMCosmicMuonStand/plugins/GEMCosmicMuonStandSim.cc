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

#include <vector>
#include <iostream>
#include <iterator> // std::distance
#include <algorithm> // std::find
#include <numeric> // std::iota
#include <string>

using namespace std;

GEMCosmicMuonStandSim::GEMCosmicMuonStandSim(const edm::ParameterSet& cfg) {
  simHitToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simHitToken"));
  recHitToken_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitToken"));
}

MonitorElement* GEMCosmicMuonStandSim::BookHist1D(DQMStore::IBooker &ibooker,
                                                  const char* name, const char* label,
                                                  unsigned int row, unsigned int coll,
                                                  unsigned int layer_num, unsigned int vfat_num,
                                                  const unsigned int Nbin, const Float_t xMin, const Float_t xMax) {
  string hist_name,hist_label;
  return ibooker.book1D( hist_name, hist_label,Nbin,xMin,xMax ); 
}

MonitorElement* GEMCosmicMuonStandSim::BookHist1D(DQMStore::IBooker &ibooker,
                                                  const char* name, const char* label,
                                                  const unsigned int Nbin, const Float_t xMin, const Float_t xMax) {
  return ibooker.book1D( name, label,Nbin,xMin,xMax );
}


void GEMCosmicMuonStandSim::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup) {
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry*  GEMGeometry_ = &*hGeom;
  if ( GEMGeometry_ == nullptr) return ;  

  LogDebug("GEMCosmicMuonStandSim")<<"Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("GEMCosmicMuonStandSim");
  LogDebug("GEMCosmicMuonStandSim")<<"ibooker set current folder\n";

  TString vfat_name_fmt = "gem_vfat_%s_%dth_chamber";


  for(Int_t chamber_id = 1; chamber_id <= 29; chamber_id += 2) {
    Int_t index = GetChamberIndex(chamber_id);

    TString total_name = TString::Format(vfat_name_fmt, "total", chamber_id);
    TString total_title = TString::Format("Total events of %d GEM Chamber", chamber_id);
    gem_vfat_total_[index] = ibooker.book2D(total_name, total_title, 8, 1, 8+1, 3, 1, 3+1);

    TString passed_name = TString::Format(vfat_name_fmt, "passed", chamber_id);
    TString passed_title = TString::Format("Passed events of %d GEM Chamber", chamber_id);
    gem_vfat_passed_[index] = ibooker.book2D(passed_name, passed_title, 8, 1, 8+1, 3, 1, 3+1);
  }
 
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


void GEMCosmicMuonStandSim::analyze(const edm::Event& e,const edm::EventSetup& iSetup) {
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
 for(edm::PSimHitContainer::const_iterator simHit = gemSimHits->begin();
                                           simHit != gemSimHits->end();
                                           ++simHit) {

    Local3DPoint simHitLP = simHit->localPosition();
    GEMDetId simDetId(simHit->detUnitId());

    GlobalPoint simHitGP = GEMGeometry_->idToDet(simDetId)->surface().toGlobal(simHitLP);

    //
    //Int_t simFiredStrip = GEMGeometry_->etaPartition(simDetId)->strip(simHitLP);
    Int_t simFiredStrip = GEMGeometry_->etaPartition(simDetId)->strip(simHit->entryPoint());
    //Int_t simFiredStrip = GEMGeometry_->etaPartition(simDetId)->strip(simHit->exitPoint());

    GEMRecHitCollection::range range = gemRecHits->get(simDetId);
    for(GEMRecHitCollection::const_iterator recHit = range.first;
                                            recHit != range.second;
                                            ++recHit) {

      LocalPoint recHitLP = recHit->localPosition();
      GEMDetId recDetId = recHit->gemId();
      GlobalPoint recHitGP = GEMGeometry_->idToDet(recDetId)->surface().toGlobal(recHitLP);

      const GEMEtaPartition* kRecRoll = GEMGeometry_->etaPartition(recDetId);
      Int_t recVFATId = GetVFATId(recHitLP.x(), kRecRoll);
      if(recVFATId == -1) continue;

      Int_t recChamberIdx = GetChamberIndex(recDetId.chamber());
      std::cout << "Chamber Index: " << recChamberIdx << std::endl;
      gem_vfat_total_[recChamberIdx]->Fill(recDetId.roll(), recVFATId);

      // XXX SimHit RecHit Matching
      // FIXME Discard useless conditions
      if(simDetId.layer() != recDetId.layer()) continue;
      if(simDetId.chamber() != recDetId.chamber()) continue;
      if(simDetId.roll() != recDetId.roll()) continue;

      Int_t recHitCLS = recHit->clusterSize();
      Int_t recFirstFiredStrip = recHit->firstClusterStrip();
      Int_t recLastFiredStrip = recFirstFiredStrip + recHitCLS;

      if((simFiredStrip < recFirstFiredStrip) or (simFiredStrip > recLastFiredStrip)) continue;

      gem_vfat_passed_[recChamberIdx]->Fill(recDetId.roll(), recVFATId);

      std::cout << simHitGP.x() << " " << recHitGP.x() << std::endl;
    }
  }

  // if( not isMC )

 
 }

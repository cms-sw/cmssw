#include "GEMCosmicMuonStandEfficiency.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

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

}

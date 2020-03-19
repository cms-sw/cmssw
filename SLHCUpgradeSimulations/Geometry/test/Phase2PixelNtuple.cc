/*
  \class Phase2PixelNtuple
*/

// DataFormats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Geometry
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"

// For ROOT
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TTree.h>

// STD
#include <memory>
#include <string>
#include <iostream>

// CLHEP (for speed of light)
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

//#define EDM_ML_DEBUG

using namespace std;
using namespace edm;
using namespace reco;

class Phase2PixelNtuple : public edm::one::EDAnalyzer<> {
public:
  explicit Phase2PixelNtuple(const edm::ParameterSet& conf);
  virtual ~Phase2PixelNtuple();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

protected:
  void fillEvt(const edm::Event&);
  void fillPRecHit(const int detid_db,
                   const int subid,
                   const int layer_num,
                   const int ladder_num,
                   const int module_num,
                   const int disk_num,
                   const int blade_num,
                   const int panel_num,
                   const int side_num,
                   SiPixelRecHitCollection::DetSet::const_iterator pixeliter,
                   const int num_simhit,
                   const PSimHit* closest_simhit,
                   const GeomDet* PixGeom);
  void fillPRecHit(const int detid_db,
                   const int subid,
                   const int layer_num,
                   const int ladder_num,
                   const int module_num,
                   const int disk_num,
                   const int blade_num,
                   const int panel_num,
                   const int side_num,
                   trackingRecHit_iterator pixeliter,
                   const int num_simhit,
                   const PSimHit* closest_simhit,
                   const GeomDet* PixGeom);
  std::pair<float, float> computeAnglesFromDetPosition(const SiPixelCluster& cl,
                                                       const PixelTopology& top,
                                                       const GeomDetUnit& det) const;

private:
  edm::ParameterSet const conf_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelRecHit>> pixelRecHits_token;
  edm::EDGetTokenT<edm::View<reco::Track>> token_recoTracks;
  bool verbose_;
  bool picky_;
  static const int DGPERCLMAX = 100;

  //--- Structures for ntupling:
  struct evt {
    int run;
    int evtnum;

    void init();
  } evt_;

  struct RecHit {
    int pdgid;
    int process;
    float q;
    float x;
    float y;
    float xx;
    float xy;
    float yy;
    float row;
    float col;
    float gx;
    float gy;
    float gz;
    int subid, module;
    int layer, ladder;             // BPix
    int disk, blade, panel, side;  // FPix
    int nsimhit;
    int spreadx, spready;
    float hx, hy, ht;
    float hrow, hcol;
    float tx, ty, tz;
    float theta, phi;

    // detector topology
    int nRowsInDet;
    int nColsInDet;
    float pitchx;
    float pitchy;
    float thickness;

    // local angles from det position
    float cotAlphaFromDet, cotBetaFromDet;

    // digis
    int fDgN;
    int fDgRow[DGPERCLMAX], fDgCol[DGPERCLMAX];
    int fDgDetId[DGPERCLMAX];
    float fDgAdc[DGPERCLMAX], fDgCharge[DGPERCLMAX];

    void init();
  } recHit_;

  edm::Service<TFileService> tFileService;
  TTree* pixeltree_;
  TTree* pixeltreeOnTrack_;
};

Phase2PixelNtuple::Phase2PixelNtuple(edm::ParameterSet const& conf)
    : trackerHitAssociatorConfig_(conf, consumesCollector()),
      pixelRecHits_token(consumes<edmNew::DetSetVector<SiPixelRecHit>>(edm::InputTag("siPixelRecHits"))),
      token_recoTracks(consumes<edm::View<reco::Track>>(conf.getParameter<edm::InputTag>("trackProducer"))),
      verbose_(conf.getUntrackedParameter<bool>("verbose", false)),
      picky_(conf.getUntrackedParameter<bool>("picky", false)),
      pixeltree_(0),
      pixeltreeOnTrack_(0) {}

Phase2PixelNtuple::~Phase2PixelNtuple() {}

void Phase2PixelNtuple::endJob() {}

void Phase2PixelNtuple::beginJob() {
  pixeltree_ = tFileService->make<TTree>("PixelNtuple", "Pixel hit analyzer ntuple");
  pixeltreeOnTrack_ = tFileService->make<TTree>("PixelNtupleOnTrack", "On-Track Pixel hit analyzer ntuple");

  int bufsize = 64000;
  //Common Branch
  pixeltree_->Branch("evt", &evt_, "run/I:evtnum/I", bufsize);
  pixeltree_->Branch("pdgid", &recHit_.pdgid, "pdgid/I");
  pixeltree_->Branch("process", &recHit_.process, "process/I");
  pixeltree_->Branch("q", &recHit_.q, "q/F");
  pixeltree_->Branch("x", &recHit_.x, "x/F");
  pixeltree_->Branch("y", &recHit_.y, "y/F");
  pixeltree_->Branch("xx", &recHit_.xx, "xx/F");
  pixeltree_->Branch("xy", &recHit_.xy, "xy/F");
  pixeltree_->Branch("yy", &recHit_.yy, "yy/F");
  pixeltree_->Branch("row", &recHit_.row, "row/F");
  pixeltree_->Branch("col", &recHit_.col, "col/F");
  pixeltree_->Branch("gx", &recHit_.gx, "gx/F");
  pixeltree_->Branch("gy", &recHit_.gy, "gy/F");
  pixeltree_->Branch("gz", &recHit_.gz, "gz/F");
  pixeltree_->Branch("subid", &recHit_.subid, "subid/I");
  pixeltree_->Branch("module", &recHit_.module, "module/I");
  pixeltree_->Branch("layer", &recHit_.layer, "layer/I");
  pixeltree_->Branch("ladder", &recHit_.ladder, "ladder/I");
  pixeltree_->Branch("disk", &recHit_.disk, "disk/I");
  pixeltree_->Branch("blade", &recHit_.blade, "blade/I");
  pixeltree_->Branch("panel", &recHit_.panel, "panel/I");
  pixeltree_->Branch("side", &recHit_.side, "side/I");
  pixeltree_->Branch("nsimhit", &recHit_.nsimhit, "nsimhit/I");
  pixeltree_->Branch("spreadx", &recHit_.spreadx, "spreadx/I");
  pixeltree_->Branch("spready", &recHit_.spready, "spready/I");
  pixeltree_->Branch("hx", &recHit_.hx, "hx/F");
  pixeltree_->Branch("hy", &recHit_.hy, "hy/F");
  pixeltree_->Branch("ht", &recHit_.ht, "ht/F");
  pixeltree_->Branch("hrow", &recHit_.hrow, "hrow/F");
  pixeltree_->Branch("hcol", &recHit_.hcol, "hcol/F");
  pixeltree_->Branch("tx", &recHit_.tx, "tx/F");
  pixeltree_->Branch("ty", &recHit_.ty, "ty/F");
  pixeltree_->Branch("tz", &recHit_.tz, "tz/F");
  pixeltree_->Branch("theta", &recHit_.theta, "theta/F");
  pixeltree_->Branch("phi", &recHit_.phi, "phi/F");
  pixeltree_->Branch("nRowsInDet", &recHit_.nRowsInDet, "nRowsInDet/I");
  pixeltree_->Branch("nColsInDet", &recHit_.nColsInDet, "nColsInDet/I");
  pixeltree_->Branch("pitchx", &recHit_.pitchx, "pitchx/F");
  pixeltree_->Branch("pitchy", &recHit_.pitchy, "pitchy/F");
  pixeltree_->Branch("thickness", &recHit_.thickness, "thickness/F");
  pixeltree_->Branch("cotAlphaFromDet", &recHit_.cotAlphaFromDet, "cotAlphaFromDet/F");
  pixeltree_->Branch("cotBetaFromDet", &recHit_.cotBetaFromDet, "cotBetaFraomDet/F");

  pixeltree_->Branch("DgN", &recHit_.fDgN, "DgN/I");
  pixeltree_->Branch("DgRow", recHit_.fDgRow, "DgRow[DgN]/I");
  pixeltree_->Branch("DgCol", recHit_.fDgCol, "DgCol[DgN]/I");
  pixeltree_->Branch("DgDetId", recHit_.fDgDetId, "DgDetId[DgN]/I");
  pixeltree_->Branch("DgAdc", recHit_.fDgAdc, "DgAdc[DgN]/F");
  pixeltree_->Branch("DgCharge", recHit_.fDgCharge, "DgCharge[DgN]/F");

  //Common Branch (on-track)
  pixeltreeOnTrack_->Branch("evt", &evt_, "run/I:evtnum/I", bufsize);
  pixeltreeOnTrack_->Branch("pdgid", &recHit_.pdgid, "pdgid/I");
  pixeltreeOnTrack_->Branch("process", &recHit_.process, "process/I");
  pixeltreeOnTrack_->Branch("q", &recHit_.q, "q/F");
  pixeltreeOnTrack_->Branch("x", &recHit_.x, "x/F");
  pixeltreeOnTrack_->Branch("y", &recHit_.y, "y/F");
  pixeltreeOnTrack_->Branch("xx", &recHit_.xx, "xx/F");
  pixeltreeOnTrack_->Branch("xy", &recHit_.xy, "xy/F");
  pixeltreeOnTrack_->Branch("yy", &recHit_.yy, "yy/F");
  pixeltreeOnTrack_->Branch("row", &recHit_.row, "row/F");
  pixeltreeOnTrack_->Branch("col", &recHit_.col, "col/F");
  pixeltreeOnTrack_->Branch("gx", &recHit_.gx, "gx/F");
  pixeltreeOnTrack_->Branch("gy", &recHit_.gy, "gy/F");
  pixeltreeOnTrack_->Branch("gz", &recHit_.gz, "gz/F");
  pixeltreeOnTrack_->Branch("subid", &recHit_.subid, "subid/I");
  pixeltreeOnTrack_->Branch("module", &recHit_.module, "module/I");
  pixeltreeOnTrack_->Branch("layer", &recHit_.layer, "layer/I");
  pixeltreeOnTrack_->Branch("ladder", &recHit_.ladder, "ladder/I");
  pixeltreeOnTrack_->Branch("disk", &recHit_.disk, "disk/I");
  pixeltreeOnTrack_->Branch("blade", &recHit_.blade, "blade/I");
  pixeltreeOnTrack_->Branch("panel", &recHit_.panel, "panel/I");
  pixeltreeOnTrack_->Branch("side", &recHit_.side, "side/I");
  pixeltreeOnTrack_->Branch("nsimhit", &recHit_.nsimhit, "nsimhit/I");
  pixeltreeOnTrack_->Branch("spreadx", &recHit_.spreadx, "spreadx/I");
  pixeltreeOnTrack_->Branch("spready", &recHit_.spready, "spready/I");
  pixeltreeOnTrack_->Branch("hx", &recHit_.hx, "hx/F");
  pixeltreeOnTrack_->Branch("hy", &recHit_.hy, "hy/F");
  pixeltreeOnTrack_->Branch("ht", &recHit_.ht, "ht/F");
  pixeltreeOnTrack_->Branch("hrow", &recHit_.hrow, "hrow/F");
  pixeltreeOnTrack_->Branch("hcol", &recHit_.hcol, "hcol/F");
  pixeltreeOnTrack_->Branch("tx", &recHit_.tx, "tx/F");
  pixeltreeOnTrack_->Branch("ty", &recHit_.ty, "ty/F");
  pixeltreeOnTrack_->Branch("tz", &recHit_.tz, "tz/F");
  pixeltreeOnTrack_->Branch("theta", &recHit_.theta, "theta/F");
  pixeltreeOnTrack_->Branch("phi", &recHit_.phi, "phi/F");
  pixeltreeOnTrack_->Branch("nRowsInDet", &recHit_.nRowsInDet, "nRowsInDet/I");
  pixeltreeOnTrack_->Branch("nColsInDet", &recHit_.nColsInDet, "nColsInDet/I");
  pixeltreeOnTrack_->Branch("pitchx", &recHit_.pitchx, "pitchx/F");
  pixeltreeOnTrack_->Branch("pitchy", &recHit_.pitchy, "pitchy/F");
  pixeltreeOnTrack_->Branch("thickness", &recHit_.thickness, "thickness/F");
  pixeltreeOnTrack_->Branch("cotAlphaFromDet", &recHit_.cotAlphaFromDet, "cotAlphaFromDet/F");
  pixeltreeOnTrack_->Branch("cotBetaFromDet", &recHit_.cotBetaFromDet, "cotBetaFraomDet/F");

  pixeltreeOnTrack_->Branch("DgN", &recHit_.fDgN, "DgN/I");
  pixeltreeOnTrack_->Branch("DgRow", recHit_.fDgRow, "DgRow[DgN]/I");
  pixeltreeOnTrack_->Branch("DgCol", recHit_.fDgCol, "DgCol[DgN]/I");
  pixeltreeOnTrack_->Branch("DgDetId", recHit_.fDgDetId, "DgDetId[DgN]/I");
  pixeltreeOnTrack_->Branch("DgAdc", recHit_.fDgAdc, "DgAdc[DgN]/F");
  pixeltreeOnTrack_->Branch("DgCharge", recHit_.fDgCharge, "DgCharge[DgN]/F");
}

// Functions that gets called by framework every event
void Phase2PixelNtuple::analyze(const edm::Event& e, const edm::EventSetup& es) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // geometry setup
  edm::ESHandle<TrackerGeometry> geometry;

  es.get<TrackerDigiGeometryRecord>().get(geometry);
  const TrackerGeometry* theGeometry = &(*geometry);

  std::vector<PSimHit> matched;
  const PSimHit* closest_simhit = nullptr;

  edm::Handle<SiPixelRecHitCollection> recHitColl;
  e.getByToken(pixelRecHits_token, recHitColl);
  // for finding matched simhit
  TrackerHitAssociator associate(e, trackerHitAssociatorConfig_);

  if ((recHitColl.product())->dataSize() > 0) {
    std::string detname;

    evt_.init();
    fillEvt(e);

    // Loop over Detector IDs
    for (auto recHitIdIterator : *(recHitColl.product())) {
      SiPixelRecHitCollection::DetSet detset = recHitIdIterator;

      if (detset.empty())
        continue;
      DetId detId = DetId(detset.detId());  // Get the Detid object

      const GeomDet* geomDet(theGeometry->idToDet(detId));

      // Loop over rechits for this detid
      for (auto iterRecHit : detset) {
        // get matched simhit
        matched.clear();
        matched = associate.associateHit(iterRecHit);
        if (!matched.empty()) {
          float closest = 9999.9;
          LocalPoint lp = iterRecHit.localPosition();
          float rechit_x = lp.x();
          float rechit_y = lp.y();
          //loop over simhits and find closest
          for (auto const& m : matched) {
            float sim_x1 = m.entryPoint().x();
            float sim_x2 = m.exitPoint().x();
            float sim_xpos = 0.5 * (sim_x1 + sim_x2);
            float sim_y1 = m.entryPoint().y();
            float sim_y2 = m.exitPoint().y();
            float sim_ypos = 0.5 * (sim_y1 + sim_y2);

            float x_res = sim_xpos - rechit_x;
            float y_res = sim_ypos - rechit_y;
            float dist = sqrt(x_res * x_res + y_res * y_res);
            if (dist < closest) {
              closest = dist;
              closest_simhit = &m;
            }
          }  // end of simhit loop
        }    // end matched emtpy
        unsigned int subid = detId.subdetId();
        int detid_db = detId.rawId();
        int layer_num = -99, ladder_num = -99, module_num = -99, disk_num = -99, blade_num = -99, panel_num = -99,
            side_num = -99;
        if ((subid == PixelSubdetector::PixelBarrel) || (subid == PixelSubdetector::PixelEndcap)) {
          // 1 = PXB, 2 = PXF
          if (subid == PixelSubdetector::PixelBarrel) {
            layer_num = tTopo->pxbLayer(detId.rawId());
            ladder_num = tTopo->pxbLadder(detId.rawId());
            module_num = tTopo->pxbModule(detId.rawId());
#ifdef EDM_ML_DEBUG
            std::cout << "\ndetId = " << subid << " : " << tTopo->pxbLayer(detId.rawId()) << " , "
                      << tTopo->pxbLadder(detId.rawId()) << " , " << tTopo->pxbModule(detId.rawId());
#endif
          } else if (subid == PixelSubdetector::PixelEndcap) {
            module_num = tTopo->pxfModule(detId());
            disk_num = tTopo->pxfDisk(detId());
            blade_num = tTopo->pxfBlade(detId());
            panel_num = tTopo->pxfPanel(detId());
            side_num = tTopo->pxfSide(detId());
          }
          int num_simhit = matched.size();
          recHit_.init();
          fillPRecHit(detid_db,
                      subid,
                      layer_num,
                      ladder_num,
                      module_num,
                      disk_num,
                      blade_num,
                      panel_num,
                      side_num,
                      &iterRecHit,
                      num_simhit,
                      closest_simhit,
                      geomDet);
          pixeltree_->Fill();
        }
      }  // end of rechit loop
    }    // end of detid loop
  }      // end of loop test on recHitColl size

  // Now loop over recotracks
  edm::Handle<View<reco::Track>> trackCollection;
  e.getByToken(token_recoTracks, trackCollection);

  if (!trackCollection.isValid()) {
    if (picky_) {
      throw cms::Exception("ProductNotValid") << "TrackCollection product not valid";
    } else {
      ;
    }
  } else {
    int rT = 0;
    for (View<reco::Track>::size_type i = 0; i < trackCollection->size(); ++i) {
      ++rT;
      RefToBase<reco::Track> track(trackCollection, i);
      int iT = 0;
#ifdef EDM_ML_DEBUG
      std::cout << " num of hits for track " << rT << " = " << track->recHitsSize() << std::endl;
#endif
      for (trackingRecHit_iterator ih = track->recHitsBegin(); ih != track->recHitsEnd(); ++ih) {
        ++iT;
        TrackingRecHit* hit = (*ih)->clone();
        const DetId& detId = hit->geographicalId();
        const GeomDet* geomDet(theGeometry->idToDet(detId));

        const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit);

        if (pixhit) {
          if (pixhit->isValid()) {
            // get matched simhit
            matched.clear();
            matched = associate.associateHit(*pixhit);

            if (!matched.empty()) {
              float closest = 9999.9;
              LocalPoint lp = pixhit->localPosition();
              float rechit_x = lp.x();
              float rechit_y = lp.y();

              //loop over simhits and find closest
              //for (std::vector<PSimHit>::const_iterator m = matched.begin(); m<matched.end(); m++)
              for (auto const& m : matched) {
                float sim_x1 = m.entryPoint().x();
                float sim_x2 = m.exitPoint().x();
                float sim_xpos = 0.5 * (sim_x1 + sim_x2);
                float sim_y1 = m.entryPoint().y();
                float sim_y2 = m.exitPoint().y();
                float sim_ypos = 0.5 * (sim_y1 + sim_y2);

                float x_res = sim_xpos - rechit_x;
                float y_res = sim_ypos - rechit_y;
                float dist = sqrt(x_res * x_res + y_res * y_res);
                if (dist < closest) {
                  closest = dist;
                  closest_simhit = &m;
                }
              }  // end of simhit loop
            }    // end matched emtpy

            int num_simhit = matched.size();

            int layer_num = -99, ladder_num = -99, module_num = -99, disk_num = -99, blade_num = -99, panel_num = -99,
                side_num = -99;

            unsigned int subid = detId.subdetId();
            int detid_db = detId.rawId();
            if ((subid == PixelSubdetector::PixelBarrel) || (subid == PixelSubdetector::PixelEndcap)) {
              // 1 = PXB, 2 = PXF
              if (subid == PixelSubdetector::PixelBarrel) {
                layer_num = tTopo->pxbLayer(detId.rawId());
                ladder_num = tTopo->pxbLadder(detId.rawId());
                module_num = tTopo->pxbModule(detId.rawId());
#ifdef EDM_ML_DEBUG
                std::cout << "\ndetId = " << subid << " : " << tTopo->pxbLayer(detId.rawId()) << " , "
                          << tTopo->pxbLadder(detId.rawId()) << " , " << tTopo->pxbModule(detId.rawId()) << std::endl;
#endif
              } else if (subid == PixelSubdetector::PixelEndcap) {
                module_num = tTopo->pxfModule(detId());
                disk_num = tTopo->pxfDisk(detId());
                blade_num = tTopo->pxfBlade(detId());
                panel_num = tTopo->pxfPanel(detId());
                side_num = tTopo->pxfSide(detId());
              }

              recHit_.init();
              fillPRecHit(detid_db,
                          subid,
                          layer_num,
                          ladder_num,
                          module_num,
                          disk_num,
                          blade_num,
                          panel_num,
                          side_num,
                          ih,
                          num_simhit,
                          closest_simhit,
                          geomDet);
              pixeltreeOnTrack_->Fill();
            }  // if ( (subid==1)||(subid==2) )
          }    // if SiPixelHit is valid
        }      // if cast is possible to SiPixelHit
        delete pixhit;
      }  //end of loop on tracking rechits
    }    // end of loop on recotracks
  }      // else track collection is valid
}  // end analyze function

// Function for filling in all the rechits
// I know it is lazy to pass everything, but I'm doing it anyway. -EB
void Phase2PixelNtuple::fillPRecHit(const int detid_db,
                                    const int subid,
                                    const int layer_num,
                                    const int ladder_num,
                                    const int module_num,
                                    const int disk_num,
                                    const int blade_num,
                                    const int panel_num,
                                    const int side_num,
                                    SiPixelRecHitCollection::DetSet::const_iterator pixeliter,
                                    const int num_simhit,
                                    const PSimHit* closest_simhit,
                                    const GeomDet* PixGeom) {
  LocalPoint lp = pixeliter->localPosition();
  LocalError le = pixeliter->localPositionError();

  recHit_.x = lp.x();
  recHit_.y = lp.y();
  recHit_.xx = le.xx();
  recHit_.xy = le.xy();
  recHit_.yy = le.yy();
  GlobalPoint GP = PixGeom->surface().toGlobal(pixeliter->localPosition());
  recHit_.gx = GP.x();
  recHit_.gy = GP.y();
  recHit_.gz = GP.z();
  GlobalPoint GP0 = PixGeom->surface().toGlobal(LocalPoint(0, 0, 0));
  recHit_.theta = GP0.theta();
  recHit_.phi = GP0.phi();

  SiPixelRecHit::ClusterRef const& Cluster = pixeliter->cluster();
  recHit_.q = Cluster->charge();
  recHit_.spreadx = Cluster->sizeX();
  recHit_.spready = Cluster->sizeY();

  recHit_.subid = subid;
  recHit_.nsimhit = num_simhit;

  recHit_.layer = layer_num;
  recHit_.ladder = ladder_num;
  recHit_.module = module_num;
  recHit_.module = module_num;
  recHit_.disk = disk_num;
  recHit_.blade = blade_num;
  recHit_.panel = panel_num;
  recHit_.side = side_num;

  /*-- module topology --*/
  const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(PixGeom);
  const PixelTopology* topol = &(theGeomDet->specificTopology());
  recHit_.nRowsInDet = topol->nrows();
  recHit_.nColsInDet = topol->ncolumns();
  recHit_.pitchx = topol->pitch().first;
  recHit_.pitchy = topol->pitch().second;
  recHit_.thickness = theGeomDet->surface().bounds().thickness();

  MeasurementPoint mp = topol->measurementPosition(LocalPoint(recHit_.x, recHit_.y));
  recHit_.row = mp.x();
  recHit_.col = mp.y();

  if (Cluster.isNonnull()) {
    // compute local angles from det position
    std::pair<float, float> local_angles = computeAnglesFromDetPosition(*Cluster, *topol, *theGeomDet);
    recHit_.cotAlphaFromDet = local_angles.first;
    recHit_.cotBetaFromDet = local_angles.second;

    // -- Get digis of this cluster
    const std::vector<SiPixelCluster::Pixel>& pixvector = Cluster->pixels();
#ifdef EDM_ML_DEBUG
    std::cout << "  Found " << pixvector.size() << " pixels for this cluster " << std::endl;
#endif
    for (unsigned int i = 0; i < pixvector.size(); ++i) {
      if (recHit_.fDgN > DGPERCLMAX - 1)
        break;
      SiPixelCluster::Pixel holdpix = pixvector[i];

      recHit_.fDgRow[recHit_.fDgN] = holdpix.x;
      recHit_.fDgCol[recHit_.fDgN] = holdpix.y;
#ifdef EDM_ML_DEBUG
      std::cout << "holdpix " << holdpix.x << " " << holdpix.y << std::endl;
#endif
      recHit_.fDgDetId[recHit_.fDgN] = detid_db;
      recHit_.fDgAdc[recHit_.fDgN] = -99.;
      recHit_.fDgCharge[recHit_.fDgN] = holdpix.adc / 1000.;
      ++recHit_.fDgN;
    }
  }  // if ( Cluster.isNonnull() )

#ifdef EDM_ML_DEBUG
  std::cout << "num_simhit = " << num_simhit << std::endl;
#endif
  if (num_simhit > 0) {
    recHit_.pdgid = closest_simhit->particleType();
    recHit_.process = closest_simhit->processType();

    float sim_x1 = closest_simhit->entryPoint().x();
    float sim_x2 = closest_simhit->exitPoint().x();
    recHit_.hx = 0.5 * (sim_x1 + sim_x2);
    float sim_y1 = closest_simhit->entryPoint().y();
    float sim_y2 = closest_simhit->exitPoint().y();
    recHit_.hy = 0.5 * (sim_y1 + sim_y2);

    float time_to_detid_ns = GP0.mag() / (CLHEP::c_light * CLHEP::ns / CLHEP::cm);  // speed of light in ns
    recHit_.ht = closest_simhit->timeOfFlight() - time_to_detid_ns;

    recHit_.tx = closest_simhit->localDirection().x();
    recHit_.ty = closest_simhit->localDirection().y();
    recHit_.tz = closest_simhit->localDirection().z();

    MeasurementPoint hmp = topol->measurementPosition(LocalPoint(recHit_.hx, recHit_.hy));
    recHit_.hrow = hmp.x();
    recHit_.hcol = hmp.y();

    // Leaving the comment below, useful for future reference
    // alpha: angle with respect to local x axis in local (x,z) plane
    // float cotalpha = sim_xdir/sim_zdir;
    // beta: angle with respect to local y axis in local (y,z) plane
    // float cotbeta = sim_ydir/sim_zdir;

#ifdef EDM_ML_DEBUG
    std::cout << "num_simhit x, y = " << 0.5 * (sim_x1 + sim_x2) << " " << 0.5 * (sim_y1 + sim_y2) << std::endl;
#endif
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Found RecHit in " << subid
            << " global x/y/z : " << PixGeom->surface().toGlobal(pixeliter->localPosition()).x() << " "
            << PixGeom->surface().toGlobal(pixeliter->localPosition()).y() << " "
            << PixGeom->surface().toGlobal(pixeliter->localPosition()).z() << std::endl;
#endif
}

// Function for filling in on track rechits
void Phase2PixelNtuple::fillPRecHit(const int detid_db,
                                    const int subid,
                                    const int layer_num,
                                    const int ladder_num,
                                    const int module_num,
                                    const int disk_num,
                                    const int blade_num,
                                    const int panel_num,
                                    const int side_num,
                                    trackingRecHit_iterator ih,
                                    const int num_simhit,
                                    const PSimHit* closest_simhit,
                                    const GeomDet* PixGeom) {
  TrackingRecHit* pixeliter = (*ih)->clone();
  LocalPoint lp = pixeliter->localPosition();
  LocalError le = pixeliter->localPositionError();

  recHit_.x = lp.x();
  recHit_.y = lp.y();
  recHit_.xx = le.xx();
  recHit_.xy = le.xy();
  recHit_.yy = le.yy();
  GlobalPoint GP = PixGeom->surface().toGlobal(pixeliter->localPosition());
  recHit_.gx = GP.x();
  recHit_.gy = GP.y();
  recHit_.gz = GP.z();
  GlobalPoint GP0 = PixGeom->surface().toGlobal(LocalPoint(0, 0, 0));
  recHit_.theta = GP0.theta();
  recHit_.phi = GP0.phi();
  recHit_.subid = subid;

  SiPixelRecHit::ClusterRef const& Cluster = dynamic_cast<const SiPixelRecHit*>(pixeliter)->cluster();
  recHit_.q = Cluster->charge();
  recHit_.spreadx = Cluster->sizeX();
  recHit_.spready = Cluster->sizeY();

  recHit_.nsimhit = num_simhit;

  recHit_.layer = layer_num;
  recHit_.ladder = ladder_num;
  recHit_.module = module_num;
  recHit_.module = module_num;
  recHit_.disk = disk_num;
  recHit_.blade = blade_num;
  recHit_.panel = panel_num;
  recHit_.side = side_num;

  /*-- module topology --*/
  const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(PixGeom);
  const PixelTopology* topol = &(theGeomDet->specificTopology());
  recHit_.nRowsInDet = topol->nrows();
  recHit_.nColsInDet = topol->ncolumns();
  recHit_.pitchx = topol->pitch().first;
  recHit_.pitchy = topol->pitch().second;
  recHit_.thickness = theGeomDet->surface().bounds().thickness();

  if (Cluster.isNonnull()) {
    // compute local angles from det position
    std::pair<float, float> local_angles = computeAnglesFromDetPosition(*Cluster, *topol, *theGeomDet);
    recHit_.cotAlphaFromDet = local_angles.first;
    recHit_.cotBetaFromDet = local_angles.second;

    // -- Get digis of this cluster
    const std::vector<SiPixelCluster::Pixel>& pixvector = Cluster->pixels();
#ifdef EDM_ML_DEBUG
    std::cout << "  Found " << pixvector.size() << " pixels for this cluster " << std::endl;
#endif
    for (unsigned int i = 0; i < pixvector.size(); ++i) {
      if (recHit_.fDgN > DGPERCLMAX - 1)
        break;
      SiPixelCluster::Pixel holdpix = pixvector[i];

      recHit_.fDgRow[recHit_.fDgN] = holdpix.x;
      recHit_.fDgCol[recHit_.fDgN] = holdpix.y;
#ifdef EDM_ML_DEBUG
      std::cout << "holdpix " << holdpix.x << " " << holdpix.y << std::endl;
#endif
      recHit_.fDgDetId[recHit_.fDgN] = detid_db;
      recHit_.fDgAdc[recHit_.fDgN] = -99.;
      recHit_.fDgCharge[recHit_.fDgN] = holdpix.adc / 1000.;
      ++recHit_.fDgN;
    }
  }  // if ( Cluster.isNonnull() )

  if (num_simhit > 0) {
    recHit_.pdgid = closest_simhit->particleType();
    recHit_.process = closest_simhit->processType();

    float sim_x1 = closest_simhit->entryPoint().x();
    float sim_x2 = closest_simhit->exitPoint().x();
    recHit_.hx = 0.5 * (sim_x1 + sim_x2);
    float sim_y1 = closest_simhit->entryPoint().y();
    float sim_y2 = closest_simhit->exitPoint().y();
    recHit_.hy = 0.5 * (sim_y1 + sim_y2);

    float time_to_detid_ns = GP0.mag() / (CLHEP::c_light * CLHEP::ns / CLHEP::cm);  // speed of light in ns
    recHit_.ht = closest_simhit->timeOfFlight() - time_to_detid_ns;

    recHit_.tx = closest_simhit->localDirection().x();
    recHit_.ty = closest_simhit->localDirection().y();
    recHit_.tz = closest_simhit->localDirection().z();

    MeasurementPoint hmp = topol->measurementPosition(LocalPoint(recHit_.hx, recHit_.hy));
    recHit_.hrow = hmp.x();
    recHit_.hcol = hmp.y();

    // Leaving the comment below, useful for future reference
    // alpha: angle with respect to local x axis in local (x,z) plane
    // float cotalpha = sim_xdir/sim_zdir;
    // beta: angle with respect to local y axis in local (y,z) plane
    // float cotbeta = sim_ydir/sim_zdir;

#ifdef EDM_ML_DEBUG
    std::cout << "num_simhit x, y = " << 0.5 * (sim_x1 + sim_x2) << " " << 0.5 * (sim_y1 + sim_y2) << std::endl;
#endif
  }

  delete pixeliter;
}

void Phase2PixelNtuple::fillEvt(const edm::Event& E) {
  evt_.run = E.id().run();
  evt_.evtnum = E.id().event();
}

void Phase2PixelNtuple::evt::init() {
  int dummy_int = 9999;
  run = dummy_int;
  evtnum = dummy_int;
}

void Phase2PixelNtuple::RecHit::init() {
  float dummy_float = 9999.0;

  pdgid = 0;
  process = 0;
  q = dummy_float;
  x = dummy_float;
  y = dummy_float;
  xx = dummy_float;
  xy = dummy_float;
  yy = dummy_float;
  row = dummy_float;
  col = dummy_float;
  gx = dummy_float;
  gy = dummy_float;
  gz = dummy_float;
  nsimhit = 0;
  subid = -99;
  module = -99;
  layer = -99;
  ladder = -99;
  disk = -99;
  blade = -99;
  panel = -99;
  side = -99;
  spreadx = 0;
  spready = 0;
  hx = dummy_float;
  hy = dummy_float;
  ht = dummy_float;
  tx = dummy_float;
  ty = dummy_float;
  tz = dummy_float;
  theta = dummy_float;
  phi = dummy_float;

  fDgN = DGPERCLMAX;
  for (int i = 0; i < fDgN; ++i) {
    fDgRow[i] = fDgCol[i] = -9999;
    fDgAdc[i] = fDgCharge[i] = -9999.;
    //    fDgRoc[i] = fDgRocR[i] = fDgRocC[i] = -9999;
  }
  fDgN = 0;
}

std::pair<float, float> Phase2PixelNtuple::computeAnglesFromDetPosition(const SiPixelCluster& cl,
                                                                        const PixelTopology& theTopol,
                                                                        const GeomDetUnit& theDet) const {
  // get cluster center of gravity (of charge)
  float xcenter = cl.x();
  float ycenter = cl.y();

  // get the cluster position in local coordinates (cm)

  // ggiurgiu@jhu.edu 12/09/2010 : This function is called without track info, therefore there are no track
  // angles to provide here. Call the default localPosition (without track info)
  LocalPoint lp = theTopol.localPosition(MeasurementPoint(xcenter, ycenter));

  // get the cluster position in global coordinates (cm)
  GlobalPoint gp = theDet.surface().toGlobal(lp);
  float gp_mod = sqrt(gp.x() * gp.x() + gp.y() * gp.y() + gp.z() * gp.z());

  // normalize
  float gpx = gp.x() / gp_mod;
  float gpy = gp.y() / gp_mod;
  float gpz = gp.z() / gp_mod;

  // make a global vector out of the global point; this vector will point from the
  // origin of the detector to the cluster
  GlobalVector gv(gpx, gpy, gpz);

  // make local unit vector along local X axis
  const Local3DVector lvx(1.0, 0.0, 0.0);

  // get the unit X vector in global coordinates/
  GlobalVector gvx = theDet.surface().toGlobal(lvx);

  // make local unit vector along local Y axis
  const Local3DVector lvy(0.0, 1.0, 0.0);

  // get the unit Y vector in global coordinates
  GlobalVector gvy = theDet.surface().toGlobal(lvy);

  // make local unit vector along local Z axis
  const Local3DVector lvz(0.0, 0.0, 1.0);

  // get the unit Z vector in global coordinates
  GlobalVector gvz = theDet.surface().toGlobal(lvz);

  // calculate the components of gv (the unit vector pointing to the cluster)
  // in the local coordinate system given by the basis {gvx, gvy, gvz}
  // note that both gv and the basis {gvx, gvy, gvz} are given in global coordinates
  float gv_dot_gvx = gv.x() * gvx.x() + gv.y() * gvx.y() + gv.z() * gvx.z();
  float gv_dot_gvy = gv.x() * gvy.x() + gv.y() * gvy.y() + gv.z() * gvy.z();
  float gv_dot_gvz = gv.x() * gvz.x() + gv.y() * gvz.y() + gv.z() * gvz.z();

  /* all the above is equivalent to  
      const Local3DPoint origin =   theDet->surface().toLocal(GlobalPoint(0,0,0)); // can be computed once...
      auto gvx = lp.x()-origin.x();
      auto gvy = lp.y()-origin.y();
      auto gvz = -origin.z();
   *  normalization not required as only ratio used... 
   */

  // calculate angles
  float cotalpha_ = gv_dot_gvx / gv_dot_gvz;
  float cotbeta_ = gv_dot_gvy / gv_dot_gvz;

  return std::make_pair(cotalpha_, cotbeta_);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2PixelNtuple);

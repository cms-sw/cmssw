
// File: SiPixelTrackingRecHitsValid.hh
// // Authors:  Xingtao Huang (Puerto Rico Univ.)
//              Gavril Giurgiu (JHU)
// Creation Date:  Oct. 2006.
//
//--------------------------------------------

#ifndef Validation_RecoTrack_SiPixelTrackingRecHitsValid_h
#define Validation_RecoTrack_SiPixelTrackingRecHitsValid_h

//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
//#include "Validation/RecoTrack/interface/TrackLocalAngle.h"
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//--- for SimHit association
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"

#include <string>

class TTree;
class TFile;

class SiPixelTrackingRecHitsValid : public DQMEDAnalyzer {
public:
  explicit SiPixelTrackingRecHitsValid(const edm::ParameterSet& conf);

  ~SiPixelTrackingRecHitsValid() override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup& es) override;
  void beginJob() override;
  void endJob() override;

  //xt std::pair<LocalPoint,LocalVector> projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,const BoundPlane& plane);
  std::pair<LocalPoint, LocalVector> projectHit(const PSimHit& hit,
                                                const PixelGeomDetUnit* pixelDet,
                                                const BoundPlane& plane);

private:
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  //TrackLocalAngle *anglefinder_;
  DQMStore* dbe_;
  bool runStandalone;
  std::string outputFile_;
  std::string debugNtuple_;
  std::string builderName_;
  edm::EDGetTokenT<SiPixelRecHitCollection> siPixelRecHitCollectionToken_;
  edm::EDGetTokenT<reco::TrackCollection> recoTrackCollectionToken_;
  bool MTCCtrack_;

  bool checkType_;  // do we check that the simHit associated with recHit is of the expected particle type ?
  int genType_;     // the type of particle that the simHit associated with recHits should be

  // Pixel barrel detector has 3 layers and 8 modules; book histograms for each module = (layer, ring) pair

  MonitorElement* mePosxBarrelLayerModule[3][8];
  MonitorElement* mePosyBarrelLayerModule[3][8];
  MonitorElement* meErrxBarrelLayerModule[3][8];
  MonitorElement* meErryBarrelLayerModule[3][8];
  MonitorElement* meResxBarrelLayerModule[3][8];
  MonitorElement* meResyBarrelLayerModule[3][8];
  MonitorElement* mePullxBarrelLayerModule[3][8];
  MonitorElement* mePullyBarrelLayerModule[3][8];
  MonitorElement* meNpixBarrelLayerModule[3][8];
  MonitorElement* meNxpixBarrelLayerModule[3][8];
  MonitorElement* meNypixBarrelLayerModule[3][8];
  MonitorElement* meChargeBarrelLayerModule[3][8];
  MonitorElement* meResXvsAlphaBarrelLayerModule[3][8];
  MonitorElement* meResYvsAlphaBarrelLayerModule[3][8];
  MonitorElement* meResXvsBetaBarrelLayerModule[3][8];
  MonitorElement* meResYvsBetaBarrelLayerModule[3][8];
  MonitorElement* mePullXvsAlphaBarrelLayerModule[3][8];
  MonitorElement* mePullYvsAlphaBarrelLayerModule[3][8];
  MonitorElement* mePullXvsBetaBarrelLayerModule[3][8];
  MonitorElement* mePullYvsBetaBarrelLayerModule[3][8];
  MonitorElement* mePullXvsPhiBarrelLayerModule[3][8];
  MonitorElement* mePullYvsPhiBarrelLayerModule[3][8];
  MonitorElement* mePullXvsEtaBarrelLayerModule[3][8];
  MonitorElement* mePullYvsEtaBarrelLayerModule[3][8];

  // All layers and modules together
  MonitorElement* mePosxBarrel;
  MonitorElement* mePosyBarrel;
  MonitorElement* meErrxBarrel;
  MonitorElement* meErryBarrel;
  MonitorElement* meResxBarrel;
  MonitorElement* meResyBarrel;
  MonitorElement* mePullxBarrel;
  MonitorElement* mePullyBarrel;
  MonitorElement* meNpixBarrel;
  MonitorElement* meNxpixBarrel;
  MonitorElement* meNypixBarrel;
  MonitorElement* meChargeBarrel;
  MonitorElement* meResXvsAlphaBarrel;
  MonitorElement* meResYvsAlphaBarrel;
  MonitorElement* meResXvsBetaBarrel;
  MonitorElement* meResYvsBetaBarrel;
  MonitorElement* mePullXvsAlphaBarrel;
  MonitorElement* mePullYvsAlphaBarrel;
  MonitorElement* mePullXvsBetaBarrel;
  MonitorElement* mePullYvsBetaBarrel;
  MonitorElement* mePullXvsPhiBarrel;
  MonitorElement* mePullYvsPhiBarrel;
  MonitorElement* mePullXvsEtaBarrel;
  MonitorElement* mePullYvsEtaBarrel;

  MonitorElement* mePosxBarrelHalfModule;
  MonitorElement* mePosxBarrelFullModule;
  MonitorElement* mePosxBarrelFlippedLadders;
  MonitorElement* mePosxBarrelNonFlippedLadders;
  MonitorElement* mePosyBarrelHalfModule;
  MonitorElement* mePosyBarrelFullModule;
  MonitorElement* mePosyBarrelFlippedLadders;
  MonitorElement* mePosyBarrelNonFlippedLadders;

  MonitorElement* meResXvsAlphaBarrelFlippedLadders;
  MonitorElement* meResYvsAlphaBarrelFlippedLadders;
  MonitorElement* meResXvsBetaBarrelFlippedLadders;
  MonitorElement* meResYvsBetaBarrelFlippedLadders;
  MonitorElement* mePullXvsAlphaBarrelFlippedLadders;
  MonitorElement* mePullYvsAlphaBarrelFlippedLadders;
  MonitorElement* mePullXvsBetaBarrelFlippedLadders;
  MonitorElement* mePullYvsBetaBarrelFlippedLadders;
  MonitorElement* mePullXvsPhiBarrelFlippedLadders;
  MonitorElement* mePullYvsPhiBarrelFlippedLadders;
  MonitorElement* mePullXvsEtaBarrelFlippedLadders;
  MonitorElement* mePullYvsEtaBarrelFlippedLadders;

  MonitorElement* meResXvsAlphaBarrelNonFlippedLadders;
  MonitorElement* meResYvsAlphaBarrelNonFlippedLadders;
  MonitorElement* meResXvsBetaBarrelNonFlippedLadders;
  MonitorElement* meResYvsBetaBarrelNonFlippedLadders;
  MonitorElement* mePullXvsAlphaBarrelNonFlippedLadders;
  MonitorElement* mePullYvsAlphaBarrelNonFlippedLadders;
  MonitorElement* mePullXvsBetaBarrelNonFlippedLadders;
  MonitorElement* mePullYvsBetaBarrelNonFlippedLadders;
  MonitorElement* mePullXvsPhiBarrelNonFlippedLadders;
  MonitorElement* mePullYvsPhiBarrelNonFlippedLadders;
  MonitorElement* mePullXvsEtaBarrelNonFlippedLadders;
  MonitorElement* mePullYvsEtaBarrelNonFlippedLadders;

  MonitorElement* meWPullXvsAlphaBarrelFlippedLadders;
  MonitorElement* meWPullYvsAlphaBarrelFlippedLadders;
  MonitorElement* meWPullXvsBetaBarrelFlippedLadders;
  MonitorElement* meWPullYvsBetaBarrelFlippedLadders;
  MonitorElement* meWPullXvsAlphaBarrelNonFlippedLadders;
  MonitorElement* meWPullYvsAlphaBarrelNonFlippedLadders;
  MonitorElement* meWPullXvsBetaBarrelNonFlippedLadders;
  MonitorElement* meWPullYvsBetaBarrelNonFlippedLadders;

  // Split barrel x/y residuals and pulls in 3 layers
  MonitorElement* meResxBarrelLayer[3];
  MonitorElement* meResyBarrelLayer[3];
  MonitorElement* mePullxBarrelLayer[3];
  MonitorElement* mePullyBarrelLayer[3];

  MonitorElement* meResXvsAlphaBarrelFlippedLaddersLayer[3];
  MonitorElement* meResYvsAlphaBarrelFlippedLaddersLayer[3];
  MonitorElement* meResXvsBetaBarrelFlippedLaddersLayer[3];
  MonitorElement* meResYvsBetaBarrelFlippedLaddersLayer[3];
  MonitorElement* meResXvsAlphaBarrelNonFlippedLaddersLayer[3];
  MonitorElement* meResYvsAlphaBarrelNonFlippedLaddersLayer[3];
  MonitorElement* meResXvsBetaBarrelNonFlippedLaddersLayer[3];
  MonitorElement* meResYvsBetaBarrelNonFlippedLaddersLayer[3];

  // Pixel forward detector has 2 sides, 2 disks per side, 2 panels per disk and either 3 or 4 plaquettes per blade
  // Panel 1 has 4 plaquettes
  // Panel 2 has 3 plaquettes

  // Negative Z side
  // Panel 1: 2 disks, 4 plaquets

  MonitorElement* mePosxZmPanel1DiskPlaq[2][4];
  MonitorElement* mePosyZmPanel1DiskPlaq[2][4];
  MonitorElement* meErrxZmPanel1DiskPlaq[2][4];
  MonitorElement* meErryZmPanel1DiskPlaq[2][4];
  MonitorElement* meResxZmPanel1DiskPlaq[2][4];
  MonitorElement* meResyZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullxZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullyZmPanel1DiskPlaq[2][4];
  MonitorElement* meNpixZmPanel1DiskPlaq[2][4];
  MonitorElement* meNxpixZmPanel1DiskPlaq[2][4];
  MonitorElement* meNypixZmPanel1DiskPlaq[2][4];
  MonitorElement* meChargeZmPanel1DiskPlaq[2][4];
  MonitorElement* meResXvsAlphaZmPanel1DiskPlaq[2][4];
  MonitorElement* meResYvsAlphaZmPanel1DiskPlaq[2][4];
  MonitorElement* meResXvsBetaZmPanel1DiskPlaq[2][4];
  MonitorElement* meResYvsBetaZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsAlphaZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsAlphaZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsBetaZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsBetaZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsPhiZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsPhiZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsEtaZmPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsEtaZmPanel1DiskPlaq[2][4];

  // Panel 2: 2 disks, 3 plaquets
  MonitorElement* mePosxZmPanel2DiskPlaq[2][3];
  MonitorElement* mePosyZmPanel2DiskPlaq[2][3];
  MonitorElement* meErrxZmPanel2DiskPlaq[2][3];
  MonitorElement* meErryZmPanel2DiskPlaq[2][3];
  MonitorElement* meResxZmPanel2DiskPlaq[2][3];
  MonitorElement* meResyZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullxZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullyZmPanel2DiskPlaq[2][3];
  MonitorElement* meNpixZmPanel2DiskPlaq[2][3];
  MonitorElement* meNxpixZmPanel2DiskPlaq[2][3];
  MonitorElement* meNypixZmPanel2DiskPlaq[2][3];
  MonitorElement* meChargeZmPanel2DiskPlaq[2][3];
  MonitorElement* meResXvsAlphaZmPanel2DiskPlaq[2][3];
  MonitorElement* meResYvsAlphaZmPanel2DiskPlaq[2][3];
  MonitorElement* meResXvsBetaZmPanel2DiskPlaq[2][3];
  MonitorElement* meResYvsBetaZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsAlphaZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsAlphaZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsBetaZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsBetaZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsPhiZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsPhiZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsEtaZmPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsEtaZmPanel2DiskPlaq[2][3];

  // Positive Z side
  // Panel 1: 2 disks, 4 plaquets
  MonitorElement* mePosxZpPanel1DiskPlaq[2][4];
  MonitorElement* mePosyZpPanel1DiskPlaq[2][4];
  MonitorElement* meErrxZpPanel1DiskPlaq[2][4];
  MonitorElement* meErryZpPanel1DiskPlaq[2][4];
  MonitorElement* meResxZpPanel1DiskPlaq[2][4];
  MonitorElement* meResyZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullxZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullyZpPanel1DiskPlaq[2][4];
  MonitorElement* meNpixZpPanel1DiskPlaq[2][4];
  MonitorElement* meNxpixZpPanel1DiskPlaq[2][4];
  MonitorElement* meNypixZpPanel1DiskPlaq[2][4];
  MonitorElement* meChargeZpPanel1DiskPlaq[2][4];
  MonitorElement* meResXvsAlphaZpPanel1DiskPlaq[2][4];
  MonitorElement* meResYvsAlphaZpPanel1DiskPlaq[2][4];
  MonitorElement* meResXvsBetaZpPanel1DiskPlaq[2][4];
  MonitorElement* meResYvsBetaZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsAlphaZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsAlphaZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsBetaZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsBetaZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsPhiZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsPhiZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullXvsEtaZpPanel1DiskPlaq[2][4];
  MonitorElement* mePullYvsEtaZpPanel1DiskPlaq[2][4];

  // Panel 2: 2 disks, 3 plaquets
  MonitorElement* mePosxZpPanel2DiskPlaq[2][3];
  MonitorElement* mePosyZpPanel2DiskPlaq[2][3];
  MonitorElement* meErrxZpPanel2DiskPlaq[2][3];
  MonitorElement* meErryZpPanel2DiskPlaq[2][3];
  MonitorElement* meResxZpPanel2DiskPlaq[2][3];
  MonitorElement* meResyZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullxZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullyZpPanel2DiskPlaq[2][3];
  MonitorElement* meNpixZpPanel2DiskPlaq[2][3];
  MonitorElement* meNxpixZpPanel2DiskPlaq[2][3];
  MonitorElement* meNypixZpPanel2DiskPlaq[2][3];
  MonitorElement* meChargeZpPanel2DiskPlaq[2][3];
  MonitorElement* meResXvsAlphaZpPanel2DiskPlaq[2][3];
  MonitorElement* meResYvsAlphaZpPanel2DiskPlaq[2][3];
  MonitorElement* meResXvsBetaZpPanel2DiskPlaq[2][3];
  MonitorElement* meResYvsBetaZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsAlphaZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsAlphaZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsBetaZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsBetaZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsPhiZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsPhiZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullXvsEtaZpPanel2DiskPlaq[2][3];
  MonitorElement* mePullYvsEtaZpPanel2DiskPlaq[2][3];

  // All disks and plaquettes together

  // Negative Z side, panel 1
  MonitorElement* mePosxZmPanel1;
  MonitorElement* mePosyZmPanel1;
  MonitorElement* meErrxZmPanel1;
  MonitorElement* meErryZmPanel1;
  MonitorElement* meResxZmPanel1;
  MonitorElement* meResyZmPanel1;
  MonitorElement* mePullxZmPanel1;
  MonitorElement* mePullyZmPanel1;
  MonitorElement* meNpixZmPanel1;
  MonitorElement* meNxpixZmPanel1;
  MonitorElement* meNypixZmPanel1;
  MonitorElement* meChargeZmPanel1;
  MonitorElement* meResXvsAlphaZmPanel1;
  MonitorElement* meResYvsAlphaZmPanel1;
  MonitorElement* meResXvsBetaZmPanel1;
  MonitorElement* meResYvsBetaZmPanel1;
  MonitorElement* mePullXvsAlphaZmPanel1;
  MonitorElement* mePullYvsAlphaZmPanel1;
  MonitorElement* mePullXvsBetaZmPanel1;
  MonitorElement* mePullYvsBetaZmPanel1;
  MonitorElement* mePullXvsPhiZmPanel1;
  MonitorElement* mePullYvsPhiZmPanel1;
  MonitorElement* mePullXvsEtaZmPanel1;
  MonitorElement* mePullYvsEtaZmPanel1;

  MonitorElement* meWPullXvsAlphaZmPanel1;
  MonitorElement* meWPullYvsAlphaZmPanel1;
  MonitorElement* meWPullXvsBetaZmPanel1;
  MonitorElement* meWPullYvsBetaZmPanel1;

  // Negative Z side, panel 2
  MonitorElement* mePosxZmPanel2;
  MonitorElement* mePosyZmPanel2;
  MonitorElement* meErrxZmPanel2;
  MonitorElement* meErryZmPanel2;
  MonitorElement* meResxZmPanel2;
  MonitorElement* meResyZmPanel2;
  MonitorElement* mePullxZmPanel2;
  MonitorElement* mePullyZmPanel2;
  MonitorElement* meNpixZmPanel2;
  MonitorElement* meNxpixZmPanel2;
  MonitorElement* meNypixZmPanel2;
  MonitorElement* meChargeZmPanel2;
  MonitorElement* meResXvsAlphaZmPanel2;
  MonitorElement* meResYvsAlphaZmPanel2;
  MonitorElement* meResXvsBetaZmPanel2;
  MonitorElement* meResYvsBetaZmPanel2;
  MonitorElement* mePullXvsAlphaZmPanel2;
  MonitorElement* mePullYvsAlphaZmPanel2;
  MonitorElement* mePullXvsBetaZmPanel2;
  MonitorElement* mePullYvsBetaZmPanel2;
  MonitorElement* mePullXvsPhiZmPanel2;
  MonitorElement* mePullYvsPhiZmPanel2;
  MonitorElement* mePullXvsEtaZmPanel2;
  MonitorElement* mePullYvsEtaZmPanel2;

  MonitorElement* meWPullXvsAlphaZmPanel2;
  MonitorElement* meWPullYvsAlphaZmPanel2;
  MonitorElement* meWPullXvsBetaZmPanel2;
  MonitorElement* meWPullYvsBetaZmPanel2;

  // Positive Z side, panel 1
  MonitorElement* mePosxZpPanel1;
  MonitorElement* mePosyZpPanel1;
  MonitorElement* meErrxZpPanel1;
  MonitorElement* meErryZpPanel1;
  MonitorElement* meResxZpPanel1;
  MonitorElement* meResyZpPanel1;
  MonitorElement* mePullxZpPanel1;
  MonitorElement* mePullyZpPanel1;
  MonitorElement* meNpixZpPanel1;
  MonitorElement* meNxpixZpPanel1;
  MonitorElement* meNypixZpPanel1;
  MonitorElement* meChargeZpPanel1;
  MonitorElement* meResXvsAlphaZpPanel1;
  MonitorElement* meResYvsAlphaZpPanel1;
  MonitorElement* meResXvsBetaZpPanel1;
  MonitorElement* meResYvsBetaZpPanel1;
  MonitorElement* mePullXvsAlphaZpPanel1;
  MonitorElement* mePullYvsAlphaZpPanel1;
  MonitorElement* mePullXvsBetaZpPanel1;
  MonitorElement* mePullYvsBetaZpPanel1;
  MonitorElement* mePullXvsPhiZpPanel1;
  MonitorElement* mePullYvsPhiZpPanel1;
  MonitorElement* mePullXvsEtaZpPanel1;
  MonitorElement* mePullYvsEtaZpPanel1;

  MonitorElement* meWPullXvsAlphaZpPanel1;
  MonitorElement* meWPullYvsAlphaZpPanel1;
  MonitorElement* meWPullXvsBetaZpPanel1;
  MonitorElement* meWPullYvsBetaZpPanel1;

  // Positive Z side, panel 2
  MonitorElement* mePosxZpPanel2;
  MonitorElement* mePosyZpPanel2;
  MonitorElement* meErrxZpPanel2;
  MonitorElement* meErryZpPanel2;
  MonitorElement* meResxZpPanel2;
  MonitorElement* meResyZpPanel2;
  MonitorElement* mePullxZpPanel2;
  MonitorElement* mePullyZpPanel2;
  MonitorElement* meNpixZpPanel2;
  MonitorElement* meNxpixZpPanel2;
  MonitorElement* meNypixZpPanel2;
  MonitorElement* meChargeZpPanel2;
  MonitorElement* meResXvsAlphaZpPanel2;
  MonitorElement* meResYvsAlphaZpPanel2;
  MonitorElement* meResXvsBetaZpPanel2;
  MonitorElement* meResYvsBetaZpPanel2;
  MonitorElement* mePullXvsAlphaZpPanel2;
  MonitorElement* mePullYvsAlphaZpPanel2;
  MonitorElement* mePullXvsBetaZpPanel2;
  MonitorElement* mePullYvsBetaZpPanel2;
  MonitorElement* mePullXvsPhiZpPanel2;
  MonitorElement* mePullYvsPhiZpPanel2;
  MonitorElement* mePullXvsEtaZpPanel2;
  MonitorElement* mePullYvsEtaZpPanel2;

  MonitorElement* meWPullXvsAlphaZpPanel2;
  MonitorElement* meWPullYvsAlphaZpPanel2;
  MonitorElement* meWPullXvsBetaZpPanel2;
  MonitorElement* meWPullYvsBetaZpPanel2;

  // all hits (not only from tracks)
  MonitorElement* mePosxBarrel_all_hits;
  MonitorElement* mePosyBarrel_all_hits;

  MonitorElement* mePosxZmPanel1_all_hits;
  MonitorElement* mePosyZmPanel1_all_hits;
  MonitorElement* mePosxZmPanel2_all_hits;
  MonitorElement* mePosyZmPanel2_all_hits;

  MonitorElement* mePosxZpPanel1_all_hits;
  MonitorElement* mePosyZpPanel1_all_hits;
  MonitorElement* mePosxZpPanel2_all_hits;
  MonitorElement* mePosyZpPanel2_all_hits;

  // control histograms
  MonitorElement* meTracksPerEvent;
  MonitorElement* mePixRecHitsPerTrack;

  // variables that go in the output tree
  float rechitx;     // x position of hit
  float rechity;     // y position of hit
  float rechitz;     // z position of hit
  float rechiterrx;  // x position error of hit (error not squared)
  float rechiterry;  // y position error of hit (error not squared)

  float rechitresx;   // difference between reconstructed hit x position and 'true' x position
  float rechitresy;   // difference between reconstructed hit y position and 'true' y position
  float rechitpullx;  // x residual divideded by error
  float rechitpully;  // y residual divideded by error

  int npix;      // number of pixel in the cluster
  int nxpix;     // size of cluster (number of pixels) along x direction
  int nypix;     // size of cluster (number of pixels) along y direction
  float charge;  // total charge in cluster

  float alpha;  // track angle in the xz plane of the module local coordinate system
  float beta;   // track angle in the yz plane of the module local coordinate system

  float phi;  // polar track angle
  float eta;  // pseudo-rapidity (function of theta, the azimuthal angle)

  int subdetId;
  int layer;
  int ladder;
  int mod;
  int side;
  int disk;
  int blade;
  int panel;
  int plaq;
  int half;     // half = 1 if the barrel module is half size and 0 if it is full size (only defined for barrel)
  int flipped;  // flipped = 1 if the module is flipped and 0 if non-flipped (only defined for barrel)

  int nsimhit;  // number of simhits associated with a rechit
  int pidhit;   // PID of the particle that produced the simHit associated with the recHit

  float simhitx;  // true x position of hit
  float simhity;  // true y position of hit

  edm::EventNumber_t evt;
  edm::RunNumber_t run;

  TFile* tfile_;
  TTree* t_;
};

#endif

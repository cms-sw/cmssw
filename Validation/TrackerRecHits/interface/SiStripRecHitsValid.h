#ifndef SiStripRecHitsValid_h
#define SiStripRecHitsValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
//only mine
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

//--- for SimHit association
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include <string>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
//For RecHit
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class SiStripDetCabling;
class SiStripDCSStatus;

class SiStripRecHitsValid : public DQMEDAnalyzer {
public:
  SiStripRecHitsValid(const edm::ParameterSet& conf);

  ~SiStripRecHitsValid() override;

  struct TotalMEs {  // MEs for total detector Level
    MonitorElement* meNumTotrphi;
    MonitorElement* meNumTotStereo;
    MonitorElement* meNumTotMatched;
  };

  struct SubDetMEs {  // MEs for Subdetector Level
    MonitorElement* meNumrphi;
    MonitorElement* meBunchrphi;
    MonitorElement* meEventrphi;
    MonitorElement* meNumStereo;
    MonitorElement* meBunchStereo;
    MonitorElement* meEventStereo;
    MonitorElement* meNumMatched;
    MonitorElement* meBunchMatched;
    MonitorElement* meEventMatched;
  };

  struct LayerMEs {  // MEs for Layer Level
    MonitorElement* meWclusrphi;
    MonitorElement* meAdcrphi;
    MonitorElement* mePosxrphi;
    MonitorElement* meResolxrphi;
    MonitorElement* meResrphi;
    MonitorElement* mePullLFrphi;
    MonitorElement* mePullMFrphi;
    MonitorElement* meChi2rphi;
    MonitorElement* meNsimHitrphi;
  };

  struct StereoAndMatchedMEs {  // MEs for Layer Level
    MonitorElement* meWclusStereo;
    MonitorElement* meAdcStereo;
    MonitorElement* mePosxStereo;
    MonitorElement* meResolxStereo;
    MonitorElement* meResStereo;
    MonitorElement* mePullLFStereo;
    MonitorElement* mePullMFStereo;
    MonitorElement* meChi2Stereo;
    MonitorElement* meNsimHitStereo;
    MonitorElement* mePosxMatched;
    MonitorElement* mePosyMatched;
    MonitorElement* meResolxMatched;
    MonitorElement* meResolyMatched;
    MonitorElement* meResxMatched;
    MonitorElement* meResyMatched;
    MonitorElement* meChi2Matched;
    MonitorElement* meNsimHitMatched;
  };

  struct RecHitProperties {
    float x;
    float y;
    //    float z;
    float resolxx;
    //    float resolxy;
    float resolyy;
    float resx;
    float resy;
    float pullMF;
    int clusiz;
    float cluchg;
    float chi2;
    int NsimHit;
    int bunch;
    int event;
  };

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup& es) override;

private:
  TotalMEs totalMEs;

  bool switchNumTotrphi;
  bool switchNumTotStereo;
  bool switchNumTotMatched;

  bool switchNumrphi;
  bool switchBunchrphi;
  bool switchEventrphi;
  bool switchNumStereo;
  bool switchBunchStereo;
  bool switchEventStereo;
  bool switchNumMatched;
  bool switchBunchMatched;
  bool switchEventMatched;

  bool switchWclusrphi;
  bool switchAdcrphi;
  bool switchPosxrphi;
  bool switchResolxrphi;
  bool switchResrphi;
  bool switchPullLFrphi;
  bool switchPullMFrphi;
  bool switchChi2rphi;
  bool switchNsimHitrphi;
  bool switchWclusStereo;
  bool switchAdcStereo;
  bool switchPosxStereo;
  bool switchResolxStereo;
  bool switchResStereo;
  bool switchPullLFStereo;
  bool switchPullMFStereo;
  bool switchChi2Stereo;
  bool switchNsimHitStereo;
  bool switchPosxMatched;
  bool switchPosyMatched;
  bool switchResolxMatched;
  bool switchResolyMatched;
  bool switchResxMatched;
  bool switchResyMatched;
  bool switchChi2Matched;
  bool switchNsimHitMatched;

  std::string topFolderName_;
  std::vector<std::string> SubDetList_;

  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, StereoAndMatchedMEs> StereoAndMatchedMEsMap;
  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, std::vector<uint32_t> > LayerDetMap;
  std::map<std::string, std::vector<uint32_t> > StereoAndMatchedDetMap;

  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;

  std::pair<LocalPoint, LocalVector> projectHit(const PSimHit& hit,
                                                const StripGeomDetUnit* stripDet,
                                                const BoundPlane& plane);
  void createMEs(DQMStore::IBooker& ibooker, const edm::EventSetup& es);
  void createTotalMEs(DQMStore::IBooker& ibooker);
  void createLayerMEs(DQMStore::IBooker& ibooker, std::string label);
  void createSubDetMEs(DQMStore::IBooker& ibooker, std::string label);
  void createStereoAndMatchedMEs(DQMStore::IBooker& ibooker, std::string label);

  MonitorElement* bookME1D(DQMStore::IBooker& ibooker,
                           const char* ParameterSetLabel,
                           const char* HistoName,
                           const char* HistoTitle);

  inline void fillME(MonitorElement* ME, float value1) {
    if (ME != nullptr)
      ME->Fill(value1);
  }
  inline void fillME(MonitorElement* ME, float value1, float value2) {
    if (ME != nullptr)
      ME->Fill(value1, value2);
  }
  inline void fillME(MonitorElement* ME, float value1, float value2, float value3) {
    if (ME != nullptr)
      ME->Fill(value1, value2, value3);
  }
  inline void fillME(MonitorElement* ME, float value1, float value2, float value3, float value4) {
    if (ME != nullptr)
      ME->Fill(value1, value2, value3, value4);
  }

  edm::ParameterSet conf_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  unsigned long long m_cacheID_;
  //const StripTopology* topol;

  /* static const int MAXHIT = 1000; */

  //  std::vector<RecHitProperties> rechitrphi;
  //  std::vector<RecHitProperties> rechitstereo;
  //  std::vector<RecHitProperties> rechitmatched;
  RecHitProperties rechitpro;

  void rechitanalysis(SiStripRecHit2D const rechit, const StripTopology& topol, TrackerHitAssociator& associate);
  void rechitanalysis_matched(SiStripMatchedRecHit2D const rechit,
                              const GluedGeomDet* gluedDet,
                              TrackerHitAssociator& associate);

  edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> matchedRecHitsToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> rphiRecHitsToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stereoRecHitsToken_;
};

#endif

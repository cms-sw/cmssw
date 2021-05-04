// -*- C++ -*-
//
// Package:    TBeamTest
// Class:      TBeamTest
//
/**\class TBeamTest TBeamTest.cc 

 Description: Access Digi collection and fill a few histograms to compare with TestBeam data

*/
//
// Author:  Suchandra Dutta, Suvankar RoyChoudhury
// Created:  July 2015
//
//
// system include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM Histograming
#include "DQMServices/Core/interface/DQMStore.h"
#include <cmath>
class TBeamTest : public DQMEDAnalyzer {
public:
  explicit TBeamTest(const edm::ParameterSet&);
  ~TBeamTest() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  struct DigiMEs {
    MonitorElement* NumberOfDigis;
    MonitorElement* PositionOfDigis;
    MonitorElement* NumberOfClusters;
    std::vector<MonitorElement*> ClusterWidths;
    MonitorElement* ClusterPosition;
  };

private:
  int matchedSimTrackIndex(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& linkHandle,
                           edm::Handle<edm::SimTrackContainer>& simTkHandle,
                           DetId detId,
                           unsigned int& channel);

  void fillClusterWidth(DigiMEs& mes, float dphi, float width);
  edm::ParameterSet config_;
  std::map<std::string, DigiMEs> detMEs;
  edm::InputTag otDigiSrc_;
  edm::InputTag digiSimLinkSrc_;
  edm::InputTag simTrackSrc_;
  std::string geomType_;

  std::vector<double> phiValues;
  const edm::EDGetTokenT<edm::DetSetVector<Phase2TrackerDigi> > otDigiToken_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > otDigiSimLinkToken_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
};
//
// constructors
//
TBeamTest::TBeamTest(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      otDigiSrc_(iConfig.getParameter<edm::InputTag>("OuterTrackerDigiSource")),
      digiSimLinkSrc_(iConfig.getParameter<edm::InputTag>("OuterTrackerDigiSimSource")),
      simTrackSrc_(iConfig.getParameter<edm::InputTag>("SimTrackSource")),
      geomType_(iConfig.getParameter<std::string>("GeometryType")),
      phiValues(iConfig.getParameter<std::vector<double> >("PhiAngles")),
      otDigiToken_(consumes<edm::DetSetVector<Phase2TrackerDigi> >(otDigiSrc_)),
      otDigiSimLinkToken_(consumes<edm::DetSetVector<PixelDigiSimLink> >(digiSimLinkSrc_)),
      simTrackToken_(consumes<edm::SimTrackContainer>(simTrackSrc_)) {
  edm::LogInfo("TBeamTest") << ">>> Construct TBeamTest ";
}

//
// destructor
//
TBeamTest::~TBeamTest() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("TBeamTest") << ">>> Destroy TBeamTest ";
}
//
// -- Analyze
//
void TBeamTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get digis

  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > digiSimLinkHandle;
  iEvent.getByToken(otDigiSimLinkToken_, digiSimLinkHandle);

  edm::Handle<edm::DetSetVector<Phase2TrackerDigi> > otDigiHandle;
  iEvent.getByToken(otDigiToken_, otDigiHandle);
  const DetSetVector<Phase2TrackerDigi>* digis = otDigiHandle.product();

  // Get SimTrack
  edm::Handle<edm::SimTrackContainer> simTrackHandle;
  iEvent.getByToken(simTrackToken_, simTrackHandle);

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* tTopo = tTopoHandle.product();

  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geomHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geomHandle);

    const TrackerGeometry* tkGeom = geomHandle.product();

    edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSViter;
    std::string moduleType;
    for (DSViter = digis->begin(); DSViter != digis->end(); DSViter++) {
      unsigned int rawid = DSViter->id;
      DetId detId(rawid);
      if (detId.det() != DetId::Detector::Tracker)
        continue;

      if (detId.subdetId() != StripSubdetector::TOB)
        continue;
      int layer = tTopo->getOTLayerNumber(rawid);
      if (layer != 4)
        continue;

      const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);

      const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(geomDetUnit);
      int nColumns = tkDetUnit->specificTopology().ncolumns();

      edm::LogInfo("TBeamTest") << " Det Id = " << rawid;

      if (layer <= 3) {
        if (nColumns > 2)
          moduleType = "PSP_Modules";
        else
          moduleType = "PSS_Modules";
      } else
        moduleType = "2S_Modules";

      std::map<std::string, DigiMEs>::iterator pos = detMEs.find(moduleType);
      if (pos != detMEs.end()) {
        DigiMEs local_mes = pos->second;
        int nDigi = 0;
        int row_last = -1;
        int col_last = -1;
        int nclus = 0;
        int width = 1;
        int position = 0;
        float dPhi = 9999.9;
        for (DetSet<Phase2TrackerDigi>::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
          int col = di->column();  // column
          int row = di->row();     // row
          MeasurementPoint mp(row + 0.5, col + 0.5);
          unsigned int channel = Phase2TrackerDigi::pixelToChannel(row, col);
          int tkIndx = matchedSimTrackIndex(digiSimLinkHandle, simTrackHandle, detId, channel);

          if (geomDetUnit && tkIndx != -1)
            dPhi = reco::deltaPhi((*simTrackHandle)[tkIndx].momentum().phi(), geomDetUnit->position().phi());

          nDigi++;
          edm::LogInfo("TBeamTest") << "  column " << col << " row " << row << std::endl;
          local_mes.PositionOfDigis->Fill(row + 1);

          if (row_last == -1) {
            width = 1;
            position = row + 1;
            nclus++;
          } else {
            if (abs(row - row_last) == 1 && col == col_last) {
              position += row + 1;
              width++;
            } else {
              position /= width;
              fillClusterWidth(local_mes, dPhi, width);
              local_mes.ClusterPosition->Fill(position);
              width = 1;
              position = row + 1;
              nclus++;
            }
          }
          edm::LogInfo("TBeamTest") << " row " << row << " col " << col << " row_last " << row_last << " col_last "
                                    << col_last << " width " << width;
          row_last = row;
          col_last = col;
        }
        position /= width;
        fillClusterWidth(local_mes, dPhi, width);
        local_mes.ClusterPosition->Fill(position);
        local_mes.NumberOfClusters->Fill(nclus);
        local_mes.NumberOfDigis->Fill(nDigi);
      }
    }
  }
}
//
// -- Book Histograms
//
void TBeamTest::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");

  std::vector<std::string> types;
  types.push_back("2S_Modules");
  types.push_back("PSP_Modules");
  types.push_back("PSS_Modules");
  ibooker.cd();

  for (const auto& itype : types) {
    std::stringstream folder_name;

    folder_name << top_folder << "/" << itype;

    edm::LogInfo("TBeamTest") << " Booking Histograms in : " << folder_name.str();
    ibooker.setCurrentFolder(folder_name.str());

    std::ostringstream HistoName;

    DigiMEs local_mes;
    edm::ParameterSet Parameters = config_.getParameter<edm::ParameterSet>("NumberOfDigisH");
    HistoName.str("");
    HistoName << "numberOfHits";
    local_mes.NumberOfDigis = ibooker.book1D(HistoName.str(),
                                             HistoName.str(),
                                             Parameters.getParameter<int32_t>("Nbins"),
                                             Parameters.getParameter<double>("xmin"),
                                             Parameters.getParameter<double>("xmax"));

    Parameters = config_.getParameter<edm::ParameterSet>("PositionOfDigisH");
    HistoName.str("");
    HistoName << "hitPositions";
    local_mes.PositionOfDigis = ibooker.book1D(HistoName.str(),
                                               HistoName.str(),
                                               Parameters.getParameter<int32_t>("Nxbins"),
                                               Parameters.getParameter<double>("xmin"),
                                               Parameters.getParameter<double>("xmax"));
    Parameters = config_.getParameter<edm::ParameterSet>("NumberOfClustersH");
    HistoName.str("");
    HistoName << "numberOfCluetsrs";
    local_mes.NumberOfClusters = ibooker.book1D(HistoName.str(),
                                                HistoName.str(),
                                                Parameters.getParameter<int32_t>("Nbins"),
                                                Parameters.getParameter<double>("xmin"),
                                                Parameters.getParameter<double>("xmax"));
    Parameters = config_.getParameter<edm::ParameterSet>("ClusterWidthH");

    for (unsigned int i = 0; i < phiValues.size(); i++) {
      HistoName.str("");
      HistoName << "clusterWidth_";
      HistoName << i;

      local_mes.ClusterWidths.push_back(ibooker.book1D(HistoName.str(),
                                                       HistoName.str(),
                                                       Parameters.getParameter<int32_t>("Nbins"),
                                                       Parameters.getParameter<double>("xmin"),
                                                       Parameters.getParameter<double>("xmax")));
    }
    Parameters = config_.getParameter<edm::ParameterSet>("ClusterPositionH");
    HistoName.str("");
    HistoName << "clusterPositions";
    local_mes.ClusterPosition = ibooker.book1D(HistoName.str(),
                                               HistoName.str(),
                                               Parameters.getParameter<int32_t>("Nbins"),
                                               Parameters.getParameter<double>("xmin"),
                                               Parameters.getParameter<double>("xmax"));
    detMEs.insert(std::make_pair(itype, local_mes));
  }
}
int TBeamTest::matchedSimTrackIndex(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& linkHandle,
                                    edm::Handle<edm::SimTrackContainer>& simTkHandle,
                                    DetId detId,
                                    unsigned int& channel) {
  int simTrkIndx = -1;
  unsigned int simTrkId = 0;
  edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = linkHandle->find(detId);

  if (isearch == linkHandle->end())
    return simTrkIndx;

  edm::DetSet<PixelDigiSimLink> link_detset = (*linkHandle)[detId];
  // Loop over DigiSimLink in this det unit
  for (edm::DetSet<PixelDigiSimLink>::const_iterator it = link_detset.data.begin(); it != link_detset.data.end();
       it++) {
    if (channel == it->channel()) {
      simTrkId = it->SimTrackId();
      break;
    }
  }
  if (simTrkId == 0)
    return simTrkIndx;
  edm::SimTrackContainer sim_tracks = (*simTkHandle.product());
  for (unsigned int itk = 0; itk < sim_tracks.size(); itk++) {
    if (sim_tracks[itk].trackId() == simTrkId) {
      simTrkIndx = itk;
      break;
    }
  }
  return simTrkIndx;
}
void TBeamTest::fillClusterWidth(DigiMEs& mes, float dphi, float width) {
  for (unsigned int i = 0; i < phiValues.size(); i++) {
    float angle_min = (phiValues[i] - 0.1) * std::acos(-1.0) / 180.0;
    float angle_max = (phiValues[i] + 0.1) * std::acos(-1.0) / 180.0;
    if (std::fabs(dphi) > angle_min && std::fabs(dphi) < angle_max) {
      mes.ClusterWidths[i]->Fill(width);
      break;
    }
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(TBeamTest);

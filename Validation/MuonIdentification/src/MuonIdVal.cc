#include "Validation/MuonIdentification/interface/MuonIdVal.h"

MuonIdVal::MuonIdVal(const edm::ParameterSet &ps)
    : trackingGeomToken_(esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()) {
  iConfig = ps;
  inputMuonCollection_ = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
  inputDTRecSegment4DCollection_ = iConfig.getParameter<edm::InputTag>("inputDTRecSegment4DCollection");
  inputCSCSegmentCollection_ = iConfig.getParameter<edm::InputTag>("inputCSCSegmentCollection");
  inputMuonTimeExtraValueMap_ = iConfig.getParameter<edm::InputTag>("inputMuonTimeExtraValueMap");
  inputMuonCosmicCompatibilityValueMap_ = iConfig.getParameter<edm::InputTag>("inputMuonCosmicCompatibilityValueMap");
  inputMuonShowerInformationValueMap_ = iConfig.getParameter<edm::InputTag>("inputMuonShowerInformationValueMap");
  useTrackerMuons_ = iConfig.getUntrackedParameter<bool>("useTrackerMuons");
  useGlobalMuons_ = iConfig.getUntrackedParameter<bool>("useGlobalMuons");
  useTrackerMuonsNotGlobalMuons_ = iConfig.getUntrackedParameter<bool>("useTrackerMuonsNotGlobalMuons");
  useGlobalMuonsNotTrackerMuons_ = iConfig.getUntrackedParameter<bool>("useGlobalMuonsNotTrackerMuons");
  makeEnergyPlots_ = iConfig.getUntrackedParameter<bool>("makeEnergyPlots");
  makeTimePlots_ = iConfig.getUntrackedParameter<bool>("makeTimePlots");
  make2DPlots_ = iConfig.getUntrackedParameter<bool>("make2DPlots");
  makeAllChamberPlots_ = iConfig.getUntrackedParameter<bool>("makeAllChamberPlots");
  makeCosmicCompatibilityPlots_ = iConfig.getUntrackedParameter<bool>("makeCosmicCompatibilityPlots");
  makeShowerInformationPlots_ = iConfig.getUntrackedParameter<bool>("makeShowerInformationPlots");
  baseFolder_ = iConfig.getUntrackedParameter<std::string>("baseFolder");

  inputMuonCollectionToken_ = consumes<reco::MuonCollection>(inputMuonCollection_);
  inputDTRecSegment4DCollectionToken_ = consumes<DTRecSegment4DCollection>(inputDTRecSegment4DCollection_);
  inputCSCSegmentCollectionToken_ = consumes<CSCSegmentCollection>(inputCSCSegmentCollection_);
  inputMuonTimeExtraValueMapCombToken_ =
      consumes<reco::MuonTimeExtraMap>(edm::InputTag(inputMuonTimeExtraValueMap_.label(), "combined"));
  inputMuonTimeExtraValueMapDTToken_ =
      consumes<reco::MuonTimeExtraMap>(edm::InputTag(inputMuonTimeExtraValueMap_.label(), "csc"));
  inputMuonTimeExtraValueMapCSCToken_ =
      consumes<reco::MuonTimeExtraMap>(edm::InputTag(inputMuonTimeExtraValueMap_.label(), "dt"));
  inputMuonCosmicCompatibilityValueMapToken_ =
      consumes<edm::ValueMap<reco::MuonCosmicCompatibility>>(inputMuonCosmicCompatibilityValueMap_);
  inputMuonShowerInformationValueMapToken_ =
      consumes<edm::ValueMap<reco::MuonShower>>(inputMuonShowerInformationValueMap_);

  subsystemname_ = iConfig.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem");
}

MuonIdVal::~MuonIdVal() {}

void MuonIdVal::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &) {
  char name[100], title[200];
  ibooker.setCurrentFolder(baseFolder_);

  // trackerMuon == 0; globalMuon == 1; trackerMuon && !globalMuon == 2;
  // globalMuon && !trackerMuon == 3
  for (unsigned int i = 0; i < 4; i++) {
    if ((i == 0 && !useTrackerMuons_) || (i == 1 && !useGlobalMuons_))
      continue;
    if ((i == 2 && !useTrackerMuonsNotGlobalMuons_) || (i == 3 && !useGlobalMuonsNotTrackerMuons_))
      continue;
    if (i == 0)
      ibooker.setCurrentFolder(baseFolder_ + "/TrackerMuons");
    if (i == 1)
      ibooker.setCurrentFolder(baseFolder_ + "/GlobalMuons");
    if (i == 2)
      ibooker.setCurrentFolder(baseFolder_ + "/TrackerMuonsNotGlobalMuons");
    if (i == 3)
      ibooker.setCurrentFolder(baseFolder_ + "/GlobalMuonsNotTrackerMuons");

    if (makeEnergyPlots_) {
      hEnergyEMBarrel[i] = ibooker.book1D("hEnergyEMBarrel", "Energy in ECAL Barrel", 100, -0.5, 2.);
      hEnergyHABarrel[i] = ibooker.book1D("hEnergyHABarrel", "Energy in HCAL Barrel", 100, -4., 12.);
      hEnergyHO[i] = ibooker.book1D("hEnergyHO", "Energy HO", 100, -2., 5.);
      hEnergyEMEndcap[i] = ibooker.book1D("hEnergyEMEndcap", "Energy in ECAL Endcap", 100, -0.5, 2.);
      hEnergyHAEndcap[i] = ibooker.book1D("hEnergyHAEndcap", "Energy in HCAL Endcap", 100, -4., 12.);
    }

    if (makeTimePlots_) {
      hMuonTimeNDOF[i] = ibooker.book1D("hMuonTimeNDOF", "MuonTime NDOF", 52, -1.5, 50.5);
      hMuonTimeTimeAtIpInOut[i] = ibooker.book1D("hMuonTimeTimeAtIpInOut", "MuonTime TimeAtIpInOut", 100, -20., 20.);
      hMuonTimeTimeAtIpInOutErr[i] =
          ibooker.book1D("hMuonTimeTimeAtIpInOutErr", "MuonTime TimeAtIpInOutErr", 100, 0., 8.);
      hMuonTimeTimeAtIpOutIn[i] = ibooker.book1D("hMuonTimeTimeAtIpOutIn", "MuonTime TimeAtIpOutIn", 100, -1., 75.);
      hMuonTimeTimeAtIpOutInErr[i] =
          ibooker.book1D("hMuonTimeTimeAtIpOutInErr", "MuonTime TimeAtIpOutInErr", 100, 0., 8.);
      hMuonTimeExtraCombinedNDOF[i] =
          ibooker.book1D("hMuonTimeExtraCombinedNDOF", "MuonTimeExtra Combined NDOF", 52, -1.5, 50.5);
      hMuonTimeExtraCombinedTimeAtIpInOut[i] =
          ibooker.book1D("hMuonTimeExtraCombinedTimeAtIpInOut", "MuonTimeExtra Combined TimeAtIpInOut", 100, -20., 20.);
      hMuonTimeExtraCombinedTimeAtIpInOutErr[i] = ibooker.book1D(
          "hMuonTimeExtraCombinedTimeAtIpInOutErr", "MuonTimeExtra Combined TimeAtIpInOutErr", 100, 0., 8.);
      hMuonTimeExtraCombinedTimeAtIpOutIn[i] =
          ibooker.book1D("hMuonTimeExtraCombinedTimeAtIpOutIn", "MuonTimeExtra Combined TimeAtIpOutIn", 100, -1., 75.);
      hMuonTimeExtraCombinedTimeAtIpOutInErr[i] = ibooker.book1D(
          "hMuonTimeExtraCombinedTimeAtIpOutInErr", "MuonTimeExtra Combined TimeAtIpOutInErr", 100, 0., 8.);
      hMuonTimeExtraCSCNDOF[i] = ibooker.book1D("hMuonTimeExtraCSCNDOF", "MuonTimeExtra CSC NDOF", 52, -1.5, 50.5);
      hMuonTimeExtraCSCTimeAtIpInOut[i] =
          ibooker.book1D("hMuonTimeExtraCSCTimeAtIpInOut", "MuonTimeExtra CSC TimeAtIpInOut", 100, -20., 20.);
      hMuonTimeExtraCSCTimeAtIpInOutErr[i] =
          ibooker.book1D("hMuonTimeExtraCSCTimeAtIpInOutErr", "MuonTimeExtra CSC TimeAtIpInOutErr", 100, 0., 8.);
      hMuonTimeExtraCSCTimeAtIpOutIn[i] =
          ibooker.book1D("hMuonTimeExtraCSCTimeAtIpOutIn", "MuonTimeExtra CSC TimeAtIpOutIn", 100, -1., 75.);
      hMuonTimeExtraCSCTimeAtIpOutInErr[i] =
          ibooker.book1D("hMuonTimeExtraCSCTimeAtIpOutInErr", "MuonTimeExtra CSC TimeAtIpOutInErr", 100, 0., 8.);
      hMuonTimeExtraDTNDOF[i] = ibooker.book1D("hMuonTimeExtraDTNDOF", "MuonTimeExtra DT NDOF", 52, -1.5, 50.5);
      hMuonTimeExtraDTTimeAtIpInOut[i] =
          ibooker.book1D("hMuonTimeExtraDTTimeAtIpInOut", "MuonTimeExtra DT TimeAtIpInOut", 100, -20., 20.);
      hMuonTimeExtraDTTimeAtIpInOutErr[i] =
          ibooker.book1D("hMuonTimeExtraDTTimeAtIpInOutErr", "MuonTimeExtra DT TimeAtIpInOutErr", 100, 0., 8.);
      hMuonTimeExtraDTTimeAtIpOutIn[i] =
          ibooker.book1D("hMuonTimeExtraDTTimeAtIpOutIn", "MuonTimeExtra DT TimeAtIpOutIn", 100, -1., 75.);
      hMuonTimeExtraDTTimeAtIpOutInErr[i] =
          ibooker.book1D("hMuonTimeExtraDTTimeAtIpOutInErr", "MuonTimeExtra DT TimeAtIpOutInErr", 100, 0., 8.);
    }

    hCaloCompat[i] = ibooker.book1D("hCaloCompat", "Calo Compatibility", 101, -0.05, 1.05);
    hSegmentCompat[i] = ibooker.book1D("hSegmentCompat", "Segment Compatibility", 101, -0.05, 1.05);
    if (make2DPlots_)
      hCaloSegmentCompat[i] = ibooker.book2D(
          "hCaloSegmentCompat", "Calo Compatibility vs. Segment Compatibility", 101, -0.05, 1.05, 101, -0.05, 1.05);
    hMuonQualityTrkRelChi2[i] = ibooker.book1D("hMuonQualityTrkRelChi2", "MuonQuality TrkRelChi2", 100, 0., 1.5);
    hMuonQualityStaRelChi2[i] = ibooker.book1D("hMuonQualityStaRelChi2", "MuonQuality StaRelChi2", 100, 0., 3.);
    hMuonQualityTrkKink[i] = ibooker.book1D("hMuonQualityTrkKink", "MuonQuality TrkKink", 100, 0., 150.);
    hGlobalMuonPromptTightBool[i] =
        ibooker.book1D("hGlobalMuonPromptTightBool", "GlobalMuonPromptTight Boolean", 2, -0.5, 1.5);
    hTMLastStationLooseBool[i] = ibooker.book1D("hTMLastStationLooseBool", "TMLastStationLoose Boolean", 2, -0.5, 1.5);
    hTMLastStationTightBool[i] = ibooker.book1D("hTMLastStationTightBool", "TMLastStationTight Boolean", 2, -0.5, 1.5);
    hTM2DCompatibilityLooseBool[i] =
        ibooker.book1D("hTM2DCompatibilityLooseBool", "TM2DCompatibilityLoose Boolean", 2, -0.5, 1.5);
    hTM2DCompatibilityTightBool[i] =
        ibooker.book1D("hTM2DCompatibilityTightBool", "TM2DCompatibilityTight Boolean", 2, -0.5, 1.5);
    hTMOneStationLooseBool[i] = ibooker.book1D("hTMOneStationLooseBool", "TMOneStationLoose Boolean", 2, -0.5, 1.5);
    hTMOneStationTightBool[i] = ibooker.book1D("hTMOneStationTightBool", "TMOneStationTight Boolean", 2, -0.5, 1.5);
    hTMLastStationOptimizedLowPtLooseBool[i] = ibooker.book1D(
        "hTMLastStationOptimizedLowPtLooseBool", "TMLastStationOptimizedLowPtLoose Boolean", 2, -0.5, 1.5);
    hTMLastStationOptimizedLowPtTightBool[i] = ibooker.book1D(
        "hTMLastStationOptimizedLowPtTightBool", "TMLastStationOptimizedLowPtTight Boolean", 2, -0.5, 1.5);
    hGMTkChiCompatibilityBool[i] =
        ibooker.book1D("hGMTkChiCompatibilityBool", "GMTkChiCompatibility Boolean", 2, -0.5, 1.5);
    hGMStaChiCompatibilityBool[i] =
        ibooker.book1D("hGMStaChiCompatibilityBool", "GMStaChiCompatibility Boolean", 2, -0.5, 1.5);
    hGMTkKinkTightBool[i] = ibooker.book1D("hGMTkKinkTightBool", "GMTkKinkTight Boolean", 2, -0.5, 1.5);
    hTMLastStationAngLooseBool[i] =
        ibooker.book1D("hTMLastStationAngLooseBool", "TMLastStationAngLoose Boolean", 2, -0.5, 1.5);
    hTMLastStationAngTightBool[i] =
        ibooker.book1D("hTMLastStationAngTightBool", "TMLastStationAngTight Boolean", 2, -0.5, 1.5);
    hTMOneStationAngLooseBool[i] =
        ibooker.book1D("hTMOneStationAngLooseBool", "TMOneStationAngLoose Boolean", 2, -0.5, 1.5);
    hTMOneStationAngTightBool[i] =
        ibooker.book1D("hTMOneStationAngTightBool", "TMOneStationAngTight Boolean", 2, -0.5, 1.5);
    hTMLastStationOptimizedBarrelLowPtLooseBool[i] = ibooker.book1D(
        "hTMLastStationOptimizedBarrelLowPtLooseBool", "TMLastStationOptimizedBarrelLowPtLoose Boolean", 2, -0.5, 1.5);
    hTMLastStationOptimizedBarrelLowPtTightBool[i] = ibooker.book1D(
        "hTMLastStationOptimizedBarrelLowPtTightBool", "TMLastStationOptimizedBarrelLowPtTight Boolean", 2, -0.5, 1.5);

    if (makeCosmicCompatibilityPlots_) {
      hCombinedCosmicCompat[i] =
          ibooker.book1D("hCombinedCosmicCompat", "hCombinedCosmicCompatibility float", 40, 0., 10.);
      hTimeCosmicCompat[i] = ibooker.book1D("hTimeCosmicCompat", "hTimeCosmicCompatibility float", 6, 0., 3.);
      hB2BCosmicCompat[i] = ibooker.book1D("hB2BCosmicCompat", "Number of back-to-back partners", 10, 0, 10);
      hOverlapCosmicCompat[i] = ibooker.book1D("hOverlapCosmicCompat", "Overlap between muons and 1Leg", 2, 0, 2);
    }

    // by station
    for (int station = 0; station < 4; ++station) {
      if (makeShowerInformationPlots_) {
        sprintf(name, "hMuonShowerSizeT%i", station + 1);
        sprintf(title, "Station %i Transverse Cluster Size", station + 1);
        hMuonShowerSizeT[i][station] = ibooker.book1D(name, title, 1000, 0, 500);
        sprintf(name, "hMuonShowerDeltaR%i", station + 1);
        sprintf(title, "Station %i DeltaR", station + 1);
        hMuonShowerDeltaR[i][station] = ibooker.book1D(name, title, 5000, 0, 0.5);
        sprintf(name, "hMuonAllHits%i", station + 1);
        sprintf(title, "Station %i Number of 1D DT or 2D CSC RecHits", station + 1);
        hMuonAllHits[i][station] = ibooker.book1D(name, title, 400, 0, 400);
        sprintf(name, "hMuonHitsFromSegments%i", station + 1);
        sprintf(title, "Station %i Hits used by 4D DT or 3D CSC Segments", station + 1);
        hMuonHitsFromSegments[i][station] = ibooker.book1D(name, title, 400, 0, 400);
        sprintf(name, "hMuonUncorrelatedHits%i", station + 1);
        sprintf(title, "Station %i Uncorrelated Hits", station + 1);
        hMuonUncorrelatedHits[i][station] = ibooker.book1D(name, title, 400, 0, 400);
      }

      sprintf(name, "hDT%iPullxPropErr", station + 1);
      sprintf(title, "DT Station %i Pull X w/ Propagation Error Only", station + 1);
      hDTPullxPropErr[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

      sprintf(name, "hDT%iPulldXdZPropErr", station + 1);
      sprintf(title, "DT Station %i Pull DxDz w/ Propagation Error Only", station + 1);
      hDTPulldXdZPropErr[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

      if (station < 3) {
        sprintf(name, "hDT%iPullyPropErr", station + 1);
        sprintf(title, "DT Station %i Pull Y w/ Propagation Error Only", station + 1);
        hDTPullyPropErr[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

        sprintf(name, "hDT%iPulldYdZPropErr", station + 1);
        sprintf(title, "DT Station %i Pull DyDz w/ Propagation Error Only", station + 1);
        hDTPulldYdZPropErr[i][station] = ibooker.book1D(name, title, 100, -20., 20.);
      }

      sprintf(name, "hDT%iDistWithSegment", station + 1);
      sprintf(title, "DT Station %i Dist When There Is A Segment", station + 1);
      hDTDistWithSegment[i][station] = ibooker.book1D(name, title, 100, -140., 30.);

      sprintf(name, "hDT%iDistWithNoSegment", station + 1);
      sprintf(title, "DT Station %i Dist When There Is No Segment", station + 1);
      hDTDistWithNoSegment[i][station] = ibooker.book1D(name, title, 100, -140., 30.);

      sprintf(name, "hDT%iPullDistWithSegment", station + 1);
      sprintf(title, "DT Station %i Pull Dist When There Is A Segment", station + 1);
      hDTPullDistWithSegment[i][station] = ibooker.book1D(name, title, 100, -140., 30.);

      sprintf(name, "hDT%iPullDistWithNoSegment", station + 1);
      sprintf(title, "DT Station %i Pull Dist When There Is No Segment", station + 1);
      hDTPullDistWithNoSegment[i][station] = ibooker.book1D(name, title, 100, -140., 30.);

      sprintf(name, "hCSC%iPullxPropErr", station + 1);
      sprintf(title, "CSC Station %i Pull X w/ Propagation Error Only", station + 1);
      hCSCPullxPropErr[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

      sprintf(name, "hCSC%iPulldXdZPropErr", station + 1);
      sprintf(title, "CSC Station %i Pull DxDz w/ Propagation Error Only", station + 1);
      hCSCPulldXdZPropErr[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

      sprintf(name, "hCSC%iPullyPropErr", station + 1);
      sprintf(title, "CSC Station %i Pull Y w/ Propagation Error Only", station + 1);
      hCSCPullyPropErr[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

      sprintf(name, "hCSC%iPulldYdZPropErr", station + 1);
      sprintf(title, "CSC Station %i Pull DyDz w/ Propagation Error Only", station + 1);
      hCSCPulldYdZPropErr[i][station] = ibooker.book1D(name, title, 100, -50., 50.);

      sprintf(name, "hCSC%iDistWithSegment", station + 1);
      sprintf(title, "CSC Station %i Dist When There Is A Segment", station + 1);
      hCSCDistWithSegment[i][station] = ibooker.book1D(name, title, 100, -70., 20.);

      sprintf(name, "hCSC%iDistWithNoSegment", station + 1);
      sprintf(title, "CSC Station %i Dist When There Is No Segment", station + 1);
      hCSCDistWithNoSegment[i][station] = ibooker.book1D(name, title, 100, -70., 20.);

      sprintf(name, "hCSC%iPullDistWithSegment", station + 1);
      sprintf(title, "CSC Station %i Pull Dist When There Is A Segment", station + 1);
      hCSCPullDistWithSegment[i][station] = ibooker.book1D(name, title, 100, -70., 20.);

      sprintf(name, "hCSC%iPullDistWithNoSegment", station + 1);
      sprintf(title, "CSC Station %i Pull Dist When There Is No Segment", station + 1);
      hCSCPullDistWithNoSegment[i][station] = ibooker.book1D(name, title, 100, -70., 20.);
    }  // station
  }

  if (make2DPlots_) {
    ibooker.setCurrentFolder(baseFolder_);
    hSegmentIsAssociatedRZ =
        ibooker.book2D("hSegmentIsAssociatedRZ", "R-Z of Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
    hSegmentIsAssociatedXY =
        ibooker.book2D("hSegmentIsAssociatedXY", "X-Y of Associated Segments", 1700, -850., 850., 1700, -850., 850.);
    hSegmentIsNotAssociatedRZ = ibooker.book2D(
        "hSegmentIsNotAssociatedRZ", "R-Z of Not Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
    hSegmentIsNotAssociatedXY = ibooker.book2D(
        "hSegmentIsNotAssociatedXY", "X-Y of Not Associated Segments", 1700, -850., 850., 1700, -850., 850.);
    hSegmentIsBestDrAssociatedRZ = ibooker.book2D("hSegmentIsBestDrAssociatedRZ",
                                                  "R-Z of Best in Station by #DeltaR Associated Segments",
                                                  2140,
                                                  -1070.,
                                                  1070.,
                                                  850,
                                                  0.,
                                                  850.);
    hSegmentIsBestDrAssociatedXY = ibooker.book2D("hSegmentIsBestDrAssociatedXY",
                                                  "X-Y of Best in Station by #DeltaR Associated Segments",
                                                  1700,
                                                  -850.,
                                                  850.,
                                                  1700,
                                                  -850.,
                                                  850.);
    hSegmentIsBestDrNotAssociatedRZ = ibooker.book2D("hSegmentIsBestDrNotAssociatedRZ",
                                                     "R-Z of Best in Station by #DeltaR Not Associated Segments",
                                                     2140,
                                                     -1070.,
                                                     1070.,
                                                     850,
                                                     0.,
                                                     850.);
    hSegmentIsBestDrNotAssociatedXY = ibooker.book2D("hSegmentIsBestDrNotAssociatedXY",
                                                     "X-Y of Best in Station by #DeltaR Not Associated Segments",
                                                     1700,
                                                     -850.,
                                                     850.,
                                                     1700,
                                                     -850.,
                                                     850.);
  }

  if (useTrackerMuons_ && makeAllChamberPlots_) {
    ibooker.setCurrentFolder(baseFolder_ + "/TrackerMuons");

    // by chamber
    for (int station = 0; station < 4; ++station) {
      // DT wheels: -2 -> 2
      for (int wheel = 0; wheel < 5; ++wheel) {
        // DT sectors: 1 -> 14
        for (int sector = 0; sector < 14; ++sector) {
          sprintf(name, "hDTChamberDx_%i_%i_%i", station + 1, wheel - 2, sector + 1);
          sprintf(title, "DT Chamber Delta X: Station %i Wheel %i Sector %i", station + 1, wheel - 2, sector + 1);
          hDTChamberDx[station][wheel][sector] = ibooker.book1D(name, title, 100, -100., 100.);

          if (station < 3) {
            sprintf(name, "hDTChamberDy_%i_%i_%i", station + 1, wheel - 2, sector + 1);
            sprintf(title, "DT Chamber Delta Y: Station %i Wheel %i Sector %i", station + 1, wheel - 2, sector + 1);
            hDTChamberDy[station][wheel][sector] = ibooker.book1D(name, title, 100, -150., 150.);
          }

          sprintf(name, "hDTChamberEdgeXWithSegment_%i_%i_%i", station + 1, wheel - 2, sector + 1);
          sprintf(title,
                  "DT Chamber Edge X When There Is A Segment: Station %i Wheel "
                  "%i Sector %i",
                  station + 1,
                  wheel - 2,
                  sector + 1);
          hDTChamberEdgeXWithSegment[station][wheel][sector] = ibooker.book1D(name, title, 100, -140., 30.);

          sprintf(name, "hDTChamberEdgeXWithNoSegment_%i_%i_%i", station + 1, wheel - 2, sector + 1);
          sprintf(title,
                  "DT Chamber Edge X When There Is No Segment: Station %i "
                  "Wheel %i Sector %i",
                  station + 1,
                  wheel - 2,
                  sector + 1);
          hDTChamberEdgeXWithNoSegment[station][wheel][sector] = ibooker.book1D(name, title, 100, -140., 30.);

          sprintf(name, "hDTChamberEdgeYWithSegment_%i_%i_%i", station + 1, wheel - 2, sector + 1);
          sprintf(title,
                  "DT Chamber Edge Y When There Is A Segment: Station %i Wheel "
                  "%i Sector %i",
                  station + 1,
                  wheel - 2,
                  sector + 1);
          hDTChamberEdgeYWithSegment[station][wheel][sector] = ibooker.book1D(name, title, 100, -140., 30.);

          sprintf(name, "hDTChamberEdgeYWithNoSegment_%i_%i_%i", station + 1, wheel - 2, sector + 1);
          sprintf(title,
                  "DT Chamber Edge Y When There Is No Segment: Station %i "
                  "Wheel %i Sector %i",
                  station + 1,
                  wheel - 2,
                  sector + 1);
          hDTChamberEdgeYWithNoSegment[station][wheel][sector] = ibooker.book1D(name, title, 100, -140., 30.);
        }  // sector
      }    // wheel

      // CSC endcaps: 1 -> 2
      for (int endcap = 0; endcap < 2; ++endcap) {
        // CSC rings: 1 -> 4
        for (int ring = 0; ring < 4; ++ring) {
          // CSC chambers: 1 -> 36
          for (int chamber = 0; chamber < 36; ++chamber) {
            sprintf(name, "hCSCChamberDx_%i_%i_%i_%i", endcap + 1, station + 1, ring + 1, chamber + 1);
            sprintf(title,
                    "CSC Chamber Delta X: Endcap %i Station %i Ring %i Chamber %i",
                    endcap + 1,
                    station + 1,
                    ring + 1,
                    chamber + 1);
            hCSCChamberDx[endcap][station][ring][chamber] = ibooker.book1D(name, title, 100, -50., 50.);

            sprintf(name, "hCSCChamberDy_%i_%i_%i_%i", endcap + 1, station + 1, ring + 1, chamber + 1);
            sprintf(title,
                    "CSC Chamber Delta Y: Endcap %i Station %i Ring %i Chamber %i",
                    endcap + 1,
                    station + 1,
                    ring + 1,
                    chamber + 1);
            hCSCChamberDy[endcap][station][ring][chamber] = ibooker.book1D(name, title, 100, -50., 50.);

            sprintf(name, "hCSCChamberEdgeXWithSegment_%i_%i_%i_%i", endcap + 1, station + 1, ring + 1, chamber + 1);
            sprintf(title,
                    "CSC Chamber Edge X When There Is A Segment: Endcap %i "
                    "Station %i Ring %i Chamber %i",
                    endcap + 1,
                    station + 1,
                    ring + 1,
                    chamber + 1);
            hCSCChamberEdgeXWithSegment[endcap][station][ring][chamber] = ibooker.book1D(name, title, 100, -70., 20.);

            sprintf(name, "hCSCChamberEdgeXWithNoSegment_%i_%i_%i_%i", endcap + 1, station + 1, ring + 1, chamber + 1);
            sprintf(title,
                    "CSC Chamber Edge X When There Is No Segment: Endcap %i "
                    "Station %i Ring %i Chamber %i",
                    endcap + 1,
                    station + 1,
                    ring + 1,
                    chamber + 1);
            hCSCChamberEdgeXWithNoSegment[endcap][station][ring][chamber] = ibooker.book1D(name, title, 100, -70., 20.);

            sprintf(name, "hCSCChamberEdgeYWithSegment_%i_%i_%i_%i", endcap + 1, station + 1, ring + 1, chamber + 1);
            sprintf(title,
                    "CSC Chamber Edge Y When There Is A Segment: Endcap %i "
                    "Station %i Ring %i Chamber %i",
                    endcap + 1,
                    station + 1,
                    ring + 1,
                    chamber + 1);
            hCSCChamberEdgeYWithSegment[endcap][station][ring][chamber] = ibooker.book1D(name, title, 100, -70., 20.);

            sprintf(name, "hCSCChamberEdgeYWithNoSegment_%i_%i_%i_%i", endcap + 1, station + 1, ring + 1, chamber + 1);
            sprintf(title,
                    "CSC Chamber Edge Y When There Is No Segment: Endcap %i "
                    "Station %i Ring %i Chamber %i",
                    endcap + 1,
                    station + 1,
                    ring + 1,
                    chamber + 1);
            hCSCChamberEdgeYWithNoSegment[endcap][station][ring][chamber] = ibooker.book1D(name, title, 100, -70., 20.);
          }  // chamber
        }    // ring
      }      // endcap
    }        // station
  }
}

void MuonIdVal::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace reco;

  iEvent.getByToken(inputMuonCollectionToken_, muonCollectionH_);
  iEvent.getByToken(inputDTRecSegment4DCollectionToken_, dtSegmentCollectionH_);
  iEvent.getByToken(inputCSCSegmentCollectionToken_, cscSegmentCollectionH_);
  iEvent.getByToken(inputMuonTimeExtraValueMapCombToken_, combinedMuonTimeExtraValueMapH_);
  iEvent.getByToken(inputMuonTimeExtraValueMapCSCToken_, cscMuonTimeExtraValueMapH_);
  iEvent.getByToken(inputMuonTimeExtraValueMapDTToken_, dtMuonTimeExtraValueMapH_);
  iEvent.getByToken(inputMuonShowerInformationValueMapToken_, muonShowerInformationValueMapH_);
  iEvent.getByToken(inputMuonCosmicCompatibilityValueMapToken_, muonCosmicCompatibilityValueMapH_);

  geometry_ = iSetup.getHandle(trackingGeomToken_);

  unsigned int muonIdx = 0;
  for (MuonCollection::const_iterator muon = muonCollectionH_->begin(); muon != muonCollectionH_->end(); ++muon) {
    // trackerMuon == 0; globalMuon == 1; trackerMuon && !globalMuon == 2;
    // globalMuon && !trackerMuon == 3
    for (unsigned int i = 0; i < 4; i++) {
      if (i == 0 && (!useTrackerMuons_ || !muon->isTrackerMuon()))
        continue;
      if (i == 1 && (!useGlobalMuons_ || !muon->isGlobalMuon()))
        continue;
      if (i == 2 && (!useTrackerMuonsNotGlobalMuons_ || (!(muon->isTrackerMuon() && !muon->isGlobalMuon()))))
        continue;
      if (i == 3 && (!useGlobalMuonsNotTrackerMuons_ || (!(muon->isGlobalMuon() && !muon->isTrackerMuon()))))
        continue;

      if (makeEnergyPlots_ && muon->isEnergyValid()) {
        // EM
        if (fabs(muon->eta()) > 1.479)
          hEnergyEMEndcap[i]->Fill(muon->calEnergy().em);
        else
          hEnergyEMBarrel[i]->Fill(muon->calEnergy().em);
        // HAD
        if (fabs(muon->eta()) > 1.4)
          hEnergyHAEndcap[i]->Fill(muon->calEnergy().had);
        else
          hEnergyHABarrel[i]->Fill(muon->calEnergy().had);
        // HO
        if (fabs(muon->eta()) < 1.26)
          hEnergyHO[i]->Fill(muon->calEnergy().ho);
      }

      if (makeTimePlots_) {
        if (muon->isTimeValid()) {
          hMuonTimeNDOF[i]->Fill(muon->time().nDof);
          hMuonTimeTimeAtIpInOut[i]->Fill(muon->time().timeAtIpInOut);
          hMuonTimeTimeAtIpInOutErr[i]->Fill(muon->time().timeAtIpInOutErr);
          hMuonTimeTimeAtIpOutIn[i]->Fill(muon->time().timeAtIpOutIn);
          hMuonTimeTimeAtIpOutInErr[i]->Fill(muon->time().timeAtIpOutInErr);
        }

        MuonRef muonRef(muonCollectionH_, muonIdx);
        MuonTimeExtra combinedMuonTimeExtra = (*combinedMuonTimeExtraValueMapH_)[muonRef];
        MuonTimeExtra cscMuonTimeExtra = (*cscMuonTimeExtraValueMapH_)[muonRef];
        MuonTimeExtra dtMuonTimeExtra = (*dtMuonTimeExtraValueMapH_)[muonRef];

        hMuonTimeExtraCombinedNDOF[i]->Fill(combinedMuonTimeExtra.nDof());
        hMuonTimeExtraCombinedTimeAtIpInOut[i]->Fill(combinedMuonTimeExtra.timeAtIpInOut());
        hMuonTimeExtraCombinedTimeAtIpInOutErr[i]->Fill(combinedMuonTimeExtra.timeAtIpInOutErr());
        hMuonTimeExtraCombinedTimeAtIpOutIn[i]->Fill(combinedMuonTimeExtra.timeAtIpOutIn());
        hMuonTimeExtraCombinedTimeAtIpOutInErr[i]->Fill(combinedMuonTimeExtra.timeAtIpOutInErr());
        hMuonTimeExtraCSCNDOF[i]->Fill(cscMuonTimeExtra.nDof());
        hMuonTimeExtraCSCTimeAtIpInOut[i]->Fill(cscMuonTimeExtra.timeAtIpInOut());
        hMuonTimeExtraCSCTimeAtIpInOutErr[i]->Fill(cscMuonTimeExtra.timeAtIpInOutErr());
        hMuonTimeExtraCSCTimeAtIpOutIn[i]->Fill(cscMuonTimeExtra.timeAtIpOutIn());
        hMuonTimeExtraCSCTimeAtIpOutInErr[i]->Fill(cscMuonTimeExtra.timeAtIpOutInErr());
        hMuonTimeExtraDTNDOF[i]->Fill(dtMuonTimeExtra.nDof());
        hMuonTimeExtraDTTimeAtIpInOut[i]->Fill(dtMuonTimeExtra.timeAtIpInOut());
        hMuonTimeExtraDTTimeAtIpInOutErr[i]->Fill(dtMuonTimeExtra.timeAtIpInOutErr());
        hMuonTimeExtraDTTimeAtIpOutIn[i]->Fill(dtMuonTimeExtra.timeAtIpOutIn());
        hMuonTimeExtraDTTimeAtIpOutInErr[i]->Fill(dtMuonTimeExtra.timeAtIpOutInErr());
      }

      if (muon->isCaloCompatibilityValid())
        hCaloCompat[i]->Fill(muon->caloCompatibility());
      hSegmentCompat[i]->Fill(muon::segmentCompatibility(*muon));
      if (make2DPlots_ && muon->isCaloCompatibilityValid())
        hCaloSegmentCompat[i]->Fill(muon->caloCompatibility(), muon::segmentCompatibility(*muon));
      if (muon->isQualityValid()) {
        hMuonQualityTrkRelChi2[i]->Fill(muon->combinedQuality().trkRelChi2);
        hMuonQualityStaRelChi2[i]->Fill(muon->combinedQuality().staRelChi2);
        hMuonQualityTrkKink[i]->Fill(muon->combinedQuality().trkKink);
      }
      hGlobalMuonPromptTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::GlobalMuonPromptTight));
      hTMLastStationLooseBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMLastStationLoose));
      hTMLastStationTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMLastStationTight));
      hTM2DCompatibilityLooseBool[i]->Fill(muon::isGoodMuon(*muon, muon::TM2DCompatibilityLoose));
      hTM2DCompatibilityTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::TM2DCompatibilityTight));
      hTMOneStationLooseBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMOneStationLoose));
      hTMOneStationTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMOneStationTight));
      hTMLastStationOptimizedLowPtLooseBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMLastStationOptimizedLowPtLoose));
      hTMLastStationOptimizedLowPtTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMLastStationOptimizedLowPtTight));
      hGMTkChiCompatibilityBool[i]->Fill(muon::isGoodMuon(*muon, muon::GMTkChiCompatibility));
      hGMStaChiCompatibilityBool[i]->Fill(muon::isGoodMuon(*muon, muon::GMStaChiCompatibility));
      hGMTkKinkTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::GMTkKinkTight));
      hTMLastStationAngLooseBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMLastStationAngLoose));
      hTMLastStationAngTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMLastStationAngTight));
      hTMOneStationAngLooseBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMOneStationAngLoose));
      hTMOneStationAngTightBool[i]->Fill(muon::isGoodMuon(*muon, muon::TMOneStationAngTight));
      hTMLastStationOptimizedBarrelLowPtLooseBool[i]->Fill(
          muon::isGoodMuon(*muon, muon::TMLastStationOptimizedBarrelLowPtLoose));
      hTMLastStationOptimizedBarrelLowPtTightBool[i]->Fill(
          muon::isGoodMuon(*muon, muon::TMLastStationOptimizedBarrelLowPtTight));

      if (makeCosmicCompatibilityPlots_) {
        MuonRef muonRef(muonCollectionH_, muonIdx);
        MuonCosmicCompatibility muonCosmicCompatibility = (*muonCosmicCompatibilityValueMapH_)[muonRef];
        hCombinedCosmicCompat[i]->Fill(muonCosmicCompatibility.cosmicCompatibility);
        hTimeCosmicCompat[i]->Fill(muonCosmicCompatibility.timeCompatibility);
        hB2BCosmicCompat[i]->Fill(muonCosmicCompatibility.backToBackCompatibility);
        hOverlapCosmicCompat[i]->Fill(muonCosmicCompatibility.overlapCompatibility);
      }

      // by station
      for (int station = 0; station < 4; ++station) {
        if (makeShowerInformationPlots_) {
          MuonRef muonRef(muonCollectionH_, muonIdx);
          MuonShower muonShowerInformation = (*muonShowerInformationValueMapH_)[muonRef];

          hMuonShowerSizeT[i][station]->Fill((muonShowerInformation.stationShowerSizeT).at(station));
          hMuonShowerDeltaR[i][station]->Fill((muonShowerInformation.stationShowerDeltaR.at(station)));
          hMuonAllHits[i][station]->Fill((muonShowerInformation.nStationHits.at(station)));
          hMuonHitsFromSegments[i][station]->Fill((muonShowerInformation.nStationCorrelatedHits.at(station)));
          hMuonUncorrelatedHits[i][station]->Fill((muonShowerInformation.nStationHits.at(station)) -
                                                  muonShowerInformation.nStationCorrelatedHits.at(station));
        }

        Fill(hDTPullxPropErr[i][station],
             muon->pullX(station + 1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, false));
        Fill(hDTPulldXdZPropErr[i][station],
             muon->pullDxDz(station + 1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, false));

        if (station < 3) {
          Fill(hDTPullyPropErr[i][station],
               muon->pullY(station + 1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, false));
          Fill(hDTPulldYdZPropErr[i][station],
               muon->pullDyDz(station + 1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, false));
        }

        float distance = muon->trackDist(station + 1, MuonSubdetId::DT);
        float error = muon->trackDistErr(station + 1, MuonSubdetId::DT);
        if (error == 0)
          error = 0.000001;

        if (muon->numberOfSegments(station + 1, MuonSubdetId::DT, Muon::NoArbitration) > 0) {
          Fill(hDTDistWithSegment[i][station], distance);
          Fill(hDTPullDistWithSegment[i][station], distance / error);
        } else {
          Fill(hDTDistWithNoSegment[i][station], distance);
          Fill(hDTPullDistWithNoSegment[i][station], distance / error);
        }

        Fill(hCSCPullxPropErr[i][station],
             muon->pullX(station + 1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, false));
        Fill(hCSCPulldXdZPropErr[i][station],
             muon->pullDxDz(station + 1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, false));
        Fill(hCSCPullyPropErr[i][station],
             muon->pullY(station + 1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, false));
        Fill(hCSCPulldYdZPropErr[i][station],
             muon->pullDyDz(station + 1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, false));

        distance = muon->trackDist(station + 1, MuonSubdetId::CSC);
        error = muon->trackDistErr(station + 1, MuonSubdetId::CSC);
        if (error == 0)
          error = 0.000001;

        if (muon->numberOfSegments(station + 1, MuonSubdetId::CSC, Muon::NoArbitration) > 0) {
          Fill(hCSCDistWithSegment[i][station], distance);
          Fill(hCSCPullDistWithSegment[i][station], distance / error);
        } else {
          Fill(hCSCDistWithNoSegment[i][station], distance);
          Fill(hCSCPullDistWithNoSegment[i][station], distance / error);
        }
      }  // station
    }

    if (!useTrackerMuons_ || !muon->isTrackerMuon())
      continue;
    if (makeAllChamberPlots_) {
      // by chamber
      for (std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
           chamberMatch != muon->matches().end();
           ++chamberMatch) {
        int station = chamberMatch->station();

        if (chamberMatch->detector() == MuonSubdetId::DT) {
          DTChamberId dtId(chamberMatch->id.rawId());
          int wheel = dtId.wheel();
          int sector = dtId.sector();

          if (chamberMatch->segmentMatches.empty()) {
            Fill(hDTChamberEdgeXWithNoSegment[station - 1][wheel + 2][sector - 1], chamberMatch->edgeX);
            Fill(hDTChamberEdgeYWithNoSegment[station - 1][wheel + 2][sector - 1], chamberMatch->edgeY);
          } else {
            Fill(hDTChamberEdgeXWithSegment[station - 1][wheel + 2][sector - 1], chamberMatch->edgeX);
            Fill(hDTChamberEdgeYWithSegment[station - 1][wheel + 2][sector - 1], chamberMatch->edgeY);

            for (std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                 segmentMatch != chamberMatch->segmentMatches.end();
                 ++segmentMatch) {
              if (segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR)) {
                Fill(hDTChamberDx[station - 1][wheel + 2][sector - 1], chamberMatch->x - segmentMatch->x);
                if (station < 4)
                  Fill(hDTChamberDy[station - 1][wheel + 2][sector - 1], chamberMatch->y - segmentMatch->y);
                break;
              }
            }  // segmentMatch
          }

          continue;
        }

        if (chamberMatch->detector() == MuonSubdetId::CSC) {
          CSCDetId cscId(chamberMatch->id.rawId());
          int endcap = cscId.endcap();
          int ring = cscId.ring();
          int chamber = cscId.chamber();

          if (chamberMatch->segmentMatches.empty()) {
            Fill(hCSCChamberEdgeXWithNoSegment[endcap - 1][station - 1][ring - 1][chamber - 1], chamberMatch->edgeX);
            Fill(hCSCChamberEdgeYWithNoSegment[endcap - 1][station - 1][ring - 1][chamber - 1], chamberMatch->edgeY);
          } else {
            Fill(hCSCChamberEdgeXWithSegment[endcap - 1][station - 1][ring - 1][chamber - 1], chamberMatch->edgeX);
            Fill(hCSCChamberEdgeYWithSegment[endcap - 1][station - 1][ring - 1][chamber - 1], chamberMatch->edgeY);

            for (std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                 segmentMatch != chamberMatch->segmentMatches.end();
                 ++segmentMatch) {
              if (segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR)) {
                Fill(hCSCChamberDx[endcap - 1][station - 1][ring - 1][chamber - 1], chamberMatch->x - segmentMatch->x);
                Fill(hCSCChamberDy[endcap - 1][station - 1][ring - 1][chamber - 1], chamberMatch->y - segmentMatch->y);
                break;
              }
            }  // segmentMatch
          }
        }
      }  // chamberMatch
    }
    ++muonIdx;
  }  // muon

  if (!make2DPlots_)
    return;

  for (DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH_->begin();
       segment != dtSegmentCollectionH_->end();
       ++segment) {
    LocalPoint segmentLocalPosition = segment->localPosition();
    LocalVector segmentLocalDirection = segment->localDirection();
    LocalError segmentLocalPositionError = segment->localPositionError();
    LocalError segmentLocalDirectionError = segment->localDirectionError();
    const GeomDet *segmentGeomDet = geometry_->idToDet(segment->geographicalId());
    GlobalPoint segmentGlobalPosition = segmentGeomDet->toGlobal(segment->localPosition());
    bool segmentFound = false;
    bool segmentBestDrFound = false;

    for (MuonCollection::const_iterator muon = muonCollectionH_->begin(); muon != muonCollectionH_->end(); ++muon) {
      if (!muon->isMatchesValid())
        continue;

      for (std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
           chamberMatch != muon->matches().end();
           ++chamberMatch) {
        for (std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
             segmentMatch != chamberMatch->segmentMatches.end();
             ++segmentMatch) {
          if (fabs(segmentMatch->x - segmentLocalPosition.x()) < 1E-6 &&
              fabs(segmentMatch->y - segmentLocalPosition.y()) < 1E-6 &&
              fabs(segmentMatch->dXdZ - segmentLocalDirection.x() / segmentLocalDirection.z()) < 1E-6 &&
              fabs(segmentMatch->dYdZ - segmentLocalDirection.y() / segmentLocalDirection.z()) < 1E-6 &&
              fabs(segmentMatch->xErr - sqrt(segmentLocalPositionError.xx())) < 1E-6 &&
              fabs(segmentMatch->yErr - sqrt(segmentLocalPositionError.yy())) < 1E-6 &&
              fabs(segmentMatch->dXdZErr - sqrt(segmentLocalDirectionError.xx())) < 1E-6 &&
              fabs(segmentMatch->dYdZErr - sqrt(segmentLocalDirectionError.yy())) < 1E-6) {
            segmentFound = true;
            if (segmentMatch->isMask(reco::MuonSegmentMatch::BestInStationByDR))
              segmentBestDrFound = true;
            break;
          }
        }  // segmentMatch
        if (segmentFound)
          break;
      }  // chamberMatch
      if (segmentFound)
        break;
    }  // muon

    if (segmentFound) {
      hSegmentIsAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
      hSegmentIsAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());

      if (segmentBestDrFound) {
        hSegmentIsBestDrAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
        hSegmentIsBestDrAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
      }
    } else {
      hSegmentIsNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
      hSegmentIsNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
      hSegmentIsBestDrNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
      hSegmentIsBestDrNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
    }
  }  // dt segment

  for (CSCSegmentCollection::const_iterator segment = cscSegmentCollectionH_->begin();
       segment != cscSegmentCollectionH_->end();
       ++segment) {
    LocalPoint segmentLocalPosition = segment->localPosition();
    LocalVector segmentLocalDirection = segment->localDirection();
    LocalError segmentLocalPositionError = segment->localPositionError();
    LocalError segmentLocalDirectionError = segment->localDirectionError();
    const GeomDet *segmentGeomDet = geometry_->idToDet(segment->geographicalId());
    GlobalPoint segmentGlobalPosition = segmentGeomDet->toGlobal(segment->localPosition());
    bool segmentFound = false;
    bool segmentBestDrFound = false;

    for (MuonCollection::const_iterator muon = muonCollectionH_->begin(); muon != muonCollectionH_->end(); ++muon) {
      if (!muon->isMatchesValid())
        continue;

      for (std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
           chamberMatch != muon->matches().end();
           ++chamberMatch) {
        for (std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
             segmentMatch != chamberMatch->segmentMatches.end();
             ++segmentMatch) {
          if (fabs(segmentMatch->x - segmentLocalPosition.x()) < 1E-6 &&
              fabs(segmentMatch->y - segmentLocalPosition.y()) < 1E-6 &&
              fabs(segmentMatch->dXdZ - segmentLocalDirection.x() / segmentLocalDirection.z()) < 1E-6 &&
              fabs(segmentMatch->dYdZ - segmentLocalDirection.y() / segmentLocalDirection.z()) < 1E-6 &&
              fabs(segmentMatch->xErr - sqrt(segmentLocalPositionError.xx())) < 1E-6 &&
              fabs(segmentMatch->yErr - sqrt(segmentLocalPositionError.yy())) < 1E-6 &&
              fabs(segmentMatch->dXdZErr - sqrt(segmentLocalDirectionError.xx())) < 1E-6 &&
              fabs(segmentMatch->dYdZErr - sqrt(segmentLocalDirectionError.yy())) < 1E-6) {
            segmentFound = true;
            if (segmentMatch->isMask(reco::MuonSegmentMatch::BestInStationByDR))
              segmentBestDrFound = true;
            break;
          }
        }  // segmentMatch
        if (segmentFound)
          break;
      }  // chamberMatch
      if (segmentFound)
        break;
    }  // muon

    if (segmentFound) {
      hSegmentIsAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
      hSegmentIsAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());

      if (segmentBestDrFound) {
        hSegmentIsBestDrAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
        hSegmentIsBestDrAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
      }
    } else {
      hSegmentIsNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
      hSegmentIsNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
      hSegmentIsBestDrNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
      hSegmentIsBestDrNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
    }
  }  // csc segment
}

void MuonIdVal::Fill(MonitorElement *me, float f) {
  if (fabs(f) > 900000)
    return;
  // if (fabs(f) < 1E-8) return;
  me->Fill(f);
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonIdVal);

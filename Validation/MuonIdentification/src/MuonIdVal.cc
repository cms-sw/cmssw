#include "Validation/MuonIdentification/interface/MuonIdVal.h"

MuonIdVal::MuonIdVal(const edm::ParameterSet& iConfig)
{
   inputMuonCollection_ = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
   inputDTRecSegment4DCollection_ = iConfig.getParameter<edm::InputTag>("inputDTRecSegment4DCollection");
   inputCSCSegmentCollection_ = iConfig.getParameter<edm::InputTag>("inputCSCSegmentCollection");
   useTrackerMuons_ = iConfig.getUntrackedParameter<bool>("useTrackerMuons");
   useGlobalMuons_ = iConfig.getUntrackedParameter<bool>("useGlobalMuons");
   makeDQMPlots_ = iConfig.getUntrackedParameter<bool>("makeDQMPlots");
   makeEnergyPlots_ = iConfig.getUntrackedParameter<bool>("makeEnergyPlots");
   makeIsoPlots_ = iConfig.getUntrackedParameter<bool>("makeIsoPlots");
   make2DPlots_ = iConfig.getUntrackedParameter<bool>("make2DPlots");
   makeAllChamberPlots_ = iConfig.getUntrackedParameter<bool>("makeAllChamberPlots");
   baseFolder_ = iConfig.getUntrackedParameter<std::string>("baseFolder");

   // If makeDQMPlots_ then disable everything else, and all but a few choice plots
   if (makeDQMPlots_) {
      makeEnergyPlots_ = false;
      makeIsoPlots_ = false;
      make2DPlots_ = false;
      makeAllChamberPlots_ = false;
   }

   dbe_ = 0;
   dbe_ = edm::Service<DQMStore>().operator->();
}

MuonIdVal::~MuonIdVal() {}

void 
MuonIdVal::beginJob(const edm::EventSetup&)
{
   // trackerMuon == 0; globalMuon == 1
   for (unsigned int i = 0; i < 2; i++) {
      if ((i == 0 && ! useTrackerMuons_) || (i == 1 && ! useGlobalMuons_)) continue;
      if (i == 0) dbe_->setCurrentFolder(baseFolder_+"/TrackerMuons");
      if (i == 1) dbe_->setCurrentFolder(baseFolder_+"/GlobalMuons");

      hNumChambers[i] = dbe_->book1D("hNumChambers", "Number of Chambers", 11, -0.5, 10.5);
      hNumMatches[i] = dbe_->book1D("hNumMatches", "Number of Matches", 11, -0.5, 10.5);
      if (! makeDQMPlots_) {
         hCaloCompat[i] = dbe_->book1D("hCaloCompat", "Calo Compatibility", 101, -0.05, 1.05);
         hSegmentCompat[i] = dbe_->book1D("hSegmentCompat", "Segment Compatibility", 101, -0.05, 1.05);
         if (make2DPlots_)
            hCaloSegmentCompat[i] = dbe_->book2D("hCaloSegmentCompat", "Calo Compatibility vs. Segment Compatibility", 101, -0.05, 1.05, 101, -0.05, 1.05);
         hGlobalMuonPromptTightBool[i] = dbe_->book1D("hGlobalMuonPromptTightBool", "GlobalMuonPromptTight Boolean", 2, -0.5, 1.5);
         hTMLastStationLooseBool[i] = dbe_->book1D("hTMLastStationLooseBool", "TMLastStationLoose Boolean", 2, -0.5, 1.5);
         hTMLastStationTightBool[i] = dbe_->book1D("hTMLastStationTightBool", "TMLastStationTight Boolean", 2, -0.5, 1.5);
         hTM2DCompatibilityLooseBool[i] = dbe_->book1D("hTM2DCompatibilityLooseBool", "TM2DCompatibilityLoose Boolean", 2, -0.5, 1.5);
         hTM2DCompatibilityTightBool[i] = dbe_->book1D("hTM2DCompatibilityTightBool", "TM2DCompatibilityTight Boolean", 2, -0.5, 1.5);
         hTMOneStationLooseBool[i] = dbe_->book1D("hTMOneStationLooseBool", "TMOneStationLoose Boolean", 2, -0.5, 1.5);
         hTMOneStationTightBool[i] = dbe_->book1D("hTMOneStationTightBool", "TMOneStationTight Boolean", 2, -0.5, 1.5);
         hTMLastStationOptimizedLowPtLooseBool[i] = dbe_->book1D("hTMLastStationOptimizedLowPtLooseBool", "TMLastStationOptimizedLowPtLoose Boolean", 2, -0.5, 1.5);
         hTMLastStationOptimizedLowPtTightBool[i] = dbe_->book1D("hTMLastStationOptimizedLowPtTightBool", "TMLastStationOptimizedLowPtTight Boolean", 2, -0.5, 1.5);

         if (makeEnergyPlots_) {
            hEnergyEMBarrel[i] = dbe_->book1D("hEnergyEMBarrel", "Energy in ECAL Barrel", 100, -0.5, 2.);
            hEnergyHABarrel[i] = dbe_->book1D("hEnergyHABarrel", "Energy in HCAL Barrel", 100, -4., 12.);
            hEnergyHO[i] = dbe_->book1D("hEnergyHO", "Energy HO", 100, -2., 5.);
            hEnergyEMEndcap[i] = dbe_->book1D("hEnergyEMEndcap", "Energy in ECAL Endcap", 100, -0.5, 2.);
            hEnergyHAEndcap[i] = dbe_->book1D("hEnergyHAEndcap", "Energy in HCAL Endcap", 100, -4., 12.);
         }

         if (makeIsoPlots_) {
            hIso03sumPt[i] = dbe_->book1D("hIso03sumPt", "Sum Pt in Cone of 0.3", 100, 0., 10.);
            hIso03emEt[i] = dbe_->book1D("hIso03emEt", "Em Et in Cone of 0.3", 100, 0., 10.);
            hIso03hadEt[i] = dbe_->book1D("hIso03hadEt", "Had Et in Cone of 0.3", 100, 0., 10.);
            hIso03hoEt[i] = dbe_->book1D("hIso03hoEt", "HO Et in Cone of 0.3", 100, 0., 10.);
            hIso03nTracks[i] = dbe_->book1D("hIso03nTracks", "Number of Tracks in Cone of 0.3", 11, -0.5, 10.5);
            hIso03nJets[i] = dbe_->book1D("hIso03nJets", "Number of Jets in Cone of 0.3", 11, -0.5, 10.5);
            hIso05sumPt[i] = dbe_->book1D("hIso05sumPt", "Sum Pt in Cone of 0.5", 100, 0., 10.);
            hIso05emEt[i] = dbe_->book1D("hIso05emEt", "Em Et in Cone of 0.5", 100, 0., 10.);
            hIso05hadEt[i] = dbe_->book1D("hIso05hadEt", "Had Et in Cone of 0.5", 100, 0., 10.);
            hIso05hoEt[i] = dbe_->book1D("hIso05hoEt", "HO Et in Cone of 0.5", 100, 0., 10.);
            hIso05nTracks[i] = dbe_->book1D("hIso05nTracks", "Number of Tracks in Cone of 0.5", 11, -0.5, 10.5);
            hIso05nJets[i] = dbe_->book1D("hIso05nJets", "Number of Jets in Cone of 0.5", 11, -0.5, 10.5);
         }
      }
   }

   if (useTrackerMuons_) {
      dbe_->setCurrentFolder(baseFolder_+"/TrackerMuons");

      char name[100], title[200];

      // by station
      for(int station = 0; station < 4; ++station)
      {
         sprintf(name, "hDT%iNumSegments", station+1);
         sprintf(title, "DT Station %i Number of Segments (No Arbitration)", station+1);
         hDTNumSegments[station] = dbe_->book1D(name, title, 11, -0.5, 10.5);

         sprintf(name, "hDT%iDx", station+1);
         sprintf(title, "DT Station %i Delta X", station+1);
         hDTDx[station] = dbe_->book1D(name, title, 100, -100., 100.);

         sprintf(name, "hDT%iPullx", station+1);
         sprintf(title, "DT Station %i Pull X", station+1);
         hDTPullx[station] = dbe_->book1D(name, title, 100, -20., 20.);

         if (station < 3) {
            sprintf(name, "hDT%iDy", station+1);
            sprintf(title, "DT Station %i Delta Y", station+1);
            hDTDy[station] = dbe_->book1D(name, title, 100, -150., 150.);

            sprintf(name, "hDT%iPully", station+1);
            sprintf(title, "DT Station %i Pull Y", station+1);
            hDTPully[station] = dbe_->book1D(name, title, 100, -20., 20.);

            if (! makeDQMPlots_) {
               sprintf(name, "hDT%iPullyPropErr", station+1);
               sprintf(title, "DT Station %i Pull Y w/ Propagation Error Only", station+1);
               hDTPullyPropErr[station] = dbe_->book1D(name, title, 100, -20., 20.);

               sprintf(name, "hDT%iDdYdZ", station+1);
               sprintf(title, "DT Station %i Delta DyDz", station+1);
               hDTDdYdZ[station] = dbe_->book1D(name, title, 100, -2., 2.);

               sprintf(name, "hDT%iPulldYdZ", station+1);
               sprintf(title, "DT Station %i Pull DyDz", station+1);
               hDTPulldYdZ[station] = dbe_->book1D(name, title, 100, -20., 20.);

               sprintf(name, "hDT%iPulldYdZPropErr", station+1);
               sprintf(title, "DT Station %i Pull DyDz w/ Propagation Error Only", station+1);
               hDTPulldYdZPropErr[station] = dbe_->book1D(name, title, 100, -20., 20.);
            }
         }

         sprintf(name, "hCSC%iNumSegments", station+1);
         sprintf(title, "CSC Station %i Number of Segments (No Arbitration)", station+1);
         hCSCNumSegments[station] = dbe_->book1D(name, title, 11, -0.5, 10.5);

         sprintf(name, "hCSC%iDx", station+1);
         sprintf(title, "CSC Station %i Delta X", station+1);
         hCSCDx[station] = dbe_->book1D(name, title, 100, -50., 50.);

         sprintf(name, "hCSC%iPullx", station+1);
         sprintf(title, "CSC Station %i Pull X", station+1);
         hCSCPullx[station] = dbe_->book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iDy", station+1);
         sprintf(title, "CSC Station %i Delta Y", station+1);
         hCSCDy[station] = dbe_->book1D(name, title, 100, -50., 50.);

         sprintf(name, "hCSC%iPully", station+1);
         sprintf(title, "CSC Station %i Pull Y", station+1);
         hCSCPully[station] = dbe_->book1D(name, title, 100, -20., 20.);

         if (! makeDQMPlots_) {
            sprintf(name, "hDT%iPullxPropErr", station+1);
            sprintf(title, "DT Station %i Pull X w/ Propagation Error Only", station+1);
            hDTPullxPropErr[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hDT%iDdXdZ", station+1);
            sprintf(title, "DT Station %i Delta DxDz", station+1);
            hDTDdXdZ[station] = dbe_->book1D(name, title, 100, -1., 1.);

            sprintf(name, "hDT%iPulldXdZ", station+1);
            sprintf(title, "DT Station %i Pull DxDz", station+1);
            hDTPulldXdZ[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hDT%iPulldXdZPropErr", station+1);
            sprintf(title, "DT Station %i Pull DxDz w/ Propagation Error Only", station+1);
            hDTPulldXdZPropErr[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hDT%iDistWithSegment", station+1);
            sprintf(title, "DT Station %i Dist When There Is A Segment", station+1);
            hDTDistWithSegment[station] = dbe_->book1D(name, title, 100, -140., 30.);

            sprintf(name, "hDT%iDistWithNoSegment", station+1);
            sprintf(title, "DT Station %i Dist When There Is No Segment", station+1);
            hDTDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -140., 30.);

            sprintf(name, "hDT%iPullDistWithSegment", station+1);
            sprintf(title, "DT Station %i Pull Dist When There Is A Segment", station+1);
            hDTPullDistWithSegment[station] = dbe_->book1D(name, title, 100, -140., 30.);

            sprintf(name, "hDT%iPullDistWithNoSegment", station+1);
            sprintf(title, "DT Station %i Pull Dist When There Is No Segment", station+1);
            hDTPullDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -140., 30.);

            sprintf(name, "hCSC%iPullxPropErr", station+1);
            sprintf(title, "CSC Station %i Pull X w/ Propagation Error Only", station+1);
            hCSCPullxPropErr[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hCSC%iDdXdZ", station+1);
            sprintf(title, "CSC Station %i Delta DxDz", station+1);
            hCSCDdXdZ[station] = dbe_->book1D(name, title, 100, -1., 1.);

            sprintf(name, "hCSC%iPulldXdZ", station+1);
            sprintf(title, "CSC Station %i Pull DxDz", station+1);
            hCSCPulldXdZ[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hCSC%iPulldXdZPropErr", station+1);
            sprintf(title, "CSC Station %i Pull DxDz w/ Propagation Error Only", station+1);
            hCSCPulldXdZPropErr[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hCSC%iPullyPropErr", station+1);
            sprintf(title, "CSC Station %i Pull Y w/ Propagation Error Only", station+1);
            hCSCPullyPropErr[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hCSC%iDdYdZ", station+1);
            sprintf(title, "CSC Station %i Delta DyDz", station+1);
            hCSCDdYdZ[station] = dbe_->book1D(name, title, 100, -1., 1.);

            sprintf(name, "hCSC%iPulldYdZ", station+1);
            sprintf(title, "CSC Station %i Pull DyDz", station+1);
            hCSCPulldYdZ[station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hCSC%iPulldYdZPropErr", station+1);
            sprintf(title, "CSC Station %i Pull DyDz w/ Propagation Error Only", station+1);
            hCSCPulldYdZPropErr[station] = dbe_->book1D(name, title, 100, -50., 50.);

            sprintf(name, "hCSC%iDistWithSegment", station+1);
            sprintf(title, "CSC Station %i Dist When There Is A Segment", station+1);
            hCSCDistWithSegment[station] = dbe_->book1D(name, title, 100, -70., 20.);

            sprintf(name, "hCSC%iDistWithNoSegment", station+1);
            sprintf(title, "CSC Station %i Dist When There Is No Segment", station+1);
            hCSCDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -70., 20.);

            sprintf(name, "hCSC%iPullDistWithSegment", station+1);
            sprintf(title, "CSC Station %i Pull Dist When There Is A Segment", station+1);
            hCSCPullDistWithSegment[station] = dbe_->book1D(name, title, 100, -70., 20.);

            sprintf(name, "hCSC%iPullDistWithNoSegment", station+1);
            sprintf(title, "CSC Station %i Pull Dist When There Is No Segment", station+1);
            hCSCPullDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -70., 20.);
         }
      }// station

      hSegmentIsAssociatedBool = dbe_->book1D("hSegmentIsAssociatedBool", "Segment Is Associated Boolean", 2, -0.5, 1.5);
      if (make2DPlots_) {
         hSegmentIsAssociatedRZ = dbe_->book2D("hSegmentIsAssociatedRZ", "R-Z of Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
         hSegmentIsAssociatedXY = dbe_->book2D("hSegmentIsAssociatedXY", "R-#phi of Associated Segments", 1700, -850., 850., 1700, -850., 850.);
         hSegmentIsNotAssociatedRZ = dbe_->book2D("hSegmentIsNotAssociatedRZ", "R-Z of Not Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
         hSegmentIsNotAssociatedXY = dbe_->book2D("hSegmentIsNotAssociatedXY", "R-#phi of Not Associated Segments", 1700, -850., 850., 1700, -850., 850.);
         hSegmentIsBestDrAssociatedRZ = dbe_->book2D("hSegmentIsBestDrAssociatedRZ", "R-Z of Best in Station by #DeltaR Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
         hSegmentIsBestDrAssociatedXY = dbe_->book2D("hSegmentIsBestDrAssociatedXY", "R-#phi of Best in Station by #DeltaR Associated Segments", 1700, -850., 850., 1700, -850., 850.);
         hSegmentIsBestDrNotAssociatedRZ = dbe_->book2D("hSegmentIsBestDrNotAssociatedRZ", "R-Z of Best in Station by #DeltaR Not Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
         hSegmentIsBestDrNotAssociatedXY = dbe_->book2D("hSegmentIsBestDrNotAssociatedXY", "R-#phi of Best in Station by #DeltaR Not Associated Segments", 1700, -850., 850., 1700, -850., 850.);
      }

      if (makeAllChamberPlots_) {
         // by chamber
         for(int station = 0; station < 4; ++station) {
            // DT wheels: -2 -> 2
            for(int wheel = 0; wheel < 5; ++wheel) {
               // DT sectors: 1 -> 14
               for(int sector = 0; sector < 14; ++sector)
               {
                  sprintf(name, "hDTChamberDx_%i_%i_%i", station+1, wheel-2, sector+1);
                  sprintf(title, "DT Chamber Delta X: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
                  hDTChamberDx[station][wheel][sector] = dbe_->book1D(name, title, 100, -100., 100.);

                  if (station < 3) {
                     sprintf(name, "hDTChamberDy_%i_%i_%i", station+1, wheel-2, sector+1);
                     sprintf(title, "DT Chamber Delta Y: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
                     hDTChamberDy[station][wheel][sector] = dbe_->book1D(name, title, 100, -150., 150.);
                  }

                  sprintf(name, "hDTChamberEdgeXWithSegment_%i_%i_%i", station+1, wheel-2, sector+1);
                  sprintf(title, "DT Chamber Edge X When There Is A Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
                  hDTChamberEdgeXWithSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -140., 30.);

                  sprintf(name, "hDTChamberEdgeXWithNoSegment_%i_%i_%i", station+1, wheel-2, sector+1);
                  sprintf(title, "DT Chamber Edge X When There Is No Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
                  hDTChamberEdgeXWithNoSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -140., 30.);

                  sprintf(name, "hDTChamberEdgeYWithSegment_%i_%i_%i", station+1, wheel-2, sector+1);
                  sprintf(title, "DT Chamber Edge Y When There Is A Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
                  hDTChamberEdgeYWithSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -140., 30.);

                  sprintf(name, "hDTChamberEdgeYWithNoSegment_%i_%i_%i", station+1, wheel-2, sector+1);
                  sprintf(title, "DT Chamber Edge Y When There Is No Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
                  hDTChamberEdgeYWithNoSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -140., 30.);
               }// sector
            }// wheel

            // CSC endcaps: 1 -> 2
            for(int endcap = 0; endcap < 2; ++endcap) {
               // CSC rings: 1 -> 4
               for(int ring = 0; ring < 4; ++ring) {
                  // CSC chambers: 1 -> 36
                  for(int chamber = 0; chamber < 36; ++chamber)
                  {
                     sprintf(name, "hCSCChamberDx_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
                     sprintf(title, "CSC Chamber Delta X: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
                     hCSCChamberDx[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -50., 50.);

                     sprintf(name, "hCSCChamberDy_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
                     sprintf(title, "CSC Chamber Delta Y: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
                     hCSCChamberDy[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -50., 50.);

                     sprintf(name, "hCSCChamberEdgeXWithSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
                     sprintf(title, "CSC Chamber Edge X When There Is A Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
                     hCSCChamberEdgeXWithSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -70., 20.);

                     sprintf(name, "hCSCChamberEdgeXWithNoSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
                     sprintf(title, "CSC Chamber Edge X When There Is No Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
                     hCSCChamberEdgeXWithNoSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -70., 20.);

                     sprintf(name, "hCSCChamberEdgeYWithSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
                     sprintf(title, "CSC Chamber Edge Y When There Is A Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
                     hCSCChamberEdgeYWithSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -70., 20.);

                     sprintf(name, "hCSCChamberEdgeYWithNoSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
                     sprintf(title, "CSC Chamber Edge Y When There Is No Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
                     hCSCChamberEdgeYWithNoSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -70., 20.);
                  }// chamber
               }// ring
            }// endcap
         }// station
      }
   }
}

void
MuonIdVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   iEvent.getByLabel(inputMuonCollection_, muonCollectionH_);
   iEvent.getByLabel(inputDTRecSegment4DCollection_, dtSegmentCollectionH_);
   iEvent.getByLabel(inputCSCSegmentCollection_, cscSegmentCollectionH_);
   iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);

   for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
         muon != muonCollectionH_->end(); ++muon)
   {
      // trackerMuon == 0; globalMuon == 1
      for (unsigned int i = 0; i < 2; i++) {
         if (i == 0 && (! useTrackerMuons_ || ! muon->isTrackerMuon())) continue;
         if (i == 1 && (! useGlobalMuons_ || ! muon->isGlobalMuon())) continue;

         hNumChambers[i]->Fill(muon->numberOfChambers());
         hNumMatches[i]->Fill(muon->numberOfMatches());
         if (! makeDQMPlots_) {
            hCaloCompat[i]->Fill(muon->caloCompatibility());
            hSegmentCompat[i]->Fill(muon->segmentCompatibility());
            if (make2DPlots_)
               hCaloSegmentCompat[i]->Fill(muon->caloCompatibility(), muon->segmentCompatibility());
            hGlobalMuonPromptTightBool[i]->Fill(muon->isGood(Muon::GlobalMuonPromptTight));
            hTMLastStationLooseBool[i]->Fill(muon->isGood(Muon::TMLastStationLoose));
            hTMLastStationTightBool[i]->Fill(muon->isGood(Muon::TMLastStationTight));
            hTM2DCompatibilityLooseBool[i]->Fill(muon->isGood(Muon::TM2DCompatibilityLoose));
            hTM2DCompatibilityTightBool[i]->Fill(muon->isGood(Muon::TM2DCompatibilityTight));
            hTMOneStationLooseBool[i]->Fill(muon->isGood(Muon::TMOneStationLoose));
            hTMOneStationTightBool[i]->Fill(muon->isGood(Muon::TMOneStationTight));
            hTMLastStationOptimizedLowPtLooseBool[i]->Fill(muon->isGood(Muon::TMLastStationOptimizedLowPtLoose));
            hTMLastStationOptimizedLowPtTightBool[i]->Fill(muon->isGood(Muon::TMLastStationOptimizedLowPtTight));

            if (makeEnergyPlots_) {
               //EM
               if (fabs(muon->eta()) > 1.479)
                  hEnergyEMEndcap[i]->Fill(muon->calEnergy().em);
               else
                  hEnergyEMBarrel[i]->Fill(muon->calEnergy().em);
               //HAD
               if (fabs(muon->eta()) > 1.4)
                  hEnergyHAEndcap[i]->Fill(muon->calEnergy().had);
               else
                  hEnergyHABarrel[i]->Fill(muon->calEnergy().had);
               //HO
               if (fabs(muon->eta()) < 1.26)
                  hEnergyHO[i]->Fill(muon->calEnergy().ho);
            }

            if (makeIsoPlots_) {
               hIso03sumPt[i]->Fill(muon->isolationR03().sumPt);
               hIso03emEt[i]->Fill(muon->isolationR03().emEt);
               hIso03hadEt[i]->Fill(muon->isolationR03().hadEt);
               hIso03hoEt[i]->Fill(muon->isolationR03().hoEt);
               hIso03nTracks[i]->Fill(muon->isolationR03().nTracks);
               hIso03nJets[i]->Fill(muon->isolationR03().nJets);
               hIso05sumPt[i]->Fill(muon->isolationR05().sumPt);
               hIso05emEt[i]->Fill(muon->isolationR05().emEt);
               hIso05hadEt[i]->Fill(muon->isolationR05().hadEt);
               hIso05hoEt[i]->Fill(muon->isolationR05().hoEt);
               hIso05nTracks[i]->Fill(muon->isolationR05().nTracks);
               hIso05nJets[i]->Fill(muon->isolationR05().nJets);
            }
         }
      }

      if (! useTrackerMuons_ || ! muon->isTrackerMuon()) continue;

      // by station
      for(int station = 0; station < 4; ++station)
      {
         hDTNumSegments[station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::DT, Muon::NoArbitration));
         hDTDx[station]->Fill(muon->dX(station+1, MuonSubdetId::DT));
         hDTPullx[station]->Fill(muon->pullX(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));

         if (station < 3) {
            hDTDy[station]->Fill(muon->dY(station+1, MuonSubdetId::DT));
            hDTPully[station]->Fill(muon->pullY(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
            if (! makeDQMPlots_) {
               hDTPullyPropErr[station]->Fill(muon->pullY(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
               hDTDdYdZ[station]->Fill(muon->dDyDz(station+1, MuonSubdetId::DT));
               hDTPulldYdZ[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
               hDTPulldYdZPropErr[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
            }
         }

         hCSCNumSegments[station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::CSC, Muon::NoArbitration));
         hCSCDx[station]->Fill(muon->dX(station+1, MuonSubdetId::CSC));
         hCSCPullx[station]->Fill(muon->pullX(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
         hCSCDy[station]->Fill(muon->dY(station+1, MuonSubdetId::CSC));
         hCSCPully[station]->Fill(muon->pullY(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));

         if (! makeDQMPlots_) {
            hDTPullxPropErr[station]->Fill(muon->pullX(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
            hDTDdXdZ[station]->Fill(muon->dDxDz(station+1, MuonSubdetId::DT));
            hDTPulldXdZ[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
            hDTPulldXdZPropErr[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));

            float distance = muon->trackDist(station+1, MuonSubdetId::DT);
            float error    = muon->trackDistErr(station+1, MuonSubdetId::DT);
            if (error == 0) error = 0.000001;

            if (muon->numberOfSegments(station+1, MuonSubdetId::DT, Muon::NoArbitration) > 0) {
               hDTDistWithSegment[station]->Fill(distance);
               hDTPullDistWithSegment[station]->Fill(distance/error);
            } else {
               hDTDistWithNoSegment[station]->Fill(distance);
               hDTPullDistWithNoSegment[station]->Fill(distance/error);
            }

            hCSCPullxPropErr[station]->Fill(muon->pullX(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            hCSCDdXdZ[station]->Fill(muon->dDxDz(station+1, MuonSubdetId::CSC));
            hCSCPulldXdZ[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
            hCSCPulldXdZPropErr[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            hCSCPullyPropErr[station]->Fill(muon->pullY(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            hCSCDdYdZ[station]->Fill(muon->dDyDz(station+1, MuonSubdetId::CSC));
            hCSCPulldYdZ[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
            hCSCPulldYdZPropErr[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));

            distance = muon->trackDist(station+1, MuonSubdetId::CSC);
            error    = muon->trackDistErr(station+1, MuonSubdetId::CSC);
            if (error == 0) error = 0.000001;

            if (muon->numberOfSegments(station+1, MuonSubdetId::CSC, Muon::NoArbitration) > 0) {
               hCSCDistWithSegment[station]->Fill(distance);
               hCSCPullDistWithSegment[station]->Fill(distance/error);
            } else {
               hCSCDistWithNoSegment[station]->Fill(distance);
               hCSCPullDistWithNoSegment[station]->Fill(distance/error);
            }
         }
      }

      if (makeAllChamberPlots_) {
         // by chamber
         for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
               chamberMatch != muon->matches().end(); ++chamberMatch)
         {
            int station = chamberMatch->station();

            if (chamberMatch->detector() == MuonSubdetId::DT) {
               DTChamberId dtId(chamberMatch->id.rawId());
               int wheel  = dtId.wheel();
               int sector = dtId.sector();

               if (chamberMatch->segmentMatches.empty()) {
                  hDTChamberEdgeXWithNoSegment[station-1][wheel+2][sector-1]->Fill(chamberMatch->edgeX);
                  hDTChamberEdgeYWithNoSegment[station-1][wheel+2][sector-1]->Fill(chamberMatch->edgeY);
               } else {
                  hDTChamberEdgeXWithSegment[station-1][wheel+2][sector-1]->Fill(chamberMatch->edgeX);
                  hDTChamberEdgeYWithSegment[station-1][wheel+2][sector-1]->Fill(chamberMatch->edgeY);

                  for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                        segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
                  {
                     if (segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR)) {
                        hDTChamberDx[station-1][wheel+2][sector-1]->Fill(chamberMatch->x-segmentMatch->x);
                        if (station < 4) hDTChamberDy[station-1][wheel+2][sector-1]->Fill(chamberMatch->y-segmentMatch->y);
                        break;
                     }
                  }// segmentMatch
               }

               continue;
            }

            if (chamberMatch->detector() == MuonSubdetId::CSC)  {
               CSCDetId cscId(chamberMatch->id.rawId());
               int endcap  = cscId.endcap();
               int ring    = cscId.ring();
               int chamber = cscId.chamber();

               if (chamberMatch->segmentMatches.empty()) {
                  hCSCChamberEdgeXWithNoSegment[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->edgeX);
                  hCSCChamberEdgeYWithNoSegment[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->edgeY);
               } else {
                  hCSCChamberEdgeXWithSegment[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->edgeX);
                  hCSCChamberEdgeYWithSegment[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->edgeY);

                  for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                        segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
                  {
                     if (segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR)) {
                        hCSCChamberDx[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->x-segmentMatch->x);
                        hCSCChamberDy[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->y-segmentMatch->y);
                        break;
                     }
                  }// segmentMatch
               }
            }
         }// chamberMatch
      }
   }// muon

   if (! useTrackerMuons_) return;

   for(DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH_->begin();
         segment != dtSegmentCollectionH_->end(); ++segment)
   {
      LocalPoint  segmentLocalPosition       = segment->localPosition();
      LocalVector segmentLocalDirection      = segment->localDirection();
      LocalError  segmentLocalPositionError  = segment->localPositionError();
      LocalError  segmentLocalDirectionError = segment->localDirectionError();
      const GeomDet* segmentGeomDet = geometry_->idToDet(segment->geographicalId());
      GlobalPoint segmentGlobalPosition = segmentGeomDet->toGlobal(segment->localPosition());
      bool segmentFound = false;
      bool segmentBestDrFound = false;

      for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
            muon != muonCollectionH_->end(); ++muon)
      {
         if (! muon->isMatchesValid())
            continue;

         for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
               chamberMatch != muon->matches().end(); ++chamberMatch) {
            for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                  segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
            {
               if (fabs(segmentMatch->x       - segmentLocalPosition.x()                           ) < 1E-6 &&
                   fabs(segmentMatch->y       - segmentLocalPosition.y()                           ) < 1E-6 &&
                   fabs(segmentMatch->dXdZ    - segmentLocalDirection.x()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->dYdZ    - segmentLocalDirection.y()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->xErr    - sqrt(segmentLocalPositionError.xx())               ) < 1E-6 &&
                   fabs(segmentMatch->yErr    - sqrt(segmentLocalPositionError.yy())               ) < 1E-6 &&
                   fabs(segmentMatch->dXdZErr - sqrt(segmentLocalDirectionError.xx())              ) < 1E-6 &&
                   fabs(segmentMatch->dYdZErr - sqrt(segmentLocalDirectionError.yy())              ) < 1E-6)
               {
                  segmentFound = true;
                  if (segmentMatch->isMask(reco::MuonSegmentMatch::BestInStationByDR)) segmentBestDrFound = true;
                  break;
               }
            }// segmentMatch
            if (segmentFound) break;
         }// chamberMatch
         if (segmentFound) break;
      }// muon

      if (segmentFound) {
         hSegmentIsAssociatedBool->Fill(1.);

         if (make2DPlots_) {
            hSegmentIsAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
            hSegmentIsAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());

            if (segmentBestDrFound) {
               hSegmentIsBestDrAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
               hSegmentIsBestDrAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
            }
         }
      } else {
         hSegmentIsAssociatedBool->Fill(0.);

         if (make2DPlots_) {
            hSegmentIsNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
            hSegmentIsNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
            hSegmentIsBestDrNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
            hSegmentIsBestDrNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
         }
      }
   }// dt segment

   for(CSCSegmentCollection::const_iterator segment = cscSegmentCollectionH_->begin();
         segment != cscSegmentCollectionH_->end(); ++segment)
   {
      LocalPoint  segmentLocalPosition       = segment->localPosition();
      LocalVector segmentLocalDirection      = segment->localDirection();
      LocalError  segmentLocalPositionError  = segment->localPositionError();
      LocalError  segmentLocalDirectionError = segment->localDirectionError();
      const GeomDet* segmentGeomDet = geometry_->idToDet(segment->geographicalId());
      GlobalPoint segmentGlobalPosition = segmentGeomDet->toGlobal(segment->localPosition());
      bool segmentFound = false;
      bool segmentBestDrFound = false;

      for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
            muon != muonCollectionH_->end(); ++muon)
      {
         if (! muon->isMatchesValid())
            continue;

         for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
               chamberMatch != muon->matches().end(); ++chamberMatch) {
            for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                  segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
            {
               if (fabs(segmentMatch->x       - segmentLocalPosition.x()                           ) < 1E-6 &&
                   fabs(segmentMatch->y       - segmentLocalPosition.y()                           ) < 1E-6 &&
                   fabs(segmentMatch->dXdZ    - segmentLocalDirection.x()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->dYdZ    - segmentLocalDirection.y()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->xErr    - sqrt(segmentLocalPositionError.xx())               ) < 1E-6 &&
                   fabs(segmentMatch->yErr    - sqrt(segmentLocalPositionError.yy())               ) < 1E-6 &&
                   fabs(segmentMatch->dXdZErr - sqrt(segmentLocalDirectionError.xx())              ) < 1E-6 &&
                   fabs(segmentMatch->dYdZErr - sqrt(segmentLocalDirectionError.yy())              ) < 1E-6)
               {
                  segmentFound = true;
                  if (segmentMatch->isMask(reco::MuonSegmentMatch::BestInStationByDR)) segmentBestDrFound = true;
                  break;
               }
            }// segmentMatch
            if (segmentFound) break;
         }// chamberMatch
         if (segmentFound) break;
      }// muon

      if (segmentFound) {
         hSegmentIsAssociatedBool->Fill(1.);

         if (make2DPlots_) {
            hSegmentIsAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
            hSegmentIsAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());

            if (segmentBestDrFound) {
               hSegmentIsBestDrAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
               hSegmentIsBestDrAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
            }
         }
      } else {
         hSegmentIsAssociatedBool->Fill(0.);

         if (make2DPlots_) {
            hSegmentIsNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
            hSegmentIsNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
            hSegmentIsBestDrNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
            hSegmentIsBestDrNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
         }
      }
   }// csc segment
}

void 
MuonIdVal::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdVal);

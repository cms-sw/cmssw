#include "Validation/MuonIdentification/interface/MuonIdVal.h"

MuonIdVal::MuonIdVal(const edm::ParameterSet& iConfig)
{
   inputMuonCollection_ = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
   inputDTRecSegment4DCollection_ = iConfig.getParameter<edm::InputTag>("inputDTRecSegment4DCollection");
   inputCSCSegmentCollection_ = iConfig.getParameter<edm::InputTag>("inputCSCSegmentCollection");
   useTrackerMuons_ = iConfig.getUntrackedParameter<bool>("useTrackerMuons");
   useGlobalMuons_ = iConfig.getUntrackedParameter<bool>("useGlobalMuons");
   makeEnergyPlots_ = iConfig.getUntrackedParameter<bool>("makeEnergyPlots");
   make2DPlots_ = iConfig.getUntrackedParameter<bool>("make2DPlots");
   makeAllChamberPlots_ = iConfig.getUntrackedParameter<bool>("makeAllChamberPlots");
   baseFolder_ = iConfig.getUntrackedParameter<std::string>("baseFolder");

   dbe_ = 0;
   dbe_ = edm::Service<DQMStore>().operator->();
}

MuonIdVal::~MuonIdVal() {}

void 
MuonIdVal::beginJob()
{
   char name[100], title[200];

   // trackerMuon == 0; globalMuon == 1
   for (unsigned int i = 0; i < 2; i++) {
      if ((i == 0 && ! useTrackerMuons_) || (i == 1 && ! useGlobalMuons_)) continue;
      if (i == 0) dbe_->setCurrentFolder(baseFolder_+"/TrackerMuons");
      if (i == 1) dbe_->setCurrentFolder(baseFolder_+"/GlobalMuons");

      if (makeEnergyPlots_) {
         hEnergyEMBarrel[i] = dbe_->book1D("hEnergyEMBarrel", "Energy in ECAL Barrel", 100, -0.5, 2.);
         hEnergyHABarrel[i] = dbe_->book1D("hEnergyHABarrel", "Energy in HCAL Barrel", 100, -4., 12.);
         hEnergyHO[i] = dbe_->book1D("hEnergyHO", "Energy HO", 100, -2., 5.);
         hEnergyEMEndcap[i] = dbe_->book1D("hEnergyEMEndcap", "Energy in ECAL Endcap", 100, -0.5, 2.);
         hEnergyHAEndcap[i] = dbe_->book1D("hEnergyHAEndcap", "Energy in HCAL Endcap", 100, -4., 12.);
      }

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

      // by station
      for(int station = 0; station < 4; ++station)
      {
         sprintf(name, "hDT%iPullxPropErr", station+1);
         sprintf(title, "DT Station %i Pull X w/ Propagation Error Only", station+1);
         hDTPullxPropErr[i][station] = dbe_->book1D(name, title, 100, -20., 20.);

         sprintf(name, "hDT%iPulldXdZPropErr", station+1);
         sprintf(title, "DT Station %i Pull DxDz w/ Propagation Error Only", station+1);
         hDTPulldXdZPropErr[i][station] = dbe_->book1D(name, title, 100, -20., 20.);

         if (station < 3) {
            sprintf(name, "hDT%iPullyPropErr", station+1);
            sprintf(title, "DT Station %i Pull Y w/ Propagation Error Only", station+1);
            hDTPullyPropErr[i][station] = dbe_->book1D(name, title, 100, -20., 20.);

            sprintf(name, "hDT%iPulldYdZPropErr", station+1);
            sprintf(title, "DT Station %i Pull DyDz w/ Propagation Error Only", station+1);
            hDTPulldYdZPropErr[i][station] = dbe_->book1D(name, title, 100, -20., 20.);
         }

         sprintf(name, "hDT%iDistWithSegment", station+1);
         sprintf(title, "DT Station %i Dist When There Is A Segment", station+1);
         hDTDistWithSegment[i][station] = dbe_->book1D(name, title, 100, -140., 30.);

         sprintf(name, "hDT%iDistWithNoSegment", station+1);
         sprintf(title, "DT Station %i Dist When There Is No Segment", station+1);
         hDTDistWithNoSegment[i][station] = dbe_->book1D(name, title, 100, -140., 30.);

         sprintf(name, "hDT%iPullDistWithSegment", station+1);
         sprintf(title, "DT Station %i Pull Dist When There Is A Segment", station+1);
         hDTPullDistWithSegment[i][station] = dbe_->book1D(name, title, 100, -140., 30.);

         sprintf(name, "hDT%iPullDistWithNoSegment", station+1);
         sprintf(title, "DT Station %i Pull Dist When There Is No Segment", station+1);
         hDTPullDistWithNoSegment[i][station] = dbe_->book1D(name, title, 100, -140., 30.);

         sprintf(name, "hCSC%iPullxPropErr", station+1);
         sprintf(title, "CSC Station %i Pull X w/ Propagation Error Only", station+1);
         hCSCPullxPropErr[i][station] = dbe_->book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iPulldXdZPropErr", station+1);
         sprintf(title, "CSC Station %i Pull DxDz w/ Propagation Error Only", station+1);
         hCSCPulldXdZPropErr[i][station] = dbe_->book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iPullyPropErr", station+1);
         sprintf(title, "CSC Station %i Pull Y w/ Propagation Error Only", station+1);
         hCSCPullyPropErr[i][station] = dbe_->book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iPulldYdZPropErr", station+1);
         sprintf(title, "CSC Station %i Pull DyDz w/ Propagation Error Only", station+1);
         hCSCPulldYdZPropErr[i][station] = dbe_->book1D(name, title, 100, -50., 50.);

         sprintf(name, "hCSC%iDistWithSegment", station+1);
         sprintf(title, "CSC Station %i Dist When There Is A Segment", station+1);
         hCSCDistWithSegment[i][station] = dbe_->book1D(name, title, 100, -70., 20.);

         sprintf(name, "hCSC%iDistWithNoSegment", station+1);
         sprintf(title, "CSC Station %i Dist When There Is No Segment", station+1);
         hCSCDistWithNoSegment[i][station] = dbe_->book1D(name, title, 100, -70., 20.);

         sprintf(name, "hCSC%iPullDistWithSegment", station+1);
         sprintf(title, "CSC Station %i Pull Dist When There Is A Segment", station+1);
         hCSCPullDistWithSegment[i][station] = dbe_->book1D(name, title, 100, -70., 20.);

         sprintf(name, "hCSC%iPullDistWithNoSegment", station+1);
         sprintf(title, "CSC Station %i Pull Dist When There Is No Segment", station+1);
         hCSCPullDistWithNoSegment[i][station] = dbe_->book1D(name, title, 100, -70., 20.);
      }// station
   }

   if (make2DPlots_) {
      dbe_->setCurrentFolder(baseFolder_);
      hSegmentIsAssociatedRZ = dbe_->book2D("hSegmentIsAssociatedRZ", "R-Z of Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
      hSegmentIsAssociatedXY = dbe_->book2D("hSegmentIsAssociatedXY", "X-Y of Associated Segments", 1700, -850., 850., 1700, -850., 850.);
      hSegmentIsNotAssociatedRZ = dbe_->book2D("hSegmentIsNotAssociatedRZ", "R-Z of Not Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
      hSegmentIsNotAssociatedXY = dbe_->book2D("hSegmentIsNotAssociatedXY", "X-Y of Not Associated Segments", 1700, -850., 850., 1700, -850., 850.);
      hSegmentIsBestDrAssociatedRZ = dbe_->book2D("hSegmentIsBestDrAssociatedRZ", "R-Z of Best in Station by #DeltaR Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
      hSegmentIsBestDrAssociatedXY = dbe_->book2D("hSegmentIsBestDrAssociatedXY", "X-Y of Best in Station by #DeltaR Associated Segments", 1700, -850., 850., 1700, -850., 850.);
      hSegmentIsBestDrNotAssociatedRZ = dbe_->book2D("hSegmentIsBestDrNotAssociatedRZ", "R-Z of Best in Station by #DeltaR Not Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
      hSegmentIsBestDrNotAssociatedXY = dbe_->book2D("hSegmentIsBestDrNotAssociatedXY", "X-Y of Best in Station by #DeltaR Not Associated Segments", 1700, -850., 850., 1700, -850., 850.);
   }

   if (useTrackerMuons_ && makeAllChamberPlots_) {
      dbe_->setCurrentFolder(baseFolder_+"/TrackerMuons");

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

         if (makeEnergyPlots_) {
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

         // by station
         for(int station = 0; station < 4; ++station)
         {
            Fill(hDTPullxPropErr[i][station], muon->pullX(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
            Fill(hDTPulldXdZPropErr[i][station], muon->pullDxDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));

            if (station < 3) {
               Fill(hDTPullyPropErr[i][station], muon->pullY(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
               Fill(hDTPulldYdZPropErr[i][station], muon->pullDyDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
            }

            float distance = muon->trackDist(station+1, MuonSubdetId::DT);
            float error    = muon->trackDistErr(station+1, MuonSubdetId::DT);
            if (error == 0) error = 0.000001;

            if (muon->numberOfSegments(station+1, MuonSubdetId::DT, Muon::NoArbitration) > 0) {
               Fill(hDTDistWithSegment[i][station], distance);
               Fill(hDTPullDistWithSegment[i][station], distance/error);
            } else {
               Fill(hDTDistWithNoSegment[i][station], distance);
               Fill(hDTPullDistWithNoSegment[i][station], distance/error);
            }

            Fill(hCSCPullxPropErr[i][station], muon->pullX(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            Fill(hCSCPulldXdZPropErr[i][station], muon->pullDxDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            Fill(hCSCPullyPropErr[i][station], muon->pullY(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            Fill(hCSCPulldYdZPropErr[i][station], muon->pullDyDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));

            distance = muon->trackDist(station+1, MuonSubdetId::CSC);
            error    = muon->trackDistErr(station+1, MuonSubdetId::CSC);
            if (error == 0) error = 0.000001;

            if (muon->numberOfSegments(station+1, MuonSubdetId::CSC, Muon::NoArbitration) > 0) {
               Fill(hCSCDistWithSegment[i][station], distance);
               Fill(hCSCPullDistWithSegment[i][station], distance/error);
            } else {
               Fill(hCSCDistWithNoSegment[i][station], distance);
               Fill(hCSCPullDistWithNoSegment[i][station], distance/error);
            }
         }// station
      }

      if (! useTrackerMuons_ || ! muon->isTrackerMuon()) continue;
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
                  Fill(hDTChamberEdgeXWithNoSegment[station-1][wheel+2][sector-1], chamberMatch->edgeX);
                  Fill(hDTChamberEdgeYWithNoSegment[station-1][wheel+2][sector-1], chamberMatch->edgeY);
               } else {
                  Fill(hDTChamberEdgeXWithSegment[station-1][wheel+2][sector-1], chamberMatch->edgeX);
                  Fill(hDTChamberEdgeYWithSegment[station-1][wheel+2][sector-1], chamberMatch->edgeY);

                  for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                        segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
                  {
                     if (segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR)) {
                        Fill(hDTChamberDx[station-1][wheel+2][sector-1], chamberMatch->x-segmentMatch->x);
                        if (station < 4) Fill(hDTChamberDy[station-1][wheel+2][sector-1], chamberMatch->y-segmentMatch->y);
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
                  Fill(hCSCChamberEdgeXWithNoSegment[endcap-1][station-1][ring-1][chamber-1], chamberMatch->edgeX);
                  Fill(hCSCChamberEdgeYWithNoSegment[endcap-1][station-1][ring-1][chamber-1], chamberMatch->edgeY);
               } else {
                  Fill(hCSCChamberEdgeXWithSegment[endcap-1][station-1][ring-1][chamber-1], chamberMatch->edgeX);
                  Fill(hCSCChamberEdgeYWithSegment[endcap-1][station-1][ring-1][chamber-1], chamberMatch->edgeY);

                  for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                        segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
                  {
                     if (segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR)) {
                        Fill(hCSCChamberDx[endcap-1][station-1][ring-1][chamber-1], chamberMatch->x-segmentMatch->x);
                        Fill(hCSCChamberDy[endcap-1][station-1][ring-1][chamber-1], chamberMatch->y-segmentMatch->y);
                        break;
                     }
                  }// segmentMatch
               }
            }
         }// chamberMatch
      }
   }// muon

   if (! make2DPlots_) return;

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
   }// csc segment
}

void 
MuonIdVal::endJob() {}

void MuonIdVal::Fill(MonitorElement* me, float f) {
   if (fabs(f) > 900000) return;
   //if (fabs(f) < 1E-8) return;
   me->Fill(f);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdVal);

#include "Validation/MuonIdentification/interface/MuonIdVal.h"
#include "DQMServices/Core/interface/DQMStore.h"

MuonIdVal::MuonIdVal(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   inputMuonCollection_ = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
   inputTrackCollection_ = iConfig.getParameter<edm::InputTag>("inputTrackCollection");
   inputDTRecSegment4DCollection_ = iConfig.getParameter<edm::InputTag>("inputDTRecSegment4DCollection");
   inputCSCSegmentCollection_ = iConfig.getParameter<edm::InputTag>("inputCSCSegmentCollection");
   outputFile_ = iConfig.getParameter<std::string>("outputFile");

   dbe_ = 0;
   dbe_ = edm::Service<DQMStore>().operator->();

   hNumChambers = dbe_->book1D("hNumChambers", "Number of Chambers", 20, 0., 20.);
   hNumMatches = dbe_->book1D("hNumMatches", "Number of Matches", 10, 0., 10.);
   hCaloCompat = dbe_->book1D("hCaloCompat", "Calo Compatibility", 100, 0., 1.);

   hEnergyEMBarrel = dbe_->book1D("hEnergyEMBarrel", "Energy in ECAL Barrel", 100, 0., 2.);
   hEnergyHABarrel = dbe_->book1D("hEnergyHABarrel", "Energy in HCAL Barrel", 100, -1., 10.);
   hEnergyHO = dbe_->book1D("hEnergyHO", "Energy HO", 100, -1., 5.);
   hEnergyEMEndcap = dbe_->book1D("hEnergyEMEndcap", "Energy in ECAL Endcap", 100, 0., 2.);
   hEnergyHAEndcap = dbe_->book1D("hEnergyHAEndcap", "Energy in HCAL Endcap", 100, -1., 10.);

   hIso03sumPt = dbe_->book1D("hIso03sumPt", "Sum Pt in Cone of 0.3", 100, 0., 10.);
   hIso03emEt = dbe_->book1D("hIso03emEt", "Em Et in Cone of 0.3", 100, 0., 10.);
   hIso03hadEt = dbe_->book1D("hIso03hadEt", "Had Et in Cone of 0.3", 100, 0., 10.);
   hIso03hoEt = dbe_->book1D("hIso03hoEt", "HO Et in Cone of 0.3", 100, 0., 10.);
   hIso03nTracks = dbe_->book1D("hIso03nTracks", "Number of Tracks in Cone of 0.3", 10, 0., 10.);
   hIso03nJets = dbe_->book1D("hIso03nJets", "Number of Jets in Cone of 0.3", 10, 0., 10.);
   hIso05sumPt = dbe_->book1D("hIso05sumPt", "Sum Pt in Cone of 0.5", 100, 0., 10.);
   hIso05emEt = dbe_->book1D("hIso05emEt", "Em Et in Cone of 0.5", 100, 0., 10.);
   hIso05hadEt = dbe_->book1D("hIso05hadEt", "Had Et in Cone of 0.5", 100, 0., 10.);
   hIso05hoEt = dbe_->book1D("hIso05hoEt", "HO Et in Cone of 0.5", 100, 0., 10.);
   hIso05nTracks = dbe_->book1D("hIso05nTracks", "Number of Tracks in Cone of 0.5", 10, 0., 10.);
   hIso05nJets = dbe_->book1D("hIso05nJets", "Number of Jets in Cone of 0.5", 10, 0., 10.);

   char name[100], title[200];

   // by station
   for(int station = 0; station < 4; ++station)
   {
      sprintf(name, "hDT%iNumSegments", station+1);
      sprintf(title, "DT Station %i Number of Segments (No Arbitration)", station+1);
      hDTNumSegments[station] = dbe_->book1D(name, title, 10, 0., 10.);

      sprintf(name, "hDT%iDx", station+1);
      sprintf(title, "DT Station %i Delta X", station+1);
      hDTDx[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hDT%iPullx", station+1);
      sprintf(title, "DT Station %i Pull X", station+1);
      hDTPullx[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hDT%iPullxPropErr", station+1);
      sprintf(title, "DT Station %i Pull X w/ Propagation Error Only", station+1);
      hDTPullxPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hDT%iDdXdZ", station+1);
      sprintf(title, "DT Station %i Delta DxDz", station+1);
      hDTDdXdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hDT%iPulldXdZ", station+1);
      sprintf(title, "DT Station %i Pull DxDz", station+1);
      hDTPulldXdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hDT%iPulldXdZPropErr", station+1);
      sprintf(title, "DT Station %i Pull DxDz w/ Propagation Error Only", station+1);
      hDTPulldXdZPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);

      if (station < 3) {
         sprintf(name, "hDT%iDy", station+1);
         sprintf(title, "DT Station %i Delta Y", station+1);
         hDTDy[station] = dbe_->book1D(name, title, 100, -10., 10.);

         sprintf(name, "hDT%iPully", station+1);
         sprintf(title, "DT Station %i Pull Y", station+1);
         hDTPully[station] = dbe_->book1D(name, title, 100, -10., 10.);

         sprintf(name, "hDT%iPullyPropErr", station+1);
         sprintf(title, "DT Station %i Pull Y w/ Propagation Error Only", station+1);
         hDTPullyPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);

         sprintf(name, "hDT%iDdYdZ", station+1);
         sprintf(title, "DT Station %i Delta DyDz", station+1);
         hDTDdYdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

         sprintf(name, "hDT%iPulldYdZ", station+1);
         sprintf(title, "DT Station %i Pull DyDz", station+1);
         hDTPulldYdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

         sprintf(name, "hDT%iPulldYdZPropErr", station+1);
         sprintf(title, "DT Station %i Pull DyDz w/ Propagation Error Only", station+1);
         hDTPulldYdZPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);
      }

      sprintf(name, "hDT%iDistWithSegment", station+1);
      sprintf(title, "DT Station %i Dist When There Is A Segment", station+1);
      hDTDistWithSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);

      sprintf(name, "hDT%iDistWithNoSegment", station+1);
      sprintf(title, "DT Station %i Dist When There Is No Segment", station+1);
      hDTDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);

      sprintf(name, "hDT%iPullDistWithSegment", station+1);
      sprintf(title, "DT Station %i Pull Dist When There Is A Segment", station+1);
      hDTPullDistWithSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);

      sprintf(name, "hDT%iPullDistWithNoSegment", station+1);
      sprintf(title, "DT Station %i Pull Dist When There Is No Segment", station+1);
      hDTPullDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);

      sprintf(name, "hCSC%iNumSegments", station+1);
      sprintf(title, "CSC Station %i Number of Segments (No Arbitration)", station+1);
      hCSCNumSegments[station] = dbe_->book1D(name, title, 10, 0., 10.);

      sprintf(name, "hCSC%iDx", station+1);
      sprintf(title, "CSC Station %i Delta X", station+1);
      hCSCDx[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPullx", station+1);
      sprintf(title, "CSC Station %i Pull X", station+1);
      hCSCPullx[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPullxPropErr", station+1);
      sprintf(title, "CSC Station %i Pull X w/ Propagation Error Only", station+1);
      hCSCPullxPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iDdXdZ", station+1);
      sprintf(title, "CSC Station %i Delta DxDz", station+1);
      hCSCDdXdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPulldXdZ", station+1);
      sprintf(title, "CSC Station %i Pull DxDz", station+1);
      hCSCPulldXdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPulldXdZPropErr", station+1);
      sprintf(title, "CSC Station %i Pull DxDz w/ Propagation Error Only", station+1);
      hCSCPulldXdZPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iDy", station+1);
      sprintf(title, "CSC Station %i Delta Y", station+1);
      hCSCDy[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPully", station+1);
      sprintf(title, "CSC Station %i Pull Y", station+1);
      hCSCPully[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPullyPropErr", station+1);
      sprintf(title, "CSC Station %i Pull Y w/ Propagation Error Only", station+1);
      hCSCPullyPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iDdYdZ", station+1);
      sprintf(title, "CSC Station %i Delta DyDz", station+1);
      hCSCDdYdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPulldYdZ", station+1);
      sprintf(title, "CSC Station %i Pull DyDz", station+1);
      hCSCPulldYdZ[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iPulldYdZPropErr", station+1);
      sprintf(title, "CSC Station %i Pull DyDz w/ Propagation Error Only", station+1);
      hCSCPulldYdZPropErr[station] = dbe_->book1D(name, title, 100, -10., 10.);

      sprintf(name, "hCSC%iDistWithSegment", station+1);
      sprintf(title, "CSC Station %i Dist When There Is A Segment", station+1);
      hCSCDistWithSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);

      sprintf(name, "hCSC%iDistWithNoSegment", station+1);
      sprintf(title, "CSC Station %i Dist When There Is No Segment", station+1);
      hCSCDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);

      sprintf(name, "hCSC%iPullDistWithSegment", station+1);
      sprintf(title, "CSC Station %i Pull Dist When There Is A Segment", station+1);
      hCSCPullDistWithSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);

      sprintf(name, "hCSC%iPullDistWithNoSegment", station+1);
      sprintf(title, "CSC Station %i Pull Dist When There Is No Segment", station+1);
      hCSCPullDistWithNoSegment[station] = dbe_->book1D(name, title, 100, -100., 30.);
   }// station

   hSegmentIsAssociatedBool = dbe_->book1D("hSegmentIsAssociatedBool", "Segment Is Associated Boolean", 2, 0., 2.);
   hSegmentIsAssociatedRZ = dbe_->book2D("hSegmentIsAssociatedRZ", "R-Z of Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
   hSegmentIsAssociatedXY = dbe_->book2D("hSegmentIsAssociatedXY", "R-#phi of Associated Segments", 1700, -850., 850., 1700, -850., 850.);
   hSegmentIsNotAssociatedRZ = dbe_->book2D("hSegmentIsNotAssociatedRZ", "R-Z of Not Associated Segments", 2140, -1070., 1070., 850, 0., 850.);
   hSegmentIsNotAssociatedXY = dbe_->book2D("hSegmentIsNotAssociatedXY", "R-#phi of Not Associated Segments", 1700, -850., 850., 1700, -850., 850.);

   // by chamber
   for(int station = 0; station < 4; ++station) {
      // DT wheels: -2 -> 2
      for(int wheel = 0; wheel < 5; ++wheel) {
         // DT sectors: 1 -> 14
         for(int sector = 0; sector < 14; ++sector)
         {
            sprintf(name, "hDTChamberDx_%i_%i_%i", station+1, wheel-2, sector+1);
            sprintf(title, "DT Chamber Delta X: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
            hDTChamberDx[station][wheel][sector] = dbe_->book1D(name, title, 100, -10., 10.);

            if (station < 3) {
               sprintf(name, "hDTChamberDy_%i_%i_%i", station+1, wheel-2, sector+1);
               sprintf(title, "DT Chamber Delta Y: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
               hDTChamberDy[station][wheel][sector] = dbe_->book1D(name, title, 100, -10., 10.);
            }

            sprintf(name, "hDTChamberEdgeXWithSegment_%i_%i_%i", station+1, wheel-2, sector+1);
            sprintf(title, "DT Chamber Edge X When There Is A Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
            hDTChamberEdgeXWithSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -100., 30.);

            sprintf(name, "hDTChamberEdgeXWithNoSegment_%i_%i_%i", station+1, wheel-2, sector+1);
            sprintf(title, "DT Chamber Edge X When There Is No Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
            hDTChamberEdgeXWithNoSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -100., 30.);

            sprintf(name, "hDTChamberEdgeYWithSegment_%i_%i_%i", station+1, wheel-2, sector+1);
            sprintf(title, "DT Chamber Edge Y When There Is A Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
            hDTChamberEdgeYWithSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -100., 30.);

            sprintf(name, "hDTChamberEdgeYWithNoSegment_%i_%i_%i", station+1, wheel-2, sector+1);
            sprintf(title, "DT Chamber Edge Y When There Is No Segment: Station %i Wheel %i Sector %i", station+1, wheel-2, sector+1);
            hDTChamberEdgeYWithNoSegment[station][wheel][sector] = dbe_->book1D(name, title, 100, -100., 30.);
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
               hCSCChamberDx[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -10., 10.);

               sprintf(name, "hCSCChamberDy_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
               sprintf(title, "CSC Chamber Delta Y: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
               hCSCChamberDy[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -10., 10.);

               sprintf(name, "hCSCChamberEdgeXWithSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
               sprintf(title, "CSC Chamber Edge X When There Is A Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
               hCSCChamberEdgeXWithSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -100., 30.);

               sprintf(name, "hCSCChamberEdgeXWithNoSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
               sprintf(title, "CSC Chamber Edge X When There Is No Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
               hCSCChamberEdgeXWithNoSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -100., 30.);

               sprintf(name, "hCSCChamberEdgeYWithSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
               sprintf(title, "CSC Chamber Edge Y When There Is A Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
               hCSCChamberEdgeYWithSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -100., 30.);

               sprintf(name, "hCSCChamberEdgeYWithNoSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1);
               sprintf(title, "CSC Chamber Edge Y When There Is No Segment: Endcap %i Station %i Ring %i Chamber %i", endcap+1, station+1, ring+1, chamber+1);
               hCSCChamberEdgeYWithNoSegment[endcap][station][ring][chamber] = dbe_->book1D(name, title, 100, -100., 30.);
            }// chamber
         }// ring
      }// endcap
   }// station
}

MuonIdVal::~MuonIdVal()
{
}

void
MuonIdVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   iEvent.getByLabel(inputMuonCollection_, muonCollectionH_);
   iEvent.getByLabel(inputTrackCollection_, trackCollectionH_);
   iEvent.getByLabel(inputDTRecSegment4DCollection_, dtSegmentCollectionH_);
   iEvent.getByLabel(inputCSCSegmentCollection_, cscSegmentCollectionH_);
   iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);

   bool yesPt = false;
   bool yesP  = false;
   for(TrackCollection::const_iterator track = trackCollectionH_->begin();
         track != trackCollectionH_->end(); ++track)
   {
      if (track->pt() >= 1.5) yesPt = true;
      if (track->p()  >= 3.) yesP = true;
      if (yesPt && yesP) break;
   }
   if (! (yesPt && yesP)) return;

   for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
         muon != muonCollectionH_->end(); ++muon)
   {
      if (! muon->isTrackerMuon()) continue;

      hNumChambers->Fill(muon->numberOfChambers());
      hNumMatches->Fill(muon->numberOfMatches());

      if (muon->isCaloCompatibilityValid())
         hCaloCompat->Fill(muon->caloCompatibility());

      if (muon->isEnergyValid()) {
         //EM
         if (fabs(muon->eta()) > 1.479)
            hEnergyEMEndcap->Fill(muon->calEnergy().em);
         else
            hEnergyEMBarrel->Fill(muon->calEnergy().em);
         //HAD
         if (fabs(muon->eta()) > 1.4)
            hEnergyHAEndcap->Fill(muon->calEnergy().had);
         else
            hEnergyHABarrel->Fill(muon->calEnergy().had);
         //HO
         if (fabs(muon->eta()) < 1.26)
            hEnergyHO->Fill(muon->calEnergy().ho);
      }

      if (muon->isIsolationValid()) {
         hIso03sumPt->Fill(muon->isolationR03().sumPt);
         hIso03emEt->Fill(muon->isolationR03().emEt);
         hIso03hadEt->Fill(muon->isolationR03().hadEt);
         hIso03hoEt->Fill(muon->isolationR03().hoEt);
         hIso03nTracks->Fill(muon->isolationR03().nTracks);
         hIso03nJets->Fill(muon->isolationR03().nJets);
         hIso05sumPt->Fill(muon->isolationR05().sumPt);
         hIso05emEt->Fill(muon->isolationR05().emEt);
         hIso05hadEt->Fill(muon->isolationR05().hadEt);
         hIso05hoEt->Fill(muon->isolationR05().hoEt);
         hIso05nTracks->Fill(muon->isolationR05().nTracks);
         hIso05nJets->Fill(muon->isolationR05().nJets);
      }

      if (muon->isMatchesValid()) {
         for(int station = 0; station < 4; ++station)
         {
            hDTNumSegments[station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::DT, Muon::NoArbitration));
            hDTDx[station]->Fill(muon->dX(station+1, MuonSubdetId::DT));
            hDTPullx[station]->Fill(muon->pullX(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
            hDTPullxPropErr[station]->Fill(muon->pullX(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
            hDTDdXdZ[station]->Fill(muon->dDxDz(station+1, MuonSubdetId::DT));
            hDTPulldXdZ[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
            hDTPulldXdZPropErr[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));

            if (station < 3) {
               hDTDy[station]->Fill(muon->dY(station+1, MuonSubdetId::DT));
               hDTPully[station]->Fill(muon->pullY(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
               hDTPullyPropErr[station]->Fill(muon->pullY(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
               hDTDdYdZ[station]->Fill(muon->dDyDz(station+1, MuonSubdetId::DT));
               hDTPulldYdZ[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
               hDTPulldYdZPropErr[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, false));
            }

            hCSCNumSegments[station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::CSC, Muon::NoArbitration));
            hCSCDx[station]->Fill(muon->dX(station+1, MuonSubdetId::CSC));
            hCSCPullx[station]->Fill(muon->pullX(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
            hCSCPullxPropErr[station]->Fill(muon->pullX(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            hCSCDdXdZ[station]->Fill(muon->dDxDz(station+1, MuonSubdetId::CSC));
            hCSCPulldXdZ[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
            hCSCPulldXdZPropErr[station]->Fill(muon->pullDxDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            hCSCDy[station]->Fill(muon->dY(station+1, MuonSubdetId::CSC));
            hCSCPully[station]->Fill(muon->pullY(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
            hCSCPullyPropErr[station]->Fill(muon->pullY(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
            hCSCDdYdZ[station]->Fill(muon->dDyDz(station+1, MuonSubdetId::CSC));
            hCSCPulldYdZ[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
            hCSCPulldYdZPropErr[station]->Fill(muon->pullDyDz(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, false));
         }

         std::vector<std::pair<const MuonChamberMatch*,const MuonChamberMatch*> > me11_pairs;
         std::map<double,const MuonChamberMatch*> me11_map;

         for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
               chamberMatch != muon->matches().end(); ++chamberMatch)
         {
            double distance = chamberMatch->dist();
            double error    = chamberMatch->distErr();
            int station     = chamberMatch->station();

            if (chamberMatch->detector() == MuonSubdetId::DT) {
               DTChamberId dtId(chamberMatch->id.rawId());
               int wheel  = dtId.wheel();
               int sector = dtId.sector();

               if (chamberMatch->segmentMatches.empty()) {
                  hDTDistWithNoSegment[station-1]->Fill(distance);
                  hDTPullDistWithNoSegment[station-1]->Fill(distance/error);
                  hDTChamberEdgeXWithNoSegment[station-1][wheel+2][sector-1]->Fill(chamberMatch->edgeX);
                  hDTChamberEdgeYWithNoSegment[station-1][wheel+2][sector-1]->Fill(chamberMatch->edgeY);
               } else {
                  hDTDistWithSegment[station-1]->Fill(distance);
                  hDTPullDistWithSegment[station-1]->Fill(distance/error);
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

               // for logical OR of ME11 segments below
               if (station == 1 && (ring == 1 || ring == 4)) {
                  if (me11_map.find(distance) != me11_map.end()) {
                     std::pair<const MuonChamberMatch*,const MuonChamberMatch*> pair((me11_map.find(distance))->second, &(*chamberMatch));
                     me11_pairs.push_back(pair);
                  } else
                     me11_map.insert(std::make_pair(distance, &(*chamberMatch)));
               }

               if (chamberMatch->segmentMatches.empty()) {
                  if(! (station == 1 && (ring == 1 || ring == 4))) {
                     hCSCDistWithNoSegment[station-1]->Fill(distance);
                     hCSCPullDistWithNoSegment[station-1]->Fill(distance/error);
                  }
                  hCSCChamberEdgeXWithNoSegment[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->edgeX);
                  hCSCChamberEdgeYWithNoSegment[endcap-1][station-1][ring-1][chamber-1]->Fill(chamberMatch->edgeY);
               } else {
                  if(! (station == 1 && (ring == 1 || ring == 4))) {
                     hCSCDistWithSegment[station-1]->Fill(distance);
                     hCSCPullDistWithSegment[station-1]->Fill(distance/error);
                  }
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

         // logical OR of ME11 segments as promised above
         for(std::vector<std::pair<const MuonChamberMatch*,const MuonChamberMatch*> >::const_iterator pair = me11_pairs.begin();
               pair != me11_pairs.end(); ++pair)
         {
            CSCDetId cscId1(pair->first->id.rawId());
            CSCDetId cscId2(pair->second->id.rawId());

            // sanity checks
            if (cscId1.endcap() != cscId2.endcap())
               LogWarning("ME11 Logical OR") << "CSCDetIds are not in the same endcap!"; 
            if (cscId1.station() != cscId2.station())
               LogWarning("ME11 Logical OR") << "CSCDetIds are not in the same station!";
            if (cscId1.ring() == cscId2.ring())
               LogWarning("ME11 Logical OR") << "CSCDetIds are in the same ring!";
            if (cscId1.chamber() != cscId2.chamber())
               LogWarning("ME11 Logical OR") << "CSCDetIds are not in the same chamber!";
            LogVerbatim("ME11 Logical OR") << "cscId1=" << cscId1.rawId() << "station=" << cscId1.station() << " ring=" << cscId1.ring();
            LogVerbatim("ME11 Logical OR") << "cscId2=" << cscId2.rawId() << "station=" << cscId2.station() << " ring=" << cscId2.ring();
            LogVerbatim("ME11 Logical OR") << "cscId1.#segmentMatches=" << pair->first->segmentMatches.size();
            LogVerbatim("ME11 Logical OR") << "cscId2.#segmentMatches=" << pair->second->segmentMatches.size();

            double distance = pair->first->dist();
            double error    = pair->first->distErr();

            if (pair->first->segmentMatches.empty() && pair->second->segmentMatches.empty()) {
               hCSCDistWithNoSegment[0]->Fill(distance);
               hCSCPullDistWithNoSegment[0]->Fill(distance/error);
               LogVerbatim("ME11 Logical OR") << "filling me11 segment true";
            } else {
               hCSCDistWithSegment[0]->Fill(distance);
               hCSCPullDistWithSegment[0]->Fill(distance/error);
               LogVerbatim("ME11 Logical OR") << "filling me11 segment false";
            }
         }// pairs
      }// matches valid
   }// muon

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

      for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
            muon != muonCollectionH_->end(); ++muon)
      {
         if (! (muon->isTrackerMuon() && muon->isMatchesValid()))
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
                  break;
               }
            }// segmentMatch
            if (segmentFound) break;
         }// chamberMatch
         if (segmentFound) break;
      }// muon

      if (segmentFound) {
         hSegmentIsAssociatedBool->Fill(1.);
         hSegmentIsAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
         hSegmentIsAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
      } else {
         hSegmentIsAssociatedBool->Fill(0.);
         hSegmentIsNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
         hSegmentIsNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
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

      for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
            muon != muonCollectionH_->end(); ++muon)
      {
         if (! (muon->isTrackerMuon() && muon->isMatchesValid()))
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
                  break;
               }
            }// segmentMatch
            if (segmentFound) break;
         }// chamberMatch
         if (segmentFound) break;
      }// muon

      if (segmentFound) {
         hSegmentIsAssociatedBool->Fill(1.);
         hSegmentIsAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
         hSegmentIsAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
      } else {
         hSegmentIsAssociatedBool->Fill(0.);
         hSegmentIsNotAssociatedRZ->Fill(segmentGlobalPosition.z(), segmentGlobalPosition.perp());
         hSegmentIsNotAssociatedXY->Fill(segmentGlobalPosition.x(), segmentGlobalPosition.y());
      }
   }// csc segment
}

void 
MuonIdVal::beginJob(const edm::EventSetup&)
{
}

void 
MuonIdVal::endJob() {
   if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdVal);

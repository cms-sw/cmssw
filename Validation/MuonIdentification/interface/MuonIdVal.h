// -*- C++ -*-
//
// Package:    Validation/MuonIdentification
// Class:      MuonIdVal
// 
/*

 Description:  Makes and fills lots of histograms using the various reco::Muon
               methods.


*/
//
// Original Author:  Jacob Ribnik
//         Created:  Wed Apr 18 13:48:08 CDT 2007
// $Id: MuonIdVal.h,v 1.5 2008/10/30 19:17:43 jribnik Exp $
//
//

#ifndef Validation_MuonIdentification_MuonIdVal_h
#define Validation_MuonIdentification_MuonIdVal_h

// system include files
#include <string>

// user include files
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

class MuonIdVal : public edm::EDAnalyzer {
   public:
      explicit MuonIdVal(const edm::ParameterSet&);
      ~MuonIdVal();

   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      virtual void Fill(MonitorElement*, float);

      DQMStore* dbe_;

      // ----------member data ---------------------------
      edm::InputTag inputMuonCollection_;
      edm::InputTag inputDTRecSegment4DCollection_;
      edm::InputTag inputCSCSegmentCollection_;
      bool useTrackerMuons_;
      bool useGlobalMuons_;
      bool makeEnergyPlots_;
      bool make2DPlots_;
      bool makeAllChamberPlots_;
      std::string baseFolder_;

      edm::Handle<reco::MuonCollection> muonCollectionH_;
      edm::Handle<DTRecSegment4DCollection> dtSegmentCollectionH_;
      edm::Handle<CSCSegmentCollection> cscSegmentCollectionH_;
      edm::ESHandle<GlobalTrackingGeometry> geometry_;

      // trackerMuon == 0; globalMuon == 1
      // energy deposits
      MonitorElement* hEnergyEMBarrel[2];
      MonitorElement* hEnergyHABarrel[2];
      MonitorElement* hEnergyHO[2];
      MonitorElement* hEnergyEMEndcap[2];
      MonitorElement* hEnergyHAEndcap[2];

      // muonid
      MonitorElement* hCaloCompat[2];
      MonitorElement* hSegmentCompat[2];
      MonitorElement* hCaloSegmentCompat[2];
      MonitorElement* hGlobalMuonPromptTightBool[2];
      MonitorElement* hTMLastStationLooseBool[2];
      MonitorElement* hTMLastStationTightBool[2];
      MonitorElement* hTM2DCompatibilityLooseBool[2];
      MonitorElement* hTM2DCompatibilityTightBool[2];
      MonitorElement* hTMOneStationLooseBool[2];
      MonitorElement* hTMOneStationTightBool[2];
      MonitorElement* hTMLastStationOptimizedLowPtLooseBool[2];
      MonitorElement* hTMLastStationOptimizedLowPtTightBool[2];

      // by station
      MonitorElement* hDTPullxPropErr[2][4];
      MonitorElement* hDTPulldXdZPropErr[2][4];
      MonitorElement* hDTPullyPropErr[2][3];
      MonitorElement* hDTPulldYdZPropErr[2][3];
      MonitorElement* hDTDistWithSegment[2][4];
      MonitorElement* hDTDistWithNoSegment[2][4];
      MonitorElement* hDTPullDistWithSegment[2][4];
      MonitorElement* hDTPullDistWithNoSegment[2][4];
      MonitorElement* hCSCPullxPropErr[2][4];
      MonitorElement* hCSCPulldXdZPropErr[2][4];
      MonitorElement* hCSCPullyPropErr[2][4];
      MonitorElement* hCSCPulldYdZPropErr[2][4];
      MonitorElement* hCSCDistWithSegment[2][4];
      MonitorElement* hCSCDistWithNoSegment[2][4];
      MonitorElement* hCSCPullDistWithSegment[2][4];
      MonitorElement* hCSCPullDistWithNoSegment[2][4];

      // by chamber, trackerMuons only
      // DT:  [station][wheel][sector]
      // CSC: [endcap][station][ring][chamber]
      MonitorElement* hDTChamberDx[4][5][14];
      MonitorElement* hDTChamberDy[3][5][14];
      MonitorElement* hDTChamberEdgeXWithSegment[4][5][14];
      MonitorElement* hDTChamberEdgeXWithNoSegment[4][5][14];
      MonitorElement* hDTChamberEdgeYWithSegment[4][5][14];
      MonitorElement* hDTChamberEdgeYWithNoSegment[4][5][14];
      MonitorElement* hCSCChamberDx[2][4][4][36];
      MonitorElement* hCSCChamberDy[2][4][4][36];
      MonitorElement* hCSCChamberEdgeXWithSegment[2][4][4][36];
      MonitorElement* hCSCChamberEdgeXWithNoSegment[2][4][4][36];
      MonitorElement* hCSCChamberEdgeYWithSegment[2][4][4][36];
      MonitorElement* hCSCChamberEdgeYWithNoSegment[2][4][4][36];

      // segment matching "efficiency"
      MonitorElement* hSegmentIsAssociatedRZ;
      MonitorElement* hSegmentIsAssociatedXY;
      MonitorElement* hSegmentIsNotAssociatedRZ;
      MonitorElement* hSegmentIsNotAssociatedXY;
      MonitorElement* hSegmentIsBestDrAssociatedRZ;
      MonitorElement* hSegmentIsBestDrAssociatedXY;
      MonitorElement* hSegmentIsBestDrNotAssociatedRZ;
      MonitorElement* hSegmentIsBestDrNotAssociatedXY;
};

#endif

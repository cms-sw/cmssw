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
// $Id: MuonIdVal.h,v 1.1 2007/11/29 18:34:09 jribnik Exp $
//
//

// system include files
#include <map>
#include <string>
#include <utility>
#include <vector>

// user include files
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DQMServices/Core/interface/DQMStore.h"

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
#include "DQMServices/Core/interface/MonitorElement.h"

class MuonIdVal : public edm::EDAnalyzer {
   public:
      explicit MuonIdVal(const edm::ParameterSet&);
      ~MuonIdVal();

   private:
      virtual void beginJob(const edm::EventSetup&);
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      DQMStore* dbe_;

      // ----------member data ---------------------------
      edm::InputTag inputMuonCollection_;
      edm::InputTag inputTrackCollection_;
      edm::InputTag inputDTRecSegment4DCollection_;
      edm::InputTag inputCSCSegmentCollection_;
      std::string outputFile_;

      edm::Handle<reco::MuonCollection> muonCollectionH_;
      edm::Handle<reco::TrackCollection> trackCollectionH_;
      edm::Handle<DTRecSegment4DCollection> dtSegmentCollectionH_;
      edm::Handle<CSCSegmentCollection> cscSegmentCollectionH_;
      edm::ESHandle<GlobalTrackingGeometry> geometry_;

      MonitorElement* hNumChambers;
      MonitorElement* hNumMatches;
      MonitorElement* hCaloCompat;

      MonitorElement* hEnergyEMBarrel;
      MonitorElement* hEnergyHABarrel;
      MonitorElement* hEnergyHO;
      MonitorElement* hEnergyEMEndcap;
      MonitorElement* hEnergyHAEndcap;

      MonitorElement* hIso03sumPt;
      MonitorElement* hIso03emEt;
      MonitorElement* hIso03hadEt;
      MonitorElement* hIso03hoEt;
      MonitorElement* hIso03nTracks;
      MonitorElement* hIso03nJets;
      MonitorElement* hIso05sumPt;
      MonitorElement* hIso05emEt;
      MonitorElement* hIso05hadEt;
      MonitorElement* hIso05hoEt;
      MonitorElement* hIso05nTracks;
      MonitorElement* hIso05nJets;

      // by station
      MonitorElement* hDTNumSegments[4];
      MonitorElement* hDTDx[4];
      MonitorElement* hDTPullx[4];
      MonitorElement* hDTPullxPropErr[4];
      MonitorElement* hDTDdXdZ[4];
      MonitorElement* hDTPulldXdZ[4];
      MonitorElement* hDTPulldXdZPropErr[4];
      MonitorElement* hDTDy[3];
      MonitorElement* hDTPully[3];
      MonitorElement* hDTPullyPropErr[3];
      MonitorElement* hDTDdYdZ[3];
      MonitorElement* hDTPulldYdZ[3];
      MonitorElement* hDTPulldYdZPropErr[3];
      MonitorElement* hDTDistWithSegment[4];
      MonitorElement* hDTDistWithNoSegment[4];
      MonitorElement* hDTPullDistWithSegment[4];
      MonitorElement* hDTPullDistWithNoSegment[4];
      MonitorElement* hCSCNumSegments[4];
      MonitorElement* hCSCDx[4];
      MonitorElement* hCSCPullx[4];
      MonitorElement* hCSCPullxPropErr[4];
      MonitorElement* hCSCDdXdZ[4];
      MonitorElement* hCSCPulldXdZ[4];
      MonitorElement* hCSCPulldXdZPropErr[4];
      MonitorElement* hCSCDy[4];
      MonitorElement* hCSCPully[4];
      MonitorElement* hCSCPullyPropErr[4];
      MonitorElement* hCSCDdYdZ[4];
      MonitorElement* hCSCPulldYdZ[4];
      MonitorElement* hCSCPulldYdZPropErr[4];
      MonitorElement* hCSCDistWithSegment[4];
      MonitorElement* hCSCDistWithNoSegment[4];
      MonitorElement* hCSCPullDistWithSegment[4];
      MonitorElement* hCSCPullDistWithNoSegment[4];

      MonitorElement* hSegmentIsAssociatedBool;
      MonitorElement* hSegmentIsAssociatedRZ;
      MonitorElement* hSegmentIsAssociatedXY;
      MonitorElement* hSegmentIsNotAssociatedRZ;
      MonitorElement* hSegmentIsNotAssociatedXY;

      // by chamber
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
};

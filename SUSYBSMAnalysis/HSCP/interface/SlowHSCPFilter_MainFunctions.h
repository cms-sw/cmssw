// -*- H -*-
//
// Original Author:  Loic QUERTENMONT
//         Created:  Fri Dec  7 10:40:51 CET 2007
// $Id: SlowHSCPFilter_MainFunctions.h,v 1.1 2007/12/11 12:37:48 querten Exp $
//
//

#ifndef SUSYBSMANALYSIS_SLOWHSCPFilter_MAINFUNCTIONS_H
#define SUSYBSMANALYSIS_SLOWHSCPFilter_MAINFUNCTIONS_H


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"



#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"


using namespace edm;


double SlowHSCPFilter_DeltaR               (double phi1, double eta1, double phi2, double eta2);
int    SlowHSCPFilter_ClosestL1Muon        (double phi, double eta, double dRMax, const l1extra::L1MuonParticleCollection L1_Muons);
int    SlowHSCPFilter_ComesFromWhichHSCP   (unsigned int TrackId, std::vector<SimTrack*> HSCPs, std::vector<SimTrack>  TrackColl, std::vector<SimVertex> VertexColl);
void   GetTrueL1MuonsAndTime               (const edm::Event&, const edm::EventSetup&,  int* recoL1Muon, double* MinDt);  



#endif


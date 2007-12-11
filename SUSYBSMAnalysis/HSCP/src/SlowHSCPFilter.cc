// -*- C++ -*-
//
// Package:    SlowHSCPFilter
// Class:      SlowHSCPFilter
// 
/**\class SlowHSCPFilter SlowHSCPFilter.cc SUSYBSMAnalysis/HSCP/src/SlowHSCPFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic QUERTENMONT
//         Created:  Fri Dec  7 10:40:51 CET 2007
// $Id: SlowHSCPFilter.cc,v 1.2 2007/12/11 12:20:27 querten Exp $
//
//


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


#include "SUSYBSMAnalysis/HSCP/src/SlowHSCPFilter_MainFunctions.h"


using namespace edm;


class SlowHSCPFilter : public edm::EDFilter {
   public:
      explicit SlowHSCPFilter(const edm::ParameterSet&);
      ~SlowHSCPFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


   public:
      double       DeltaTMax;
      std::string  Txt_File;
   
      unsigned int NEvents;
      unsigned int NGoodEvents;
};


SlowHSCPFilter::SlowHSCPFilter(const edm::ParameterSet& iConfig)
{
      NEvents     = 0;
      NGoodEvents = 0;
      DeltaTMax   = iConfig.getUntrackedParameter<double >("DeltaTMax");
      Txt_File    = iConfig.getUntrackedParameter<std::string >("Txt_output");
}


SlowHSCPFilter::~SlowHSCPFilter()
{
      if(Txt_File.size()>3){
          FILE* f = fopen(Txt_File.c_str(), "w");
          fprintf(f,"Accepted Event = %6.2f%%\n",(100.0*NGoodEvents)/NEvents);
          fclose(f);
      }

      printf("\n\nAccepted Event = %6.2f%%\n",(100.0*NGoodEvents)/NEvents);
}

bool
SlowHSCPFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
int    recoL1Muon[2];
double MinDt[2];

GetTrueL1MuonsAndTime(iEvent, iSetup, recoL1Muon, MinDt);

 bool Good = false;
 if(recoL1Muon[0]<0  && recoL1Muon[1]<0      ) Good = true;
 if(recoL1Muon[0]>=0 && MinDt[0] < DeltaTMax ) Good = true;
 if(recoL1Muon[1]>=0 && MinDt[1] < DeltaTMax ) Good = true;

 
 NEvents++;
 if(Good){
    NGoodEvents++;
     printf("SlowHSCPFilter : This event has been accepted\n");
    return true;
 }

 printf("SlowHSCPFilter : This event has been rejected\n");
 return false;
}

void 
SlowHSCPFilter::beginJob(const edm::EventSetup&)
{
}

void 
SlowHSCPFilter::endJob() {
}

DEFINE_FWK_MODULE(SlowHSCPFilter);

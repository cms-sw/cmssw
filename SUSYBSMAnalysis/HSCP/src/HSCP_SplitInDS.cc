// -*- C++ -*-
//
// Package:    HSCP_SplitInDS
// Class:      HSCP_SplitInDS
// 
/**\class HSCP_SplitInDS HSCP_SplitInDS.cc SUSYBSMAnalysis/HSCP/src/HSCP_SplitInDS.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic QUERTENMONT
//         Created:  Fri Dec  7 10:40:51 CET 2007
// $Id: HSCP_SplitInDS.cc,v 1.4 2007/12/11 17:04:02 querten Exp $
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


#include "SUSYBSMAnalysis/HSCP/interface/SlowHSCPFilter_MainFunctions.h"
#include "SUSYBSMAnalysis/HSCP/interface/HSCP_Trigger_MainFunctions.h"

using namespace edm;


class HSCP_SplitInDS : public edm::EDFilter {
   public:
      explicit HSCP_SplitInDS(const edm::ParameterSet&);
      ~HSCP_SplitInDS();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


   public:
      double       DeltaTMax;
      std::string  Txt_File;
   
      unsigned int NEvents;
      unsigned int NGoodEvents;


      bool         Init;
      unsigned int L1_NPath;
      unsigned int HLT_NPath;


      std::vector<unsigned int> DataSet_HLTPaths;


};


HSCP_SplitInDS::HSCP_SplitInDS(const edm::ParameterSet& iConfig)
{
      NEvents     = 0;
      NGoodEvents = 0;
      DeltaTMax   = iConfig.getUntrackedParameter<double >("DeltaTMax");
      Txt_File    = iConfig.getUntrackedParameter<std::string >("Txt_output");

      Init        = false;
      L1_NPath    = 0;
      HLT_NPath   = 0;

      DataSet_HLTPaths = iConfig.getUntrackedParameter<std::vector<unsigned int> >("DataSet_HLTPaths");
}


HSCP_SplitInDS::~HSCP_SplitInDS()
{
      if(Txt_File.size()>3){
          FILE* f = fopen(Txt_File.c_str(), "w");
          fprintf(f,"Read     Events = %i\n"     ,NEvents);
          fprintf(f,"Accepted Events = %i\n"     ,NGoodEvents);
          fprintf(f,"Accepted Ratio  = %05.2f%%\n",(100.0*NGoodEvents)/NEvents);
          fclose(f);
      }

      printf("Read     Events = %i\n"     ,NEvents);
      printf("Accepted Events = %i\n"     ,NGoodEvents);
      printf("Accepted Ratio  = %05.2f%%\n",(100.0*NGoodEvents)/NEvents);
}

bool
HSCP_SplitInDS::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   int    recoL1Muon[2];
   double MinDt[2];

   GetTrueL1MuonsAndTime(iEvent, iSetup, recoL1Muon, MinDt);


   Handle<l1extra::L1MuonParticleCollection> L1_Muons_h;
   iEvent.getByLabel("l1extraParticles", L1_Muons_h);
   const l1extra::L1MuonParticleCollection L1_Muons = *L1_Muons_h.product();

   Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
   try {iEvent.getByLabel("l1extraParticleMap",L1GTRR);} catch (...) {;}

   //Initialisation
   if(!Init){
        L1_NPath           = L1GTRR->decisionWord().size();
   }

   bool* L1_Trigger_Bits_tmp = new bool[L1_NPath];
   for(unsigned int i=0;i<L1_NPath;i++){
        if(i == l1extra::L1ParticleMap::kSingleMu7){
                L1_Trigger_Bits_tmp[i] = HSCP_Trigger_L1MuonAbovePtThreshold(L1_Muons,7, recoL1Muon, MinDt, DeltaTMax);                 
        }else if(i == l1extra::L1ParticleMap::kDoubleMu3){
                L1_Trigger_Bits_tmp[i] = HSCP_Trigger_L1TwoMuonAbovePtThreshold(L1_Muons,3, recoL1Muon, MinDt, DeltaTMax);      
        }else{
                L1_Trigger_Bits_tmp[i] = L1GTRR->decisionWord()[i];
        }
   }

   Handle<TriggerResults> HLTR;
   InputTag tag("TriggerResults","","HLT");
   try {iEvent.getByLabel(tag,HLTR);} catch (...) {;}

   //Initialisation
   if(!Init){
        HLT_NPath           = HLTR->size();
        Init = true;
   }

  bool* HLT_Trigger_Bits_tmp = new bool[HLT_NPath];
  for(unsigned int i=0;i<HLT_NPath;i++){
       HLT_Trigger_Bits_tmp[i] = HLTR->accept(i) && HSCP_Trigger_IsL1ConditionTrue(i,L1_Trigger_Bits_tmp);
   }


   NEvents++;
   for(unsigned int i=0; i<DataSet_HLTPaths.size();i++)
   {
	if(DataSet_HLTPaths[i]<0         ) continue;
        if(DataSet_HLTPaths[i]>=HLT_NPath) continue;

	if(HLT_Trigger_Bits_tmp[DataSet_HLTPaths[i]]){
		NGoodEvents++;
		return true;
	}	
   }
   return false;

}

void 
HSCP_SplitInDS::beginJob(const edm::EventSetup&)
{
}

void 
HSCP_SplitInDS::endJob() {
}

DEFINE_FWK_MODULE(HSCP_SplitInDS);

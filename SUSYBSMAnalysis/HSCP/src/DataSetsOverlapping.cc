// -*- C++ -*-
//
// Package:    HSCP_DataSetsOverlapping
// Class:      HSCP_DataSetsOverlapping
// 
/**\class HSCP_DataSetsOverlapping HSCP_DataSetsOverlapping.cc SUSYBSMAnalysis/HSCP/src/HSCP_DataSetsOverlapping.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic QUERTENMONT
//         Created:  Fri Dec  7 10:40:51 CET 2007
// $Id: HSCP_DataSetsOverlapping.cc,v 1.1 2007/12/16 08:35:38 querten Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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


class HSCP_DataSetsOverlapping : public edm::EDAnalyzer {
   public:
      explicit HSCP_DataSetsOverlapping(const edm::ParameterSet&);
      ~HSCP_DataSetsOverlapping();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze (const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


   public:
      double       DeltaTMax;
      std::string  Txt_File;
   
      unsigned int NEvents;
      unsigned int NEventsDS1;
      unsigned int NEventsDS2;
      unsigned int NEventsDS1and2;
      unsigned int NEventsDS1or2;

      bool         Init;
      unsigned int L1_NPath;
      unsigned int HLT_NPath;

      std::vector<unsigned int> DataSet1_Paths;
      std::vector<unsigned int> DataSet2_Paths;
};


HSCP_DataSetsOverlapping::HSCP_DataSetsOverlapping(const edm::ParameterSet& iConfig)
{
      NEvents         = 0;
      NEventsDS1      = 0;
      NEventsDS2      = 0;
      NEventsDS1and2  = 0;
      NEventsDS1or2   = 0;


      DeltaTMax   = iConfig.getUntrackedParameter<double >("DeltaTMax");
      Txt_File    = iConfig.getUntrackedParameter<std::string >("Txt_output");

      Init        = false;
      L1_NPath    = 0;
      HLT_NPath   = 0;

      DataSet1_Paths = iConfig.getUntrackedParameter<std::vector<unsigned int> >("DataSet1_Paths");
      DataSet2_Paths = iConfig.getUntrackedParameter<std::vector<unsigned int> >("DataSet2_Paths");
}


HSCP_DataSetsOverlapping::~HSCP_DataSetsOverlapping()
{
      if(Txt_File.size()>3){
          FILE* f = fopen(Txt_File.c_str(), "w");
          fprintf(f,"#Events                = %5i                     \n", NEvents);
          fprintf(f,"#Events in DS1         = %5i  --> ratio = %6.2f%%\n", NEventsDS1     , (100.0*NEventsDS1    ) / NEvents);
          fprintf(f,"#Events in DS2         = %5i  --> ratio = %6.2f%%\n", NEventsDS2     , (100.0*NEventsDS2    ) / NEvents);
          fprintf(f,"#Events in DS1 or  DS2 = %5i  --> ratio = %6.2f%%\n", NEventsDS1or2  , (100.0*NEventsDS1or2 ) / NEvents);
          fprintf(f,"#Events in DS1 and DS2 = %5i  --> ratio = %6.2f%%\n", NEventsDS1and2 , (100.0*NEventsDS1and2) / NEvents);
          fclose(f);
      }

      printf("#Events                = %5i                     \n", NEvents);
      printf("#Events in DS1         = %5i  --> ratio = %6.2f%%\n", NEventsDS1     , (100.0*NEventsDS1    ) / NEvents);
      printf("#Events in DS2         = %5i  --> ratio = %6.2f%%\n", NEventsDS2     , (100.0*NEventsDS2    ) / NEvents);
      printf("#Events in DS1 or  DS2 = %5i  --> ratio = %6.2f%%\n", NEventsDS1or2  , (100.0*NEventsDS1or2 ) / NEvents);
      printf("#Events in DS1 and DS2 = %5i  --> ratio = %6.2f%%\n", NEventsDS1and2 , (100.0*NEventsDS1and2) / NEvents);
}

void
HSCP_DataSetsOverlapping::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   int    recoL1Muon[2];
   double MinDt[2];

   if(DeltaTMax>=0){
      GetTrueL1MuonsAndTime(iEvent, iSetup, recoL1Muon, MinDt);
   }

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
        if(i == l1extra::L1ParticleMap::kSingleMu7 && DeltaTMax>=0){
                L1_Trigger_Bits_tmp[i] = HSCP_Trigger_L1MuonAbovePtThreshold(L1_Muons,7, recoL1Muon, MinDt, DeltaTMax);                 
        }else if(i == l1extra::L1ParticleMap::kDoubleMu3 && DeltaTMax>=0){
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

   bool DS1 = false;
   bool DS2 = false;

   for(unsigned int i=0; i<DataSet1_Paths.size();i++)
   {
	if(DataSet1_Paths[i]<0         ) continue;
        if(DataSet1_Paths[i]>=HLT_NPath) continue;

	if(HLT_Trigger_Bits_tmp[DataSet1_Paths[i]]){DS1=true;}	
   }

   for(unsigned int i=0; i<DataSet2_Paths.size();i++)
   { 
        if(DataSet2_Paths[i]<0         ) continue;
        if(DataSet2_Paths[i]>=HLT_NPath) continue;
 
        if(HLT_Trigger_Bits_tmp[DataSet2_Paths[i]]){DS2=true;}   
   } 

                NEvents++;
   if(DS1)      NEventsDS1++;
   if(DS2)      NEventsDS2++;
   if(DS1&&DS2) NEventsDS1and2++;
   if(DS1||DS2) NEventsDS1or2++;
}

void 
HSCP_DataSetsOverlapping::beginJob(const edm::EventSetup&)
{
}

void 
HSCP_DataSetsOverlapping::endJob() {
}

DEFINE_FWK_MODULE(HSCP_DataSetsOverlapping);

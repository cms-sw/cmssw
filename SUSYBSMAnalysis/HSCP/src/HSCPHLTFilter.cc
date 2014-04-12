#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"


//
// class declaration
//

using namespace edm;

class HSCPHLTFilter : public edm::EDFilter {
   public:
      explicit HSCPHLTFilter(const edm::ParameterSet&);
      ~HSCPHLTFilter();

   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;
      bool isDuplicate(unsigned int Run, unsigned int Event);

      bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut, int NObjectAboveThreshold, bool averageThreshold);

      std::string TriggerProcess;
      edm::EDGetTokenT< trigger::TriggerEvent > trEvToken;
      std::map<std::string, bool > DuplicateMap;

      unsigned int CountEvent;
      unsigned int MaxPrint;
      bool         RemoveDuplicates;
      int          MuonTrigger1Mask;
      int          MuonTrigger2Mask;
      int          PFMetTriggerMask;
      int          CaloMetTriggerMask;
};


/////////////////////////////////////////////////////////////////////////////////////
HSCPHLTFilter::HSCPHLTFilter(const edm::ParameterSet& iConfig)
{
   RemoveDuplicates      = iConfig.getParameter<bool>                ("RemoveDuplicates");

   TriggerProcess        = iConfig.getParameter<std::string>         ("TriggerProcess");
   trEvToken = consumes< trigger::TriggerEvent >( edm::InputTag( "hltTriggerSummaryAOD" ) );
   MuonTrigger1Mask       = iConfig.getParameter<int>                 ("MuonTrigger1Mask");
   PFMetTriggerMask        = iConfig.getParameter<int>                 ("PFMetTriggerMask");

   CountEvent = 0;
   MaxPrint = 10000;
}

/////////////////////////////////////////////////////////////////////////////////////
HSCPHLTFilter::~HSCPHLTFilter(){
}

/////////////////////////////////////////////////////////////////////////////////////
void HSCPHLTFilter::beginJob() {
}

/////////////////////////////////////////////////////////////////////////////////////
void HSCPHLTFilter::endJob(){
}


bool HSCPHLTFilter::isDuplicate(unsigned int Run, unsigned int Event){
   char tmp[255];sprintf(tmp,"%i_%i",Run,Event);
   std::map<std::string, bool >::iterator it = DuplicateMap.find(std::string(tmp));
   if(it==DuplicateMap.end()){
      DuplicateMap[std::string(tmp)] = true;
      return false;
   }
   return true;
}


/////////////////////////////////////////////////////////////////////////////////////
bool HSCPHLTFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::TriggerResultsByName tr = iEvent.triggerResultsByName(TriggerProcess);
   if(!tr.isValid()){    printf("NoValidTrigger\n");  }


   if(RemoveDuplicates && isDuplicate(iEvent.eventAuxiliary().run(),iEvent.eventAuxiliary().event()))return false;


//   for(unsigned int i=0;i<tr.size();i++){
//      printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
//   }fflush(stdout);


   edm::Handle< trigger::TriggerEvent > trEvHandle;
   iEvent.getByToken(trEvToken, trEvHandle);
   trigger::TriggerEvent trEv = *trEvHandle;

   CountEvent++;
   //if(CountEvent<MaxPrint)printf("------------------------\n");


   unsigned int TrIndex_Unknown     = tr.size();


   bool MuonTrigger1 = false;
   bool PFMetTrigger  = false;


   // HLT TRIGGER BASED ON 1 MUON!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v5")) {
     if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v5"))){MuonTrigger1 = true;}
   }else{
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v4")) {
     if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v4"))){MuonTrigger1 = true;}
   }else{
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v1")) {
     if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v1"))){MuonTrigger1 = true;}
   }else{
     if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v8")){
       if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L2QualL3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
     }else{
       if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v7")){
	 if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L2QualL3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
       }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v6")){
	   if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L2QualL3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
         }else{
	   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v5")){
	     if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
	   }else{
	     if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v4")){
	       if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
	     }else{
	       if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v3")){
		 if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
	       }else{
		 if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v2")){
		   if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
		 }else{
		   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v1")){
		     if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","",TriggerProcess), 40, 2.1, 1, false)){MuonTrigger1 = true;}
		   }else{
		     printf("HSCPHLTFilter --> HLT_Mu30_v1  not found\n");
		     for(unsigned int i=0;i<tr.size();i++){
		       printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
		     }fflush(stdout);
		     exit(0);
		   }
		 }
	       }
	     }
           }
	 }
       }
     }
   }
   }
   }

   // HLT TRIGGER BASED ON PF MET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v17")){
     if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v17"))){PFMetTrigger = true;}
   }else{
   if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v16")){
     if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v16"))){PFMetTrigger = true;}
   }else{
   if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v12")){
      if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v12"))){PFMetTrigger = true;}
   }else{
       if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v11")){
          if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v11"))){PFMetTrigger = true;}
       }else{
           if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v10")){
              if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v10"))){PFMetTrigger = true;}
           }else{
              if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v9")){
                 if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v9"))){PFMetTrigger = true;}
              }else{
                  if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v8")){
                     if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v8"))){PFMetTrigger = true;}
                  }else{
                     if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v7")){
                        if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v7"))){PFMetTrigger = true;}
                     }else{
                        if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v6")){
                           if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v6"))){PFMetTrigger = true;}
                        }else{
                           if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v5")){
                              if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v5"))){PFMetTrigger = true;}
                           }else{
                              if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v4")){
                                 if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v4"))){PFMetTrigger = true;}
                              }else{
                                 if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v3")){
                                    if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v3"))){PFMetTrigger = true;}
                                 }else{
                                    if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v2")){
                                       if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v2"))){PFMetTrigger = true;}
                                    }else{
                                       if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v1")){
                                          if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v1"))){PFMetTrigger = true;}
                                       }else{
                                          printf("HSCPHLTFilter --> HLT_PFMHT150_v2 or v1  not found\n");
                                          for(unsigned int i=0;i<tr.size();i++){
                                             printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
                                          }fflush(stdout);
                                          exit(0);
                                      }
                                   }
                                }
                             }
                          }
                       }
                    }
                 }
               }
           }
       }
   }
   }
   }

   //printf("Bits = %1i %1i %1i X Mask = %+2i %+2i %+2i -->",MuonTrigger,CaloMetTrigger,CaloMetTrigger,MuonTriggerMask,CaloMetTriggerMask,CaloMetTriggerMask);

   if(MuonTrigger1Mask==0)MuonTrigger1=false;
   if(PFMetTriggerMask ==0)PFMetTrigger =false;

   //Allow option of requiring that one of the triggers did NOT fire to remove duplicated events
   if(MuonTrigger1Mask<0 && MuonTrigger1) return false;
   if(PFMetTriggerMask<0 && PFMetTrigger) return false;


   bool d =  (MuonTrigger1 | PFMetTrigger);
   /* printf("%i\n",d);*/return d;

}

   bool HSCPHLTFilter::IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut, int NObjectAboveThreshold, bool averageThreshold)
{
   unsigned int filterIndex = trEv.filterIndex(InputPath);
   //if(filterIndex<trEv.sizeFilters())printf("SELECTED INDEX =%i --> %s    XXX   %s\n",filterIndex,trEv.filterTag(filterIndex).label().c_str(), trEv.filterTag(filterIndex).process().c_str());

   if (filterIndex<trEv.sizeFilters()){
      const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
      const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
      const int nI(VIDS.size());
      const int nK(KEYS.size());
      assert(nI==nK);
      const int n(std::max(nI,nK));
      const trigger::TriggerObjectCollection& TOC(trEv.getObjects());


      if(!averageThreshold){
         int NObjectAboveThresholdObserved = 0;
         for (int i=0; i!=n; ++i) {
	   if(TOC[KEYS[i]].pt()> NewThreshold && fabs(TOC[KEYS[i]].eta())<etaCut) NObjectAboveThresholdObserved++;
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if(NObjectAboveThresholdObserved>=NObjectAboveThreshold)return true;

      }else{
         std::vector<double> ObjPt;

         for (int i=0; i!=n; ++i) {
            ObjPt.push_back(TOC[KEYS[i]].pt());
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if((int)(ObjPt.size())<NObjectAboveThreshold)return false;
         std::sort(ObjPt.begin(), ObjPt.end());

         double Average = 0;
         for(int i=0; i<NObjectAboveThreshold;i++){
            Average+= ObjPt[ObjPt.size()-1-i];
         }Average/=NObjectAboveThreshold;
         //cout << "AVERAGE = " << Average << endl;

         if(Average>NewThreshold)return true;
      }
   }
   return false;
}










DEFINE_FWK_MODULE(HSCPHLTFilter);





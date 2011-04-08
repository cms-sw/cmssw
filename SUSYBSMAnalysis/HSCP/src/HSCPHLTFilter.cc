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
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      bool isDuplicate(unsigned int Run, unsigned int Event);

      bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, int NObjectAboveThreshold, bool averageThreshold);

      std::string TriggerProcess;

      std::map<std::string, bool > DuplicateMap;

      unsigned int CountEvent;
      unsigned int MaxPrint;
      bool         RemoveDuplicates;
      int          MuonTriggerMask;
      int          METTriggerMask;
      int          JetTriggerMask;
};


/////////////////////////////////////////////////////////////////////////////////////
HSCPHLTFilter::HSCPHLTFilter(const edm::ParameterSet& iConfig)
{
   RemoveDuplicates      = iConfig.getParameter<bool>                ("RemoveDuplicates");

   TriggerProcess        = iConfig.getParameter<std::string>         ("TriggerProcess");
   MuonTriggerMask       = iConfig.getParameter<int>                 ("MuonTriggerMask");
   METTriggerMask        = iConfig.getParameter<int>                 ("METTriggerMask");
   JetTriggerMask        = iConfig.getParameter<int>                 ("JetTriggerMask");

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
   iEvent.getByLabel("hltTriggerSummaryAOD", trEvHandle);
   trigger::TriggerEvent trEv = *trEvHandle;

   CountEvent++;
   //if(CountEvent<MaxPrint)printf("------------------------\n");


   unsigned int TrIndex_Unknown     = tr.size();


   bool MuonTrigger = false;
   bool METTrigger  = false;
   bool JetTrigger  = false;


   // HLT TRIGGER BASED ON 1 MUON!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu24_v1")){
          //if(CountEvent<MaxPrint)printf("Use HLT_Mu24_v1\n");
          if(tr.accept(tr.triggerIndex("HLT_Mu24_v1"))){MuonTrigger = true;}
   }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu15_v1")){
          if(IncreasedTreshold(trEv, InputTag("hltSingleMu15L3Filtered11","",TriggerProcess), 24, 1, false)){MuonTrigger = true;}
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu11")){
               //if(CountEvent<MaxPrint)printf("Use HLT_Mu11 Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hltSingleMu11L3Filtered11","",TriggerProcess), 24, 1, false)){MuonTrigger = true;}
         }else{
             if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu9")){
               //if(CountEvent<MaxPrint)printf("Use HLT_Mu9 Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hltSingleMu9L3Filtered9","",TriggerProcess), 24, 1, false)){MuonTrigger = true;}
            }else{
                printf("HSCPHLTFilter --> BUG with HLT_Mu9\n");
            }
         }
      }
   }


   // HLT TRIGGER BASED ON 2 MUONS!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_DoubleMu6_v1")){
             if(tr.accept(tr.triggerIndex("HLT_DoubleMu6_v1"))){MuonTrigger = true;}
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_DoubleMu3_v2")){
          if(IncreasedTreshold(trEv, InputTag("hltDoubleMu3L3Filtered11","",TriggerProcess), 6, 2, false)){MuonTrigger = true;}
       }else{
          if(TrIndex_Unknown != tr.triggerIndex("HLT_DoubleMu3")){
             //if(CountEvent<MaxPrint)printf("Use HLT_DoubleMu3\n");
          if(IncreasedTreshold(trEv, InputTag("hltDoubleMu3L3Filtered11","",TriggerProcess), 6, 2, false)){MuonTrigger = true;}
          }else{
             printf("HSCPHLTFilter --> BUG with HLT_DoubleMu3\n");
         }
      }
   }

   // HLT TRIGGER BASED ON MET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v1")){
      if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v1"))){METTrigger = true;}
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_MET100_v3")){
         //if(CountEvent<MaxPrint)printf("Use HLT_MET100_v3\n");
         if(tr.accept(tr.triggerIndex("HLT_MET100_v3"))){METTrigger = true;}
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_MET100_v2")){
            //if(CountEvent<MaxPrint)printf("Use HLT_MET100_v2\n");
            if(tr.accept(tr.triggerIndex("HLT_MET100_v2"))){METTrigger = true;}
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_MET100")){
               //if(CountEvent<MaxPrint)printf("Use HLT_MET100\n");
               if(tr.accept(tr.triggerIndex("HLT_MET100"))){METTrigger = true;}
            }else{
               printf("HSCPHLTFilter --> BUG with HLT_MET100\n");
            }
         }
      }
   }

   // HLT TRIGGER BASED ON 1 JET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet370_v1")){
      if(tr.accept(tr.triggerIndex("HLT_Jet370_v1"))){JetTrigger = true;}
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet140U_v3")){
         //if(CountEvent<MaxPrint)printf("Use HLT_Jet140U_v3\n");
         if(IncreasedTreshold(trEv, InputTag("hlt1jet140U","",TriggerProcess), 370, 1, false)){JetTrigger = true;}
      }else{ 
         if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet140U_v1")){
             //if(CountEvent<MaxPrint)printf("Use HLT_Jet140U_v1\n");
            if(IncreasedTreshold(trEv, InputTag("hlt1jet140U","",TriggerProcess), 370, 1, false)){JetTrigger = true;}
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet100U")){
                //if(CountEvent<MaxPrint)printf("Use HLT_Jet100U Rescaled\n");
                if(IncreasedTreshold(trEv, InputTag("hlt1jet100U","",TriggerProcess), 370, 1, false)){JetTrigger = true;}
            }else{
               if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet70U")){   
                  //if(CountEvent<MaxPrint)printf("Use HLT_Jet70U Rescaled\n");
                  if(IncreasedTreshold(trEv, InputTag("hlt1jet70U","",TriggerProcess), 370, 1, false)){JetTrigger = true;}
               }else{
                  if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet50U")){
                     //if(CountEvent<MaxPrint)printf("Use HLT_Jet50U Rescaled\n");
                     if(IncreasedTreshold(trEv, InputTag("hlt1jet50U","",TriggerProcess), 370, 1, false)){JetTrigger = true;}
                  }else{
                     printf("HSCPHLTFilter --> BUG with HLT_Jet50U\n");
                  }
               }
            }
         }
      }
   }
/*
   // HLT TRIGGER BASED ON 2 JETS!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_DiJetAve140U_v3")){
      //if(CountEvent<MaxPrint)printf("Use HLT_DiJetAve140U_v3\n");
      if(tr.accept(tr.triggerIndex("HLT_DiJetAve140U_v3") )){JetTrigger = true;}
   }else{  
      if(TrIndex_Unknown != tr.triggerIndex("OpenHLT_DiJetAve70")){
         //if(CountEvent<MaxPrint)printf("Use OpenHLT_DiJetAve70\n");
         if(IncreasedTreshold(trEv, InputTag("hltDiJetAve70U","",TriggerProcess), 140, 2, true)){JetTrigger = true;}
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("OpenHLT_DiJetAve50U")){
               //if(CountEvent<MaxPrint)printf("Use OpenHLT_DiJetAve50 Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hltDiJetAve50U","",TriggerProcess), 140, 2, true)){JetTrigger = true;}
               if(IncreasedTreshold(trEv, InputTag("openhltDiJetAve50U","",TriggerProcess), 140, 2, true)){JetTrigger = true;}
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_DiJetAve50U")){
               //if(CountEvent<MaxPrint)printf("Use HLT_DiJetAve50 Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hltDiJetAve50U","",TriggerProcess), 140, 2, true)){JetTrigger = true;}
            }else{
               if(TrIndex_Unknown != tr.triggerIndex("HLT_DiJetAve30U_8E29")){ 
                  //if(CountEvent<MaxPrint)printf("Use HLT_DiJetAve30U_8E29 Rescaled\n");
                  if(IncreasedTreshold(trEv, InputTag("hltDiJetAve30U8E29","",TriggerProcess), 140, 2, true)){JetTrigger = true;}
               }else{
                   printf("HSCPHLTFilter --> BUG with HLT_DiJetAve30U_8E29\n");
               }
            }
         }
      }
   }


   // HLT TRIGGER BASED ON 4 JETS!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_QuadJet25U_v3")){
      //if(CountEvent<MaxPrint)printf("Rescale HLT_QuadJet25U_v3\n");
      if(IncreasedTreshold(trEv, InputTag("hlt4jet25U","",TriggerProcess), 30, 4, false)){JetTrigger = true;}
//      if(tr.accept(tr.triggerIndex("HLT_QuadJet25U_v3") )){JetTrigger = true;}
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_QuadJet25U_v2")){
         //if(CountEvent<MaxPrint)printf("Rescale HLT_QuadJet25U_v2\n");
         if(IncreasedTreshold(trEv, InputTag("hlt4jet25U","",TriggerProcess), 30, 4, false)){JetTrigger = true;}
//         if(tr.accept(tr.triggerIndex("HLT_QuadJet25U_v2") )){JetTrigger = true;}
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("OpenHLT_QuadJet25U")){
            //if(CountEvent<MaxPrint)printf("Rescale OpenHLT_QuadJet25U\n");
            if(IncreasedTreshold(trEv, InputTag("hlt4jet25U","",TriggerProcess), 30, 4, false)){JetTrigger = true;}
//            if(tr.accept(tr.triggerIndex("OpenHLT_QuadJet25U") )){JetTrigger = true;}
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_QuadJet15U")){
               //if(CountEvent<MaxPrint)printf("Use HLT_QuadJet15U Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hlt4jet15U","",TriggerProcess), 30, 4, false)){JetTrigger = true;}
            }else{
               printf("HSCPHLTFilter --> BUG with HLT_QuadJet15U\n");
            }
         }
      }
   }
*/

   //printf("Bits = %1i %1i %1i X Mask = %+2i %+2i %+2i -->",MuonTrigger,METTrigger,JetTrigger,MuonTriggerMask,METTriggerMask,JetTriggerMask);

   if(MuonTriggerMask==0)MuonTrigger=false;
   if(METTriggerMask ==0)METTrigger =false;
   if(JetTriggerMask ==0)JetTrigger =false;

   if(MuonTriggerMask>=0 && METTriggerMask>=0 && JetTriggerMask>=0){bool d =  (MuonTrigger | METTrigger | JetTrigger);/* printf("%i\n",d);*/return d;}

   if(MuonTriggerMask<0 && METTriggerMask <0){bool d =  !MuonTrigger & !METTrigger & JetTrigger; /*printf("%i\n",d);*/return d;}
   if(MuonTriggerMask<0 && JetTriggerMask <0){bool d =  !MuonTrigger & !JetTrigger & METTrigger; /*printf("%i\n",d);*/return d;}
   if(METTriggerMask <0 && JetTriggerMask <0){bool d =  !METTrigger  & !JetTrigger & MuonTrigger; /*printf("%i\n",d);*/return d;}

   if(MuonTriggerMask<0){bool d =  !MuonTrigger & (METTrigger  | JetTrigger); /*printf("%i\n",d);*/return d;}
   if(METTriggerMask <0){bool d =  !METTrigger  & (MuonTrigger | JetTrigger); /*printf("%i\n",d);*/return d;}
   if(JetTriggerMask <0){bool d =  !JetTrigger  & (MuonTrigger | METTrigger); /*printf("%i\n",d);*/return d;}

   /*printf("0\n");*/return false;
}

bool HSCPHLTFilter::IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, int NObjectAboveThreshold, bool averageThreshold)
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
            if(TOC[KEYS[i]].pt()> NewThreshold) NObjectAboveThresholdObserved++;
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





#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDFilter.h"

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

      bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, int NObjectAboveThreshold, bool averageThreshold);

      std::string TriggerProcess;

      unsigned int CountEvent;
      unsigned int MaxPrint;
};


/////////////////////////////////////////////////////////////////////////////////////
HSCPHLTFilter::HSCPHLTFilter(const edm::ParameterSet& iConfig)
{
   TriggerProcess        = iConfig.getParameter<std::string>          ("TriggerProcess");
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

/////////////////////////////////////////////////////////////////////////////////////
bool HSCPHLTFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::TriggerResultsByName tr = iEvent.triggerResultsByName(TriggerProcess);
   if(!tr.isValid()){    printf("NoValidTrigger\n");  }

//   for(unsigned int i=0;i<tr.size();i++){
//      printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
//   }fflush(stdout);


   edm::Handle< trigger::TriggerEvent > trEvHandle;
   iEvent.getByLabel("hltTriggerSummaryAOD", trEvHandle);
   trigger::TriggerEvent trEv = *trEvHandle;

   CountEvent++;
   if(CountEvent<MaxPrint)printf("------------------------\n");


   unsigned int TrIndex_Unknown     = tr.size();

   // HLT TRIGGER BASED ON 1 MUON!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu15_v1")){
       if(CountEvent<MaxPrint)printf("Use HLT_Mu15_v1\n");
      if(tr.accept(tr.triggerIndex("HLT_Mu15_v1")))return true;
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu11")){
            if(CountEvent<MaxPrint)printf("Use HLT_Mu11 Rescaled\n");
            if(IncreasedTreshold(trEv, InputTag("hltSingleMu11L3Filtered11","",TriggerProcess), 15, 1, false))return true;
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu9")){
            if(CountEvent<MaxPrint)printf("Use HLT_Mu9 Rescaled\n");
            if(IncreasedTreshold(trEv, InputTag("hltSingleMu9L3Filtered9","",TriggerProcess), 15, 1, false))return true;
         }else{
             printf("HSCPHLTFilter --> BUG with HLT_Mu9\n");
         }
      }
   }


   // HLT TRIGGER BASED ON 2 MUONS!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_DoubleMu3_v2")){
         if(CountEvent<MaxPrint)printf("Use HLT_DoubleMu3_v2\n");
         if(tr.accept(tr.triggerIndex("HLT_DoubleMu3_v2")))return true;
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_DoubleMu3")){
         if(CountEvent<MaxPrint)printf("Use HLT_DoubleMu3\n");
         if(tr.accept(tr.triggerIndex("HLT_DoubleMu3")))return true;
      }else{
         printf("HSCPHLTFilter --> BUG with HLT_DoubleMu3\n");
      }
   }


   // HLT TRIGGER BASED ON MET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_MET100_v3")){
      if(CountEvent<MaxPrint)printf("Use HLT_MET100_v3\n");
      if(tr.accept(tr.triggerIndex("HLT_MET100_v3")))return true;
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_MET100_v2")){
         if(CountEvent<MaxPrint)printf("Use HLT_MET100_v2\n");
         if(tr.accept(tr.triggerIndex("HLT_MET100_v2")))return true;
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_MET100")){
            if(CountEvent<MaxPrint)printf("Use HLT_MET100\n");
            if(tr.accept(tr.triggerIndex("HLT_MET100")))return true;
         }else{
            printf("HSCPHLTFilter --> BUG with HLT_MET100\n");
         }
      }
   }


   // HLT TRIGGER BASED ON 1 JET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet140U_v3")){
      if(CountEvent<MaxPrint)printf("Use HLT_Jet140U_v3\n");
      if(tr.accept(tr.triggerIndex("HLT_Jet140U_v3")))return true;
   }else{ 
      if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet140U_v1")){
         if(CountEvent<MaxPrint)printf("Use HLT_Jet140U_v1\n");
         if(tr.accept(tr.triggerIndex("HLT_Jet140U_v1")))return true;
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet100U")){
             if(CountEvent<MaxPrint)printf("Use HLT_Jet100U Rescaled\n");
             if(IncreasedTreshold(trEv, InputTag("hlt1jet100U","",TriggerProcess), 140, 1, false))return true;
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet70U")){   
               if(CountEvent<MaxPrint)printf("Use HLT_Jet70U Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hlt1jet70U","",TriggerProcess), 140, 1, false))return true;
            }else{
               if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet50U")){
                  if(CountEvent<MaxPrint)printf("Use HLT_Jet50U Rescaled\n");
                  if(IncreasedTreshold(trEv, InputTag("hlt1jet50U","",TriggerProcess), 140, 1, false))return true;
               }else{
                  printf("HSCPHLTFilter --> BUG with HLT_Jet50U\n");
               }
            }
         }
      }
   }


   // HLT TRIGGER BASED ON 2 JETS!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_DiJetAve140U_v3")){
      if(CountEvent<MaxPrint)printf("Use HLT_DiJetAve140U_v3\n");
      if(tr.accept(tr.triggerIndex("HLT_DiJetAve140U_v3") ))return true;
   }else{  
      if(TrIndex_Unknown != tr.triggerIndex("OpenHLT_DiJetAve70")){
         if(CountEvent<MaxPrint)printf("Use OpenHLT_DiJetAve70\n");
         if(IncreasedTreshold(trEv, InputTag("hltDiJetAve70U","",TriggerProcess), 140, 2, true))return true;
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("OpenHLT_DiJetAve50U")){
               if(CountEvent<MaxPrint)printf("Use OpenHLT_DiJetAve50 Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hltDiJetAve50U","",TriggerProcess), 140, 2, true))return true;
               if(IncreasedTreshold(trEv, InputTag("openhltDiJetAve50U","",TriggerProcess), 140, 2, true))return true;
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_DiJetAve50U")){
               if(CountEvent<MaxPrint)printf("Use HLT_DiJetAve50 Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hltDiJetAve50U","",TriggerProcess), 140, 2, true))return true;
            }else{
               if(TrIndex_Unknown != tr.triggerIndex("HLT_DiJetAve30U_8E29")){ 
                  if(CountEvent<MaxPrint)printf("Use HLT_DiJetAve30U_8E29 Rescaled\n");
                  if(IncreasedTreshold(trEv, InputTag("hltDiJetAve30U8E29","",TriggerProcess), 140, 2, true))return true;
               }else{
                   printf("HSCPHLTFilter --> BUG with HLT_DiJetAve30U_8E29\n");
               }
            }
         }
      }
   }


   // HLT TRIGGER BASED ON 4 JETS!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_QuadJet25U_v3")){
      if(CountEvent<MaxPrint)printf("Use HLT_QuadJet25U_v3\n");
      if(tr.accept(tr.triggerIndex("HLT_QuadJet25U_v3") ))return true;
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_QuadJet25U_v2")){
         if(CountEvent<MaxPrint)printf("Use HLT_QuadJet25U_v2\n");
         if(tr.accept(tr.triggerIndex("HLT_QuadJet25U_v2") ))return true;
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("OpenHLT_QuadJet25U")){
            if(CountEvent<MaxPrint)printf("Use OpenHLT_QuadJet25U\n");
            if(tr.accept(tr.triggerIndex("OpenHLT_QuadJet25U") ))return true;
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_QuadJet15U")){
               if(CountEvent<MaxPrint)printf("Use HLT_QuadJet15U Rescaled\n");
               if(IncreasedTreshold(trEv, InputTag("hlt4jet15U","",TriggerProcess), 25, 4, false))return true;
            }else{
               printf("HSCPHLTFilter --> BUG with HLT_QuadJet15U\n");
            }
         }
      }
   }
 
   return false;
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





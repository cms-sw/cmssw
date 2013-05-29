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
      if(tr.accept(tr.triggerIndex("HLT_Mu24_v1"))){MuonTrigger = true;}
   }else{
      printf("HSCPHLTFilter --> HLT_Mu24_v1  not found\n");
      for(unsigned int i=0;i<tr.size();i++){
         printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
      }fflush(stdout);
      exit(0);
   }


   // HLT TRIGGER BASED ON 2 MUONS!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_DoubleMu6_v1")){
      if(tr.accept(tr.triggerIndex("HLT_DoubleMu6_v1"))){MuonTrigger = true;}
   }else{
      printf("HSCPHLTFilter --> HLT_DoubleMu6_v1  not found\n");
      for(unsigned int i=0;i<tr.size();i++){
         printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
      }fflush(stdout);
      exit(0);
   }

   // HLT TRIGGER BASED ON MET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v2")){
      if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v2"))){METTrigger = true;}
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v1")){
         if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v1"))){METTrigger = true;}
      }else{
         printf("HSCPHLTFilter --> HLT_PFMHT150_v2 or v1  not found\n");
         for(unsigned int i=0;i<tr.size();i++){
            printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
         }fflush(stdout);
         exit(0);
      }
   }

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





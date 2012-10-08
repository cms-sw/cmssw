// Original Author:  Loic Quertenmont

#ifndef HSCP_ANALYSIS_SAMPLE
#define HSCP_ANALYSIS_SAMPLE

class stSample{
   public:
   std::string CMSSW;
   int         Type;
   std::string Name;
   std::string FileName;
   std::string Legend;
   std::string Pileup;
   double      Mass;
   double      XSec;
   bool        MakePlot;
   float       WNC0;//weight for signal event with 0 Charged HSCP
   float       WNC1;//weight for signal event with 1 Charged HSCP
   float       WNC2;//weight for signal event with 2 Charged HSCP

   //These weights are used to reweight the gluino samples for different scenario of gluinobll fraction (f).  
   //The weights are independent on the mass and are computed as a reweighting factor w.r.t a f=10% gluino ball sample.
   //Weights have been computed in 2010 to be : 
   //for f=00% --> [0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015]
   //for f=10% --> [0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015]
   //for f=50% --> [0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015]

   stSample(){};
   stSample(std::string CMSSW_, int Type_, std::string Name_, std::string FileName_, std::string Legend_, std::string Pileup_, double Mass_, double XSec_, bool MakePlot_, float WNC0_, float WNC1_, float WNC2_){CMSSW=CMSSW_; Type=Type_; Name=Name_; FileName=FileName_; Legend=Legend_; Pileup=Pileup_; Mass=Mass_; XSec=XSec_; MakePlot=MakePlot_; WNC0=WNC0_; WNC1=WNC1_; WNC2=WNC2_;}

   int readFromFile(FILE* pFile){ 
      char line[4096];
      char* toReturn = fgets(line, 4096, pFile);
      if(!toReturn)return EOF;
      char* pch=strtok(line,","); int Arg=0; string tmp; int temp;
      while (pch!=NULL){
         switch(Arg){
            case  0: tmp = pch;  CMSSW    = tmp.substr(tmp.find("\"")+1,tmp.rfind("\"")-tmp.find("\"")-1); break;
            case  1: sscanf(pch, "%d", &Type); break;
            case  2: tmp = pch;  Name     = tmp.substr(tmp.find("\"")+1,tmp.rfind("\"")-tmp.find("\"")-1); break;
            case  3: tmp = pch;  FileName = tmp.substr(tmp.find("\"")+1,tmp.rfind("\"")-tmp.find("\"")-1); break;
            case  4: tmp = pch;  Legend   = tmp.substr(tmp.find("\"")+1,tmp.rfind("\"")-tmp.find("\"")-1); break;
            case  5: tmp = pch;  Pileup   = tmp.substr(tmp.find("\"")+1,tmp.rfind("\"")-tmp.find("\"")-1); break;
            case  6: sscanf(pch, "%lf", &Mass); break;
            case  7: sscanf(pch, "%lf", &XSec); break;
            case  8: sscanf(pch, "%d", &temp); MakePlot=(temp>0); break;
            case  9: sscanf(pch, "%f", &WNC0); break;
            case 10: sscanf(pch, "%f", &WNC1); break;
            case 11: sscanf(pch, "%f", &WNC2); break;
            default:return EOF;break;
         }
         pch=strtok(NULL,",");Arg++;
      }
      return 0;
   }

   void print(FILE* pFile=stdout){
      fprintf(pFile, "%-9s, %1d, %-30s, %-40s, %-60s, %-8s, % 5g, %+12.10E, %1i, %5.3f, %5.3f, %5.3f\n", (string("\"")+CMSSW+"\"").c_str(), Type, (string("\"")+Name+"\"").c_str(), (string("\"")+FileName+"\"").c_str(), (string("\"")+Legend+"\"").c_str(), (string("\"")+Pileup+"\"").c_str(), Mass, XSec, MakePlot, WNC0, WNC1, WNC2);
   }

   std::string ModelName(){
      char MassStr[255];sprintf(MassStr, "%.0f",Mass);
      string toReturn=Name; toReturn.erase(toReturn.find(MassStr), string(MassStr).length());
      return toReturn;
   }

   std::string ModelLegend(){
      char MassStr[255];sprintf(MassStr, "%.0f",Mass);
      string toReturn=Legend; toReturn.erase(toReturn.find(MassStr), string(MassStr).length());
      if(toReturn.find(" GeV/#font[12]{c}^{2}")!=string::npos)toReturn.erase(toReturn.find(" GeV/#font[12]{c}^{2}"), string(" GeV/#font[12]{c}^{2}").length());
      return toReturn;
   }

   double GetFGluinoWeight(int NChargedHSCP){ 
      if(Type!=2)return 1;
      double Weight;
      switch(NChargedHSCP){
         case 0: Weight=WNC0; break;
         case 1: Weight=WNC1; break;
         case 2: Weight=WNC2; break;
         default: return 1.0;
      }
      if(Weight<0)return 1.0;
      return Weight;
   }

};

void GetSampleDefinition(std::vector<stSample>& samples, std::string sampleTxtFile="Analysis_Samples.txt"){
      FILE* pFile = fopen(sampleTxtFile.c_str(),"r");
         if(!pFile){printf("Can't open %s\n","Analysis_Samples.txt"); return;}
         stSample newSample;      
         while(newSample.readFromFile(pFile)!=EOF){samples.push_back(newSample);}
      fclose(pFile);
}

void GetInputFiles(stSample sample, std::string BaseDirectory_, std::vector<std::string>& inputFiles, int period=0){
   std::vector<string> fileNames;
   char* tmp = (char*)sample.FileName.c_str();
   char* pch=strtok(tmp,";");
   while (pch!=NULL){
      fileNames.push_back(pch);
      pch=strtok(NULL,",");
   }

   for(unsigned int f=0;f<fileNames.size();f++){  
      if(sample.Type>=2){ //MC Signal
        if (period==0) inputFiles.push_back(BaseDirectory_ + fileNames[f] + "_BX0.root");
        if (period==1) inputFiles.push_back(BaseDirectory_ + fileNames[f] + "_BX1.root");
      }else{ //Data or MC Background
        inputFiles.push_back(BaseDirectory_ + fileNames[f] + ".root");
      }
   }
}

int JobIdToIndex(string JobId, const std::vector<stSample>& samples){
   for(unsigned int s=0;s<samples.size();s++){
      if(samples[s].Name==JobId)return s;
   }return -1;
}

void keepOnlySamplesOfTypeX(std::vector<stSample>& samples, int TypeX){
   for(unsigned int s=0;s<samples.size();s++){if(samples[s].Type!=TypeX){samples.erase(samples.begin()+s);s--;} }
}

void keepOnlyTheXtoYSamples(std::vector<stSample>& samples, unsigned int X, unsigned int Y){
   std::vector<stSample> tmp; for(unsigned int s=X;s<=Y && s<samples.size();s++){tmp.push_back(samples[s]);} samples.clear();
   for(unsigned int s=0;s<tmp.size();s++)samples.push_back(tmp[s]);
}

void keepOnlySamplesOfNameX(std::vector<stSample>& samples, string NameX){
   for(unsigned int s=0;s<samples.size();s++){if(samples[s].Name!=NameX){samples.erase(samples.begin()+s);s--;} }
}  





//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Genertic code related to samples processing in FWLITE --> functions below will be loaded only if FWLITE compiler variable is defined

#ifdef FWLITE

unsigned long GetInitialNumberOfMCEvent(const vector<string>& fileNames);
double GetSampleWeight  (const double& IntegratedLuminosityInPb=-1, const double& IntegratedLuminosityInPbBeforeTriggerChange=-1, const double& CrossSection=0, const double& MCEvents=0, int period=0);
double GetSampleWeightMC(const double& IntegratedLuminosityInPb, const std::vector<string> fileNames, const double& XSection, const double& SampleSize, double MaxEvent);
double GetPUWeight      (const fwlite::ChainEvent& ev, const std::string& pileup, double &PUSystFactor);

// loop on all the lumi blocks for an EDM file in order to count the number of events that are in a sample
// this is useful to determine how to normalize the events (compute weight)
unsigned long GetInitialNumberOfMCEvent(const vector<string>& fileNames)
{
   unsigned long Total = 0;
   for(unsigned int f=0;f<fileNames.size();f++){
      TFile *file;
      size_t place=fileNames[f].find("dcache");
      if(place!=string::npos){
 	 string name=fileNames[f];
         name.replace(place, 7, "dcap://cmsgridftp.fnal.gov:24125");
         file = new TDCacheFile (name.c_str());
      }else{
         file = TFile::Open(fileNames[f].c_str());
      }

      fwlite::LuminosityBlock ls( file );
      for(ls.toBegin(); !ls.atEnd(); ++ls){
         fwlite::Handle<edm::MergeableCounter> nEventsTotalCounter;
         nEventsTotalCounter.getByLabel(ls,"nEventsBefSkim");
         if(!nEventsTotalCounter.isValid()){printf("Invalid nEventsTotalCounterH\n");continue;}
         Total+= nEventsTotalCounter->value;
      }
   }
   return Total;
}

// compute event weight for the signal samples based on number of events in the sample, cross section and intergated luminosity
double GetSampleWeight(const double& IntegratedLuminosityInPb, const double& IntegratedLuminosityInPbBeforeTriggerChange, const double& CrossSection, const double& MCEvents, int period){
  double Weight = 1.0;
  if(IntegratedLuminosityInPb>=IntegratedLuminosityInPbBeforeTriggerChange && IntegratedLuminosityInPb>0){
    double NMCEvents = MCEvents;
    //if(MaxEntry>0)NMCEvents=std::min(MCEvents,(double)MaxEntry);
    if      (period==0)Weight = (CrossSection * IntegratedLuminosityInPbBeforeTriggerChange) / NMCEvents;
    else if (period==1)Weight = (CrossSection * (IntegratedLuminosityInPb-IntegratedLuminosityInPbBeforeTriggerChange)) / NMCEvents;
  }
  return Weight;
}

// compute event weight for the MC background samples based on number of events in the sample, cross section and intergated luminosity
double GetSampleWeightMC(const double& IntegratedLuminosityInPb, const std::vector<string> fileNames, const double& XSection, const double& SampleSize, double MaxEvent){
  double Weight = 1.0;
   unsigned long InitNumberOfEvents = GetInitialNumberOfMCEvent(fileNames); 
   double SampleEquivalentLumi = InitNumberOfEvents / XSection;
   if(MaxEvent<0)MaxEvent=SampleSize;
   printf("GetSampleWeight MC: IntLumi = %6.2E  SampleLumi = %6.2E --> EventWeight = %6.2E --> ",IntegratedLuminosityInPb,SampleEquivalentLumi, IntegratedLuminosityInPb/SampleEquivalentLumi);
//   printf("Sample NEvent = %6.2E   SampleEventUsed = %6.2E --> Weight Rescale = %6.2E\n",SampleSize, MaxEvent, SampleSize/MaxEvent);
   Weight = (IntegratedLuminosityInPb/SampleEquivalentLumi) * (SampleSize/MaxEvent);
   printf("FinalWeight = %6.2f\n",Weight);
   return Weight;
}

// compute a weight to make sure that the pileup distribution in Signal/Background MC is compatible to the pileup distribution in data
double GetPUWeight(const fwlite::ChainEvent& ev, const std::string& pileup, double &PUSystFactor, edm::LumiReWeighting& LumiWeightsMC, reweight::PoissonMeanShifter& PShift){
   fwlite::Handle<std::vector<PileupSummaryInfo> > PupInfo;
   PupInfo.getByLabel(ev, "addPileupInfo");
   if(!PupInfo.isValid()){printf("PileupSummaryInfo Collection NotFound\n");return 1.0;}
   double PUWeight_thisevent=1;
   std::vector<PileupSummaryInfo>::const_iterator PVI;
   int npv = -1;
   if(pileup=="S4"){
      float sum_nvtx = 0;
      for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
         npv = PVI->getPU_NumInteractions();
         sum_nvtx += float(npv);
      }
      float ave_nvtx = sum_nvtx/3.;
      PUWeight_thisevent = LumiWeightsMC.weight( ave_nvtx );
      PUSystFactor = PShift.ShiftWeight( ave_nvtx );
   }else if(pileup=="S3"){
      for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
         int BX = PVI->getBunchCrossing();
         if(BX == 0) {
            npv = PVI->getPU_NumInteractions();
            continue;
         }
      }
      PUWeight_thisevent = LumiWeightsMC.weight( npv );
      PUSystFactor = PShift.ShiftWeight( npv );
   }
   return PUWeight_thisevent;
}

#endif //end FWLITE block



//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Define Theoretical XSection and error band for 7TeVStops and 7TeVGluino

//NLO Stop@7TeV
double THXSEC7TeV_Stop_Mass [] = {100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.,320.,340.,360.,380.,400.,420.,440.,460.,480.,500.,520.,540.,560.,580.,600.,620.,640.,660.,680.,700.,720.,740.,760.,800.,820.,840.,860.,880.,900.,920.,940.,960.,980.,1000.};
double THXSEC7TeV_Stop_Cen  [] = {4.23E+02,1.77E+02,8.27E+01,4.20E+01,2.27E+01,1.30E+01,7.71E+00,4.75E+00,3.02E+00,1.97E+00,1.31E+00,8.86E-01,6.11E-01,4.28E-01,3.03E-01,2.18E-01,1.58E-01,1.16E-01,8.55E-02,6.37E-02,4.78E-02,3.62E-02,2.75E-02,2.10E-02,1.62E-02,1.25E-02,9.68E-03,7.51E-03,5.84E-03,4.55E-03,3.56E-03,2.78E-03,2.17E-03,1.67E-03,1.14E-03,9.14E-04,7.33E-04,5.89E-04,4.75E-04,3.82E-04,3.09E-04,2.50E-04,2.03E-04,1.64E-04,1.33E-04};
double THXSEC7TeV_Stop_Low  [] = {4.86E+02,2.02E+02,9.41E+01,4.76E+01,2.57E+01,1.47E+01,8.72E+00,5.37E+00,3.41E+00,2.22E+00,1.48E+00,1.00E+00,6.92E-01,4.84E-01,3.44E-01,2.47E-01,1.79E-01,1.32E-01,9.75E-02,7.28E-02,5.47E-02,4.15E-02,3.15E-02,2.42E-02,1.87E-02,1.44E-02,1.12E-02,8.71E-03,6.78E-03,5.29E-03,4.14E-03,3.23E-03,2.52E-03,1.94E-03,1.35E-03,1.08E-03,8.68E-04,7.01E-04,5.65E-04,4.56E-04,3.69E-04,3.00E-04,2.44E-04,1.98E-04,1.61E-04};
double THXSEC7TeV_Stop_High [] = {3.74E+02,1.56E+02,7.29E+01,3.70E+01,2.01E+01,1.14E+01,6.77E+00,4.17E+00,2.65E+00,1.72E+00,1.14E+00,7.73E-01,5.31E-01,3.70E-01,2.63E-01,1.88E-01,1.36E-01,9.93E-02,7.31E-02,5.44E-02,4.06E-02,3.07E-02,2.33E-02,1.77E-02,1.36E-02,1.05E-02,8.11E-03,6.27E-03,4.87E-03,3.78E-03,2.95E-03,2.30E-03,1.79E-03,1.38E-03,9.31E-04,7.42E-04,5.93E-04,4.74E-04,3.81E-04,3.06E-04,2.47E-04,1.98E-04,1.60E-04,1.29E-04,1.04E-04};

//NLO Stop@8TeV
double THXSEC8TeV_Stop_Mass [] = {100.0,120.0,140.0,160.0,180.0,200.0,220.0,240.0,260.0,280.0,300.0,320.0,340.0,360.0,380.0,400.0,420.0,440.0,460.0,480.0,500.0,520.0,540.0,560.0,580.0,600.0,620.0,640.0,660.0,680.0,700.0,720.0,740.0,760.0,780.0,800.0,820.0,840.0,860.0,880.0,900.0,920.0,940.0,960.0,980.0,1000.0,1020.0,1040.0,1060.0,1080.0,1100.0,1120.0,1140.0,1160.0,1180.0,1200.0,1220.0,1240.0,1260.0,1280.0,1300.0,1320.0,1340.0,1360.0,1380.0,1400.0,1420.0,1440.0,1460.0,1480.0,1500.0,1520.0,1540.0,1560.0,1580.0,1600.0,1620.0,1640.0,1660.0,1680.0,1700.0,1720.0,1740.0,1760.0,1780.0,1800.0,1820.0,1840.0,1860.0,1880.0,1900.0,1920.0,1940.0,1960.0,1980.0};
double THXSEC8TeV_Stop_Cen  [] = {526.0,226.0,107.0,55.0,30.3,17.7,10.7,6.69,4.32,2.85,1.93,1.33,0.935,0.666,0.48,0.351,0.259,0.194,0.146,0.111,0.0846,0.0652,0.0505,0.0394,0.0309,0.0244,0.0193,0.0153,0.0123,0.00985,0.00794,0.00642,0.0052,0.00423,0.00345,0.00282,0.00231,0.0019,0.00156,0.00129,0.00107,0.000888,0.000738,0.000614,0.000512,0.000427,0.000357,0.000298,0.00025,0.00021,0.000177,0.000149,0.000126,0.000106,8.95e-05,7.54e-05,6.37e-05,5.4e-05,4.58e-05,3.88e-05,3.29e-05,2.79e-05,2.37e-05,2.01e-05,1.71e-05,1.45e-05,1.24e-05,1.05e-05,8.98e-06,7.65e-06,6.51e-06,5.53e-06,4.7e-06,3.99e-06,3.38e-06,2.86e-06,2.4e-06,2.02e-06,1.73e-06,1.49e-06,1.29e-06,1.1e-06,9.35e-07,7.95e-07,6.77e-07,5.77e-07,4.92e-07,4.19e-07,3.57e-07,3.04e-07,2.58e-07,2.19e-07,1.87e-07,1.59e-07,1.35e-07};
double THXSEC8TeV_Stop_Low [] = {456.40976717,195.648917903,92.2540349723,47.2865665445,25.9397832106,15.0956844963,9.08927760689,5.65777076356,3.6403946419,2.38881618792,1.61175270749,1.10604627471,0.775030543643,0.549718373166,0.394375877481,0.287159,0.211071455628,0.157614743299,0.118115158583,0.0896125182004,0.067981415327,0.0521458627606,0.0402437661268,0.0312470919086,0.0244182952548,0.019218893828,0.0151215196692,0.0119430303758,0.00956579804952,0.00763312081598,0.00613258287528,0.0049339106769,0.00397887414586,0.00321918065883,0.00261564964342,0.00212811380764,0.00173487125664,0.00142268794902,0.00116204134147,0.000956279372476,0.000788408712819,0.000650528932861,0.000538210968336,0.000445184820841,0.000368817901272,0.000305789996708,0.000254186271492,0.000210958143595,0.000175969661542,0.000146735307698,0.000123013224389,0.000102929880545,8.6402235565e-05,7.222995e-05,6.05164427274e-05,5.0580220783e-05,4.24247029546e-05,3.5715773314e-05,3.00635721872e-05,2.52451336458e-05,2.12193012116e-05,1.78446031904e-05,1.50354723785e-05,1.26276243982e-05,1.06536409697e-05,8.94732163468e-06,7.59026356243e-06,6.35376571526e-06,5.37945064143e-06,4.53447994822e-06,3.81776527832e-06,3.21539615667e-06,2.71447659044e-06,2.28387219518e-06,1.91493168116e-06,1.60622722428e-06,1.34502348655e-06,1.1253862558e-06,9.53375342241e-07,8.07186767383e-07,6.85400212545e-07,5.77264615054e-07,4.84520464268e-07,4.06967984563e-07,3.41287787427e-07,2.86338664102e-07,2.40809064183e-07,2.02382848192e-07,1.70268348502e-07,1.4267677199e-07,1.19072781243e-07,9.95434872399e-08,8.37672797081e-08,7.02461674403e-08,5.8546055207e-08};
double THXSEC8TeV_Stop_High  [] = {612.151729265,262.612588103,124.288464259,63.9150906166,35.2825924691,20.6572041873,12.51422531,7.84901646422,5.08233985546,3.36513880209,2.28626927651,1.57937,1.11448052719,0.796049681797,0.575829786962,0.422291028968,0.312903555384,0.235042620074,0.177634138358,0.135439676246,0.103594494844,0.0801823918578,0.0623096661467,0.0488132526053,0.0384157751721,0.030457195299,0.02419270263,0.0192543023786,0.0155348683796,0.0124867180798,0.0101081955359,0.00820720230018,0.00667610315198,0.00545393409975,0.00446719444332,0.00366690218399,0.0030168136425,0.00249290452531,0.00205643855325,0.00170910492719,0.0014249051169,0.00118823222984,0.000992225516805,0.00082975230322,0.000695336835754,0.000582782847848,0.000489743672799,0.000410791069838,0.000346582593019,0.000292517320796,0.000247919343036,0.00020993472894,0.000178644320362,0.000151183052034,0.000128497606094,0.000108904879045,9.25617106816e-05,7.88942071601e-05,6.73641928454e-05,5.74235824351e-05,4.90287275026e-05,4.18712011298e-05,3.58244154887e-05,3.05926261591e-05,2.62141570875e-05,2.23830855768e-05,1.92778016416e-05,1.64463855993e-05,1.41720928606e-05,1.21588956729e-05,1.04173416472e-05,8.9066137467e-06,7.61913263458e-06,6.50912689733e-06,5.54723440685e-06,4.71830691087e-06,3.96587175902e-06,3.34925029375e-06,2.88869395402e-06,2.51958334975e-06,2.20968359995e-06,1.90103535042e-06,1.62823072952e-06,1.3949661943e-06,1.1975604273e-06,1.02928057305e-06,8.84816255638e-07,7.59765184201e-07,6.52949889306e-07,5.6101392013e-07,4.80393896187e-07,4.11598677578e-07,3.54793913074e-07,3.04265637883e-07,2.60687256456e-07};

//NLO Gluino@7TeV
double THXSEC7TeV_Gluino_Mass [] = {300.000000,320.000000,340.000000,360.000000,380.000000,400.000000,420.000000,440.000000,460.000000,480.000000,500.000000,520.000000,540.000000,560.000000,580.000000,600.000000,620.000000,640.000000,660.000000,680.000000,700.000000,720.000000,740.000000,760.000000,780.000000,800.000000,820.000000,840.000000,860.000000,880.000000,900.000000,920.000000,940.000000,960.000000,980.000000,1000.000000,1020.000000,1040.000000,1060.000000,1080.000000,1100.000000,1120.000000,1140.000000,1160.000000,1180.000000,1200.000000};
double THXSEC7TeV_Gluino_Cen  [] = {6.580000E+01,4.480000E+01,3.100000E+01,2.170000E+01,1.550000E+01,1.120000E+01,8.150000E+00,6.010000E+00,4.470000E+00,3.360000E+00,2.540000E+00,1.930000E+00,1.480000E+00,1.140000E+00,8.880000E-01,6.930000E-01,5.430000E-01,4.280000E-01,3.390000E-01,2.690000E-01,2.140000E-01,1.720000E-01,1.370000E-01,1.110000E-01,8.950000E-02,7.250000E-02,5.880000E-02,4.790000E-02,3.910000E-02,3.200000E-02,2.620000E-02,2.150000E-02,1.760000E-02,1.450000E-02,1.190000E-02,9.870000E-03,8.150000E-03,6.740000E-03,5.590000E-03,4.640000E-03,3.860000E-03,3.200000E-03,2.660000E-03,2.220000E-03,1.850000E-03,1.540000E-03};
double THXSEC7TeV_Gluino_Low  [] = {7.553474E+01,5.146393E+01,3.566432E+01,2.512864E+01,1.804830E+01,1.304305E+01,9.556500E+00,7.044302E+00,5.271630E+00,3.988566E+00,3.018432E+00,2.308958E+00,1.773087E+00,1.374688E+00,1.070057E+00,8.414255E-01,6.640146E-01,5.237697E-01,4.145975E-01,3.315876E-01,2.657578E-01,2.134368E-01,1.716366E-01,1.387783E-01,1.127959E-01,9.142058E-02,7.469319E-02,6.086165E-02,4.999024E-02,4.092504E-02,3.378647E-02,2.791464E-02,2.286025E-02,1.896689E-02,1.560576E-02,1.300309E-02,1.082652E-02,8.957356E-03,7.480351E-03,6.255235E-03,5.204033E-03,4.420000E-03,3.730000E-03,3.160000E-03,2.670000E-03,2.270000E-03};
double THXSEC7TeV_Gluino_High [] = {5.742197E+01,3.883067E+01,2.678981E+01,1.876433E+01,1.338010E+01,9.580894E+00,6.973463E+00,5.091237E+00,3.788946E+00,2.845079E+00,2.130996E+00,1.626191E+00,1.243271E+00,9.589442E-01,7.381439E-01,5.751390E-01,4.515518E-01,3.522239E-01,2.787396E-01,2.208681E-01,1.759448E-01,1.397393E-01,1.116011E-01,8.996432E-02,7.195999E-02,5.827633E-02,4.733762E-02,3.809243E-02,3.105941E-02,2.538284E-02,2.054873E-02,1.688133E-02,1.379495E-02,1.136371E-02,9.251707E-03,7.653260E-03,6.325800E-03,5.168580E-03,4.283127E-03,3.515531E-03,2.919556E-03,2.300000E-03,1.890000E-03,1.530000E-03,1.250000E-03,1.000000E-03};

//NLO Gluino@8TeV
double THXSEC8TeV_Gluino_Mass [] = {300.0,320.0,340.0,360.0,380.0,400.0,420.0,440.0,460.0,480.0,500.0,520.0,540.0,560.0,580.0,600.0,620.0,640.0,660.0,680.0,700.0,720.0,740.0,760.0,780.0,800.0,820.0,840.0,860.0,880.0,900.0,920.0,940.0,960.0,980.0,1000.0,1020.0,1040.0,1060.0,1080.0,1100.0,1120.0,1140.0,1160.0,1180.0,1200.0,1220.0,1240.0,1260.0,1280.0,1300.0,1320.0,1340.0,1360.0,1380.0,1400.0,1420.0,1440.0,1460.0,1480.0,1500.0,1520.0,1540.0,1560.0,1580.0,1600.0,1620.0,1640.0,1660.0,1680.0,1700.0,1720.0,1740.0,1760.0,1780.0,1800.0,1820.0,1840.0,1860.0,1880.0,1900.0,1920.0,1940.0,1960.0,1980.0};
double THXSEC8TeV_Gluino_Cen  [] = {103.0,70.6,49.3,35.1,25.3,18.5,13.6,10.2,7.66,5.82,4.46,3.44,2.66,2.08,1.63,1.29,1.02,0.816,0.653,0.525,0.424,0.344,0.279,0.228,0.186,0.153,0.126,0.104,0.0856,0.0709,0.0588,0.0489,0.0408,0.0341,0.0285,0.0239,0.02,0.0168,0.0141,0.0119,0.01,0.00845,0.00714,0.00605,0.00513,0.00435,0.00369,0.00314,0.00267,0.00227,0.00193,0.00164,0.0014,0.00119,0.00102,0.000866,0.000739,0.000631,0.000539,0.00046,0.000393,0.000336,0.000287,0.000245,0.000209,0.000179,0.000153,0.000131,0.000112,9.52e-05,8.13e-05,6.94e-05,5.93e-05,5.06e-05,4.31e-05,3.68e-05,3.14e-05,2.67e-05,2.28e-05,1.94e-05,1.65e-05,1.4e-05,1.19e-05,1.02e-05,8.63e-06};
double THXSEC8TeV_Gluino_Low [] = {85.4522933503,58.3024212534,40.5404072161,28.7311959371,20.6110330718,14.9981486143,10.9817116552,8.2045980396,6.13599856717,4.64639599616,3.54679219469,2.7264626978,2.09915349508,1.63562861604,1.27699308705,1.00685323305,0.792356354919,0.632441971506,0.503862668979,0.403411581822,0.324940255218,0.262909295502,0.212319587364,0.173005424976,0.140570743272,0.115316873907,0.0947326706086,0.0778816775066,0.0638144532538,0.0526220125758,0.0434789865444,0.0359951306958,0.029944567911,0.0249059836578,0.020754521583,0.0173207451689,0.0144267991917,0.0120799394065,0.0100863120603,0.00847183872318,0.00708717142879,0.00595886341748,0.00501035858415,0.00422415500896,0.00356494485684,0.00300727877119,0.00253860291188,0.00214977087027,0.00181901909611,0.00153605196417,0.00129944484705,0.00109821151599,0.000931476154739,0.000787499214937,0.000671494422707,0.000566354149908,0.000479894886148,0.000407420453834,0.000345671282251,0.000292733637163,0.000248262306276,0.000210814299455,0.000178718470665,0.000151342195875,0.000127869581852,0.000108804207517,9.23062817067e-05,7.82498615939e-05,6.63400478633e-05,5.58615231572e-05,4.72597016726e-05,3.99537211035e-05,3.38756321908e-05,2.86208917333e-05,2.41297344895e-05,2.0403820875e-05,1.72341338273e-05,1.44941525431e-05,1.22590511119e-05,1.03016340378e-05,8.66483254755e-06,7.26134469144e-06,6.10042739754e-06,5.16703392541e-06,4.32379602699e-06};
double THXSEC8TeV_Gluino_High  [] = {123.042915069,84.6204861909,59.3621530586,42.4270342231,30.720245281,22.5429742994,16.6400543991,12.521,9.44012836998,7.19888313862,5.53783839898,4.28718343406,3.32951446035,2.61303391779,2.0546626767,1.63189165427,1.29573309725,1.03959612686,0.835270344823,0.673688637441,0.546222752456,0.444771492296,0.362012997158,0.297010039438,0.243115317971,0.200703894146,0.165936031728,0.137421289844,0.113579295087,0.094359434471,0.0785768709139,0.0656067780802,0.054914187159,0.0460700308168,0.0386624800886,0.0325504144943,0.0273588316481,0.0230737980716,0.019440405649,0.0164822382849,0.0139146663585,0.0118012857472,0.0100182943312,0.00852079529061,0.00725778032384,0.00618194507243,0.00526737821502,0.00450197140212,0.00384501422077,0.00328618755105,0.00280677982274,0.00239852281744,0.00205691112089,0.00175754634874,0.00151421701532,0.00129260450299,0.00110862278184,0.000951522205324,0.000817402023602,0.000701118190558,0.000602322798422,0.000517825498222,0.000444861065588,0.000382177349458,0.000327902228753,0.00028253307733,0.000243007105712,0.00020936089293,0.000180051621273,0.000154016576997,0.000132434560814,0.000113806344946,9.78639305045e-05,8.40542824159e-05,7.21176753055e-05,6.20064018569e-05,5.32790981711e-05,4.56654057754e-05,3.92622063114e-05,3.3654042914e-05,2.88430383571e-05,2.46542444561e-05,2.11238854891e-05,1.82236973231e-05,1.55451905173e-05};



//LO GMSB Stau@7TeV
double THXSEC7TeV_GMStau_Mass [] = {100,126,156,200,247,308,370,432,494};
double THXSEC7TeV_GMStau_Cen  [] = {1.3398,0.274591,0.0645953,0.0118093,0.00342512,0.00098447,0.000353388,0.000141817,6.17749e-05};
double THXSEC7TeV_GMStau_Low  [] = {1.18163,0.242982,0.0581651,0.0109992,0.00324853,0.00093519,0.000335826,0.000134024,5.83501e-05};
double THXSEC7TeV_GMStau_High [] = {1.48684,0.304386,0.0709262,0.012632,0.00358232,0.00102099,0.000366819,0.000147665,6.45963e-05};

//LO PairProduced Stau@7TeV
double THXSEC7TeV_PPStau_Mass [] = {100,126,156,200,247,308};
double THXSEC7TeV_PPStau_Cen  [] = {0.038200,0.0161,0.007040,0.002470,0.001010,0.000353};
double THXSEC7TeV_PPStau_Low  [] = {0.037076,0.0155927,0.0067891,0.00237277,0.00096927,0.000335308};
double THXSEC7TeV_PPStau_High [] = {0.0391443,0.016527,0.00723151,0.00253477,0.00103844,0.000363699};

#endif


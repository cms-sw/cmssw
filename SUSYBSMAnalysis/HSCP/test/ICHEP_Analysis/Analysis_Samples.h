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
   if(sample.Type>=2){ //MC Signal
     if (period==0) inputFiles.push_back(BaseDirectory_ + sample.FileName + ".root");
     if (period==1) inputFiles.push_back(BaseDirectory_ + sample.FileName + "BX1.root");
   }else{ //Data or MC Background
     inputFiles.push_back(BaseDirectory_ + sample.FileName + ".root");
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
   fwlite::ChainEvent tree(fileNames);

   for(unsigned int f=0;f<fileNames.size();f++){
      TFile *file;
      size_t place=fileNames[f].find("dcache");
      if(place!=string::npos){
         string name=fileNames[f];
         name.replace(place, 7, "dcap://cmsgridftp.fnal.gov:24125");
         file = new TDCacheFile (name.c_str());
      }else{
         file = new TFile (fileNames[f].c_str());
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

//NLO Gluino@7TeV
double THXSEC7TeV_Gluino_Mass [] = {300.000000,320.000000,340.000000,360.000000,380.000000,400.000000,420.000000,440.000000,460.000000,480.000000,500.000000,520.000000,540.000000,560.000000,580.000000,600.000000,620.000000,640.000000,660.000000,680.000000,700.000000,720.000000,740.000000,760.000000,780.000000,800.000000,820.000000,840.000000,860.000000,880.000000,900.000000,920.000000,940.000000,960.000000,980.000000,1000.000000,1020.000000,1040.000000,1060.000000,1080.000000,1100.000000,1120.000000,1140.000000,1160.000000,1180.000000,1200.000000};
double THXSEC7TeV_Gluino_Cen  [] = {6.580000E+01,4.480000E+01,3.100000E+01,2.170000E+01,1.550000E+01,1.120000E+01,8.150000E+00,6.010000E+00,4.470000E+00,3.360000E+00,2.540000E+00,1.930000E+00,1.480000E+00,1.140000E+00,8.880000E-01,6.930000E-01,5.430000E-01,4.280000E-01,3.390000E-01,2.690000E-01,2.140000E-01,1.720000E-01,1.370000E-01,1.110000E-01,8.950000E-02,7.250000E-02,5.880000E-02,4.790000E-02,3.910000E-02,3.200000E-02,2.620000E-02,2.150000E-02,1.760000E-02,1.450000E-02,1.190000E-02,9.870000E-03,8.150000E-03,6.740000E-03,5.590000E-03,4.640000E-03,3.860000E-03,3.200000E-03,2.660000E-03,2.220000E-03,1.850000E-03,1.540000E-03};
double THXSEC7TeV_Gluino_Low  [] = {7.553474E+01,5.146393E+01,3.566432E+01,2.512864E+01,1.804830E+01,1.304305E+01,9.556500E+00,7.044302E+00,5.271630E+00,3.988566E+00,3.018432E+00,2.308958E+00,1.773087E+00,1.374688E+00,1.070057E+00,8.414255E-01,6.640146E-01,5.237697E-01,4.145975E-01,3.315876E-01,2.657578E-01,2.134368E-01,1.716366E-01,1.387783E-01,1.127959E-01,9.142058E-02,7.469319E-02,6.086165E-02,4.999024E-02,4.092504E-02,3.378647E-02,2.791464E-02,2.286025E-02,1.896689E-02,1.560576E-02,1.300309E-02,1.082652E-02,8.957356E-03,7.480351E-03,6.255235E-03,5.204033E-03,4.420000E-03,3.730000E-03,3.160000E-03,2.670000E-03,2.270000E-03};
double THXSEC7TeV_Gluino_High [] = {5.742197E+01,3.883067E+01,2.678981E+01,1.876433E+01,1.338010E+01,9.580894E+00,6.973463E+00,5.091237E+00,3.788946E+00,2.845079E+00,2.130996E+00,1.626191E+00,1.243271E+00,9.589442E-01,7.381439E-01,5.751390E-01,4.515518E-01,3.522239E-01,2.787396E-01,2.208681E-01,1.759448E-01,1.397393E-01,1.116011E-01,8.996432E-02,7.195999E-02,5.827633E-02,4.733762E-02,3.809243E-02,3.105941E-02,2.538284E-02,2.054873E-02,1.688133E-02,1.379495E-02,1.136371E-02,9.251707E-03,7.653260E-03,6.325800E-03,5.168580E-03,4.283127E-03,3.515531E-03,2.919556E-03,2.300000E-03,1.890000E-03,1.530000E-03,1.250000E-03,1.000000E-03};

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


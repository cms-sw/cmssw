
#ifndef HSCP_ANALYSIS_SAMPLE
#define HSCP_ANALYSIS_SAMPLE


#define SID_GL200     0
#define SID_GL300     1
#define SID_GL400     2
#define SID_GL500     3
#define SID_GL600     4
#define SID_GL900     5
#define SID_GL200N    6
#define SID_GL300N    7
#define SID_GL400N    8
#define SID_GL500N    9
#define SID_GL600N   10
#define SID_GL900N   11
#define SID_ST130    12
#define SID_ST200    13
#define SID_ST300    14
#define SID_ST500    15
#define SID_ST800    16
#define SID_ST130N   17
#define SID_ST200N   18
#define SID_ST300N   19
#define SID_ST500N   20
#define SID_ST800N   21
#define SID_GS100    22
#define SID_GS126    23
#define SID_GS156    24
#define SID_GS200    25
#define SID_GS247    26
#define SID_GS308    27
#define SID_PS100    28
#define SID_PS126    29
#define SID_PS156    30
#define SID_PS200    31
#define SID_PS247    32
#define SID_PS308    33
#define SID_DS121    34
#define SID_DS182    35
#define SID_DS242    36
#define SID_DS302    37


int                  RunningPeriods = 2;
double               IntegratedLuminosity = 1091.399638766; //976.204518023; //705.273820; //342.603275; //204.160928; //191.04;
double               IntegratedLuminosityBeforeTriggerChange = 337.484034005; // Total luminosity taken before RPC L1 trigger change (went into effect on run 165970)
float                Event_Weight = 1;
int                  MaxEntry = -1;


class stSignal{
   public:
   std::string Type;
   std::string Name;
   std::string FileName;
   std::string Legend;
   double Mass;
   double XSec;
   bool   MakePlot;

   stSignal(); 
   stSignal(std::string Type_, std::string Name_, std::string FileName_, std::string Legend_, double Mass_, bool MakePlot_, double XSec_){Type=Type_; Name=Name_; FileName=FileName_; Legend=Legend_; Mass=Mass_; MakePlot=MakePlot_; XSec=XSec_;}
};


void GetSignalDefinition(std::vector<stSignal>& signals){

// signals.push_back(stSignal("Gluino", "Gluino200"    , "#tilde{g} 200"                 , 200,  1, 606.000000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino300", "Gluino300"    , "#tilde{g} 300"                 , 300,  1,  65.800000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino400", "Gluino400"    , "#tilde{g} 400"                 , 400,  1,   11.20000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino500", "Gluino500"    , "#tilde{g} 500"                 , 500,  1,   2.540000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino600", "Gluino600"    , "#tilde{g} 600"                 , 600,  1,   0.693000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino700", "Gluino700"    , "#tilde{g} 700"                 , 700,  1,   0.214000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino800", "Gluino800"    , "#tilde{g} 800"                 , 800,  1,   0.072500) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino900", "Gluino900"    , "#tilde{g} 900"                 , 900,  1,   0.026200) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1000", "Gluino1000"  , "#tilde{g} 1000"                ,1000,  1,   0.0098700) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1100", "Gluino1100"  , "#tilde{g} 1100"                ,1100,  1,   0.0038600) ); //NLO 
  //signals.push_back(stSignal("Gluino", "Gluino1200", "Gluino1200"   , "#tilde{g} 1200"                ,1200,  1,   0.004300) ); //NLO

  signals.push_back(stSignal("Gluino", "Gluino300N", "Gluinoneutralonly300"   , "#tilde{g} 300 CS"              , 300,  1,  65.800000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino400N", "Gluinoneutralonly400"   , "#tilde{g} 400 CS"              , 400,  1,   11.20000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino500N", "Gluinoneutralonly500"   , "#tilde{g} 500 CS"              , 500,  1,   2.540000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino600N", "Gluinoneutralonly600"   , "#tilde{g} 600 CS"              , 600,  1,   0.693000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino700N", "Gluinoneutralonly700"   , "#tilde{g} 700 CS"              , 700,  1,   0.214000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino800N", "Gluinoneutralonly800"   , "#tilde{g} 800 CS"              , 800,  1,   0.072500) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino900N", "Gluinoneutralonly900"   , "#tilde{g} 900 CS"              , 900,  1,   0.026200) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1000N", "Gluinoneutralonly1000"  , "#tilde{g} 1000 CS"             ,1000,  1,   0.0098700) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1100N", "Gluinoneutralonly1100"  , "#tilde{g} 1100 CS"             ,1100,  1,   0.0038600) ); //NLO
  //signals.push_back(stSignal("Gluino", "Gluino1200N", "Gluinoneutralonly1200.root"  , "#tilde{g} 1200 CS"             ,1200,  1,   ) ); //NLO

   //signals.push_back(stSignal("Gluino", "Gluino600Z"   , "#tilde{g} 600 Z2"              , 600,  1,   0.465000) ); //NLO
   //signals.push_back(stSignal("Gluino", "Gluino700Z"   , "#tilde{g} 700 Z2"              , 700,  1,   0.130000) ); //NLO
   //signals.push_back(stSignal("Gluino", "Gluino800Z"   , "#tilde{g} 800 Z2"              , 800,  1,   0.039600) ); //NLO

  signals.push_back(stSignal("Stop"  , "Stop130", "stop_M-130"      , "#tilde{t}_{1} 130"             , 130,  1, 120.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop200", "stop_M-200"      , "#tilde{t}_{1} 200"             , 200,  1,  13.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop300", "stop_M-300"      , "#tilde{t}_{1} 300"             , 300,  1,   1.310000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop400", "stop_M-400"      , "#tilde{t}_{1} 400"             , 400,  1,   0.218000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop500", "stop_M-500"      , "#tilde{t}_{1} 500"             , 500,  0,   0.047800) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop600", "stop_M-600"      , "#tilde{t}_{1} 600"             , 600,  1,   0.012500) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop700", "stop_M-700"      , "#tilde{t}_{1} 700"             , 700,  1,   0.003560) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop800", "stop_M-800"      , "#tilde{t}_{1} 800"             , 800,  1,   0.001140) ); //NLO

  signals.push_back(stSignal("Stop"  , "Stop130N", "stoponlyneutral_M-130"      , "#tilde{t}_{1} 130 CS"          , 130,  1, 120.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop200N", "stoponlyneutral_M-200"     , "#tilde{t}_{1} 200 CS"          , 200,  1,  13.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop300N", "stoponlyneutral_M-300"     , "#tilde{t}_{1} 300 CS"          , 300,  1,   1.310000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop400N", "stoponlyneutral_M-400"     , "#tilde{t}_{1} 400 CS"          , 400,  1,   0.218000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop500N", "stoponlyneutral_M-500"     , "#tilde{t}_{1} 500 CS"          , 500,  0,   0.047800) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop600N", "stoponlyneutral_M-600"     , "#tilde{t}_{1} 600 CS"          , 600,  1,   0.012500) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop700N", "stoponlyneutral_M-700"     , "#tilde{t}_{1} 700 CS"          , 700,  1,   0.003560) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop800N", "stoponlyneutral_M-800"     , "#tilde{t}_{1} 800 CS"          , 800,  1,   0.001140) ); //NLO

  //signals.push_back(stSignal("Stop"  , "Stop300Z"     , "#tilde{t}_{1} 300 Z2"          , 300,  1,   1.310000) ); //NLO
  //signals.push_back(stSignal("Stop"  , "Stop400Z"     , "#tilde{t}_{1} 400 Z2"          , 400,  1,   0.218000) ); //NLO
  //signals.push_back(stSignal("Stop"  , "Stop500Z"     , "#tilde{t}_{1} 500 Z2"          , 500,  0,   0.047800) ); //NLO

  signals.push_back(stSignal("Stau"  , "GMStau100", "stau_M-100"    , "GMSB #tilde{#tau}_{1} 100"     , 100,  1,   1.326000) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau126", "stau_M-126"    , "GMSB #tilde{#tau}_{1} 126"     , 126,  1,   0.330000) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau156", "stau_M-156"    , "GMSB #tilde{#tau}_{1} 156"     , 156,  0,   0.105000) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau200", "stau_M-200"    , "GMSB #tilde{#tau}_{1} 200"     , 200,  1,   0.025000) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau247", "stau_M-247"    , "GMSB #tilde{#tau}_{1} 247"     , 247,  0,   0.008000) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau308", "stau_M-308"    , "GMSB #tilde{#tau}_{1} 308"     , 308,  1,   0.002000) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau370", "stau_M-370"    , "GMSB #tilde{#tau}_{1} 370"     , 370,  1,   0.0007395) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau432", "stau_M-432"    , "GMSB #tilde{#tau}_{1} 432"     , 432,  1,   0.0002824) ); //LO
  signals.push_back(stSignal("Stau"  , "GMStau494", "stau_M-494"    , "GMSB #tilde{#tau}_{1} 494"     , 494,  1,   0.0001139) ); //LO

  signals.push_back(stSignal("Stau"  , "PPStau100", "PPStau100", "Pair #tilde{#tau}_{1} 100"     , 100,  1,   0.0382) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau126", "PPStau126", "Pair #tilde{#tau}_{1} 126"     , 126,  0,   0.01620) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau156", "PPStau156", "Pair #tilde{#tau}_{1} 156"     , 156,  0,   0.00703) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau200", "PPStau200", "Pair #tilde{#tau}_{1} 200"     , 200,  1,   0.00247) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau247", "PPStau247", "Pair #tilde{#tau}_{1} 247"     , 247,  0,   0.00100) ); //NLO
/*   signals.push_back(stSignal("Stau"  , "PPStau308"    , "Pair #tilde{#tau}_{1} 308"     , 308,  1,   0.000300) ); //LO
   signals.push_back(stSignal("Stau"  , "DCStau121"    , "DICHAMP #tilde{#tau}_{1} 121"  , 121,  1,   0.450000) ); //LO
   signals.push_back(stSignal("Stau"  , "DCStau182"    , "DICHAMP #tilde{#tau}_{1} 182"  , 182,  0,   0.083000) ); //LO
   signals.push_back(stSignal("Stau"  , "DCStau242"    , "DICHAMP #tilde{#tau}_{1} 242"  , 242,  0,   0.022800) ); //LO
   signals.push_back(stSignal("Stau"  , "DCStau302"    , "DICHAMP #tilde{#tau}_{1} 302"  , 302,  1,   0.007700) ); //LO
*/
}

struct stMC{
   std::string Name;
   double ILumi;
   double MaxPtHat;
   double MaxEvent;

   stMC();
   stMC(std::string Name_, double ILumi_, double MaxPtHat_, int MaxEvent_){Name = Name_; ILumi = ILumi_; MaxPtHat = MaxPtHat_; MaxEvent = MaxEvent_;}
};

void GetMCDefinition(std::vector<stMC>& MC){
   MC.push_back(stMC("MC_QCD_30to50"     , 0.12, -1, -1 ));
   MC.push_back(stMC("MC_QCD_50to80"     , 1.0377, -1, -1 ));
   MC.push_back(stMC("MC_QCD_80to120"    , 8.40555612, -1, -1 ));
   MC.push_back(stMC("MC_QCD_120to170"   , 53.28285, -1, -1 ));
   MC.push_back(stMC("MC_QCD_170to300"   , 255.97366, -1, -1 ));
   MC.push_back(stMC("MC_QCD_300to470"   , 5498.0076923, -1, -1 ));
   MC.push_back(stMC("MC_QCD_470to600"   , 56838.8 , -1, -1 ));
   MC.push_back(stMC("MC_QCD_600to800"   , 272159.9 , -1, -1 ));
   MC.push_back(stMC("MC_QCD_800to1000"  , 2203200.000000, -1, -1 ));
   MC.push_back(stMC("MC_QCD_1000to1400" , 6304885.542, -1, -1 ));
   MC.push_back(stMC("MC_QCD_1400to1800" , 201486238.5321, -1, -1 ));
   MC.push_back(stMC("MC_QCD_1800toInf"  , 818824022.3    , -1, -1 ));
   MC.push_back(stMC("MC_DYToTauTau"     , 1563.489, -1, -1 ));
   MC.push_back(stMC("MC_DYToMuMu"       , 1652.55769, -1, -1 ));
   MC.push_back(stMC("MC_WToMuNu"        , 685.309, -1, -1 ));
   MC.push_back(stMC("MC_WToTauNu"       , 696.29, -1, -1 ));
   MC.push_back(stMC("MC_TTBar"          , 11591.76, -1, -1 ));
}

void GetInputFiles(std::vector<std::string>& inputFiles, std::string SampleName, int period=0){
   //std::string BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_3/11_07_14/";
  //std::string BaseDirectory = "dcap://cmsdca.fnal.gov:24125/pnfs/cms/WAX/11/store/user/farrell3/EDMFiles/";
  std::string BaseDirectory = "dcache:/pnfs/cms/WAX/11/store/user/jchen/11_07_14_HSCP2011/";

   if(SampleName=="Data"){
     inputFiles.push_back(BaseDirectory + "Data_RunA_160404_163869.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_165001_166033.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_166034_166500.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_166501_166893.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_166894_167151.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_167153_167913.root");
   }else if(SampleName.find("MC_",0)<std::string::npos){
     inputFiles.push_back(BaseDirectory + SampleName + ".root");
   }else{
     if (period==0) inputFiles.push_back(BaseDirectory + SampleName + ".root");
     if (period==1) inputFiles.push_back(BaseDirectory + SampleName + "BX1.root");
   }
}

#endif

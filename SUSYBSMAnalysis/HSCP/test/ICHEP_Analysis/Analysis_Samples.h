
#ifndef HSCP_ANALYSIS_SAMPLE
#define HSCP_ANALYSIS_SAMPLE


#define SID_GL300     0
#define SID_GL400     1
#define SID_GL500     2
#define SID_GL600     3
#define SID_GL700     4
#define SID_GL800     5
#define SID_GL900     6
#define SID_GL1000    7
#define SID_GL1100    8
#define SID_GL300N    9
#define SID_GL400N    10
#define SID_GL500N    11
#define SID_GL600N    12
#define SID_GL700N    13
#define SID_GL800N    14
#define SID_GL900N    15
#define SID_GL1000N   16
#define SID_GL1100N   17
#define SID_ST130     18
#define SID_ST200     19
#define SID_ST300     20
#define SID_ST400     21
#define SID_ST500     22
#define SID_ST600     23
#define SID_ST700     24
#define SID_ST800     25
#define SID_ST130N    26
#define SID_ST200N    27
#define SID_ST300N    28
#define SID_ST400N    29
#define SID_ST500N    30
#define SID_ST600N    31
#define SID_ST700N    32
#define SID_ST800N    33
#define SID_GS100     34
#define SID_GS126     35
#define SID_GS156     36
#define SID_GS200     37
#define SID_GS247     38
#define SID_GS308     39
#define SID_GS370     40
#define SID_GS432     41
#define SID_GS494     42
#define SID_PS100     43
#define SID_PS126     44
#define SID_PS156     45
#define SID_PS200     46
#define SID_PS247     47
#define SID_PS308     48
#define SID_DS100     49
#define SID_DS121     50
#define SID_DS182     51
#define SID_DS242     52
#define SID_DS302     53
#define SID_DS350     54
#define SID_DS395     55
#define SID_DS420     56
#define SID_DS500     57

int                  RunningPeriods = 2;
double               IntegratedLuminosity = 1947; //1631; //976.204518023; //705.273820; //342.603275; //204.160928; //191.04;
double               IntegratedLuminosityBeforeTriggerChange = 353.494; // Total luminosity taken before RPC L1 trigger change (went into effect on run 165970)
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
   bool   IsS4PileUp;

   stSignal(); 
      stSignal(std::string Type_, std::string Name_, std::string FileName_, std::string Legend_, double Mass_, bool MakePlot_, bool IsS4PileUp_, double XSec_){Type=Type_; Name=Name_; FileName=FileName_; Legend=Legend_; Mass=Mass_; MakePlot=MakePlot_; IsS4PileUp=IsS4PileUp_;XSec=XSec_;}
};


void GetSignalDefinition(std::vector<stSignal>& signals){
  signals.push_back(stSignal("Gluino", "Gluino300", "Gluino300"    , "#tilde{g} 300"                 , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino400", "Gluino400"    , "#tilde{g} 400"                 , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino500", "Gluino500"    , "#tilde{g} 500"                 , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino600", "Gluino600"    , "#tilde{g} 600"                 , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino700", "Gluino700"    , "#tilde{g} 700"                 , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino800", "Gluino800"    , "#tilde{g} 800"                 , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino900", "Gluino900"    , "#tilde{g} 900"                 , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1000", "Gluino1000"  , "#tilde{g} 1000"                ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1100", "Gluino1100"  , "#tilde{g} 1100"                ,1100,  1, 1,   0.0038600) ); //NLO 
  //signals.push_back(stSignal("Gluino", "Gluino1200", "Gluino1200"   , "#tilde{g} 1200"                ,1200,  1, 1,   0.004300) ); //NLO

  signals.push_back(stSignal("Gluino", "Gluino300N", "Gluinoneutralonly300"   , "#tilde{g} 300 CS"              , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino400N", "Gluinoneutralonly400"   , "#tilde{g} 400 CS"              , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino500N", "Gluinoneutralonly500"   , "#tilde{g} 500 CS"              , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino600N", "Gluinoneutralonly600"   , "#tilde{g} 600 CS"              , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino700N", "Gluinoneutralonly700"   , "#tilde{g} 700 CS"              , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino800N", "Gluinoneutralonly800"   , "#tilde{g} 800 CS"              , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino900N", "Gluinoneutralonly900"   , "#tilde{g} 900 CS"              , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1000N", "Gluinoneutralonly1000"  , "#tilde{g} 1000 CS"             ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1100N", "Gluinoneutralonly1100"  , "#tilde{g} 1100 CS"             ,1100,  1, 1,   0.0038600) ); //NLO
  //signals.push_back(stSignal("Gluino", "Gluino1200N", "Gluinoneutralonly1200"  , "#tilde{g} 1200 CS"             ,1200,  1, 1,   ) ); //NLO

   //signals.push_back(stSignal("Gluino", "Gluino600Z"   , "#tilde{g} 600 Z2"              , 600,  1, 1,   0.465000) ); //NLO
   //signals.push_back(stSignal("Gluino", "Gluino700Z"   , "#tilde{g} 700 Z2"              , 700,  1, 1,   0.130000) ); //NLO
   //signals.push_back(stSignal("Gluino", "Gluino800Z"   , "#tilde{g} 800 Z2"              , 800,  1, 1,   0.039600) ); //NLO

  signals.push_back(stSignal("Stop"  , "Stop130", "stop_M-130"      , "#tilde{t}_{1} 130"             , 130,  1, 1, 120.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop200", "stop_M-200"      , "#tilde{t}_{1} 200"             , 200,  1, 1,  13.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop300", "stop_M-300"      , "#tilde{t}_{1} 300"             , 300,  1, 1,   1.310000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop400", "stop_M-400"      , "#tilde{t}_{1} 400"             , 400,  1, 1,   0.218000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop500", "stop_M-500"      , "#tilde{t}_{1} 500"             , 500,  0, 1,  0.047800) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop600", "stop_M-600"      , "#tilde{t}_{1} 600"             , 600,  1, 1,   0.012500) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop700", "stop_M-700"      , "#tilde{t}_{1} 700"             , 700,  1, 1,   0.003560) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop800", "stop_M-800"      , "#tilde{t}_{1} 800"             , 800,  1, 1,   0.001140) ); //NLO

  signals.push_back(stSignal("Stop"  , "Stop130N", "stoponlyneutral_M-130"      , "#tilde{t}_{1} 130 CS"          , 130,  1, 1, 120.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop200N", "stoponlyneutral_M-200"     , "#tilde{t}_{1} 200 CS"          , 200,  1, 1,  13.000000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop300N", "stoponlyneutral_M-300"     , "#tilde{t}_{1} 300 CS"          , 300,  1, 1,   1.310000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop400N", "stoponlyneutral_M-400"     , "#tilde{t}_{1} 400 CS"          , 400,  1, 1,   0.218000) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop500N", "stoponlyneutral_M-500"     , "#tilde{t}_{1} 500 CS"          , 500,  0, 1,  0.047800) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop600N", "stoponlyneutral_M-600"     , "#tilde{t}_{1} 600 CS"          , 600,  1, 1,   0.012500) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop700N", "stoponlyneutral_M-700"     , "#tilde{t}_{1} 700 CS"          , 700,  1, 1,   0.003560) ); //NLO
  signals.push_back(stSignal("Stop"  , "Stop800N", "stoponlyneutral_M-800"     , "#tilde{t}_{1} 800 CS"          , 800,  1, 1,   0.001140) ); //NLO

  //signals.push_back(stSignal("Stop"  , "Stop300Z"     , "#tilde{t}_{1} 300 Z2"          , 300,  1, 1,   1.310000) ); //NLO
  //signals.push_back(stSignal("Stop"  , "Stop400Z"     , "#tilde{t}_{1} 400 Z2"          , 400,  1, 1,   0.218000) ); //NLO
  //signals.push_back(stSignal("Stop"  , "Stop500Z"     , "#tilde{t}_{1} 500 Z2"          , 500,  0,   0.047800) ); //NLO

  signals.push_back(stSignal("Stau"  , "GMStau100", "stau_M-100"    , "GMSB #tilde{#tau}_{1} 100"     , 100,  1, 1,   1.3398) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau126", "stau_M-126"    , "GMSB #tilde{#tau}_{1} 126"     , 126,  1, 1,   0.274591) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau156", "stau_M-156"    , "GMSB #tilde{#tau}_{1} 156"     , 156,  0, 1,  0.0645953) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau200", "stau_M-200"    , "GMSB #tilde{#tau}_{1} 200"     , 200,  1, 1,   0.0118093) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau247", "stau_M-247"    , "GMSB #tilde{#tau}_{1} 247"     , 247,  0, 1,  0.00342512) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau308", "stau_M-308"    , "GMSB #tilde{#tau}_{1} 308"     , 308,  1, 1,  0.00098447 ) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau370", "stau_M-370"    , "GMSB #tilde{#tau}_{1} 370"     , 370,  1, 1,   0.000353388) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau432", "stau_M-432"    , "GMSB #tilde{#tau}_{1} 432"     , 432,  1, 1,   0.000141817) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau494", "stau_M-494"    , "GMSB #tilde{#tau}_{1} 494"     , 494,  1, 1,   0.00006177) ); //NLO

  signals.push_back(stSignal("Stau"  , "PPStau100", "PPStau100", "Pair #tilde{#tau}_{1} 100"     , 100,  1, 1,   0.0382) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau126", "PPStau126", "Pair #tilde{#tau}_{1} 126"     , 126,  0, 1,  0.0161) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau156", "PPStau156", "Pair #tilde{#tau}_{1} 156"     , 156,  0, 1,  0.00704) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau200", "PPStau200", "Pair #tilde{#tau}_{1} 200"     , 200,  1, 1,   0.00247) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau247", "PPStau247", "Pair #tilde{#tau}_{1} 247"     , 247,  0, 1,  0.00101) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau308", "PPStau308", "Pair #tilde{#tau}_{1} 308"     , 308,  0, 1,  0.000353) ); //NLO  

   signals.push_back(stSignal("Stau" , "DCStau100" , "DCStau100"    , "DICHAMP #tilde{#tau}_{1} 100"  , 100,  1, 1,   1.405000) ); //LO
   signals.push_back(stSignal("Stau" , "DCStau121" , "DCStau121"    , "DICHAMP #tilde{#tau}_{1} 121"  , 121,  1, 1,   0.979000) ); //LO
   signals.push_back(stSignal("Stau" , "DCStau182"  , "DCStau182"    , "DICHAMP #tilde{#tau}_{1} 182"  , 182,  0, 1,   0.560000) ); //LO
   signals.push_back(stSignal("Stau" , "DCStau242" , "DCStau242"    , "DICHAMP #tilde{#tau}_{1} 242"  , 242,  0, 1,  0.489000) ); //LO
   signals.push_back(stSignal("Stau"  , "DCStau302"  , "DCStau302"    , "DICHAMP #tilde{#tau}_{1} 302"  , 302,  1, 1,   0.463000) ); //LO
   signals.push_back(stSignal("Stau" , "DCStau350" , "DCStau350"    , "DICHAMP #tilde{#tau}_{1} 350"  , 350,  1, 1,   0.473000) ); //LO
   signals.push_back(stSignal("Stau" , "DCStau395"  , "DCStau395"    , "DICHAMP #tilde{#tau}_{1} 395"  , 395,  1, 1,   0.420000) ); //LO
   signals.push_back(stSignal("Stau" , "DCStau420"  , "DCStau420"    , "DICHAMP #tilde{#tau}_{1} 420"  , 420,  1, 1,   0.003500) ); //LO
   signals.push_back(stSignal("Stau" , "DCStau500" , "DCStau500"    , "DICHAMP #tilde{#tau}_{1} 500"  , 500,  1, 1,   0.0002849) ); //LO

}

struct stMC{
   std::string Name;
   double XSection;
   double MaxPtHat;
   double MaxEvent;
   bool   IsS4PileUp;

   stMC();
      stMC(std::string Name_, double XSection_, double MaxPtHat_, int MaxEvent_, bool IsS4PileUp_){Name = Name_; XSection = XSection_; MaxPtHat = MaxPtHat_; MaxEvent = MaxEvent_;IsS4PileUp = IsS4PileUp_;}
};

void GetMCDefinition(std::vector<stMC>& MC){

   MC.push_back(stMC("MC_DYToTauTau"            ,     1.300E3  , -1, -1, 0));
   MC.push_back(stMC("MC_DYToMuMu"              ,     1.300E3  , -1, -1, 0));
   MC.push_back(stMC("MC_WJetsToLNu"            ,     2.777E4  , -1, -1, 1));
   MC.push_back(stMC("MC_TTJets"                 ,     9.400E1  , -1, -1, 1));
   MC.push_back(stMC("MC_QCD_Pt-30to50"         ,     5.310E7  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-50to80"         ,     6.360E6  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-80to120"        ,     7.840E5  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-120to170"       ,     1.150E5  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-170to300"       ,     2.430E4  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-300to470"       ,     1.170E3  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-470to600"       ,     7.020E1  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-600to800"       ,     1.560E1  , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-800to1000"      ,     1.84     , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-1000to1400"     ,     3.320E-1 , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-1400to1800"     ,     1.090E-2 , -1, -1, 0));
   MC.push_back(stMC("MC_QCD_Pt-1800"           ,     3.580E-4 , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-0to15"   ,     4.280E3  , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-15to20"  ,     1.450E2  , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-20to30"  ,     1.310E2  , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-30to50"  ,     8.400E1  , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-50to80"  ,     3.220E1  , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-80to120" ,     9.98     , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-120to170",     2.73     , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-170to230",     7.21E-1  , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-230to300",     1.94E-1  , -1, -1, 0));
   MC.push_back(stMC("MC_ZJetToMuMu_Pt-300"     ,     7.59E-2  , -1, -1, 0));
   MC.push_back(stMC("MC_ZZ"                    ,     4.287    , -1, -1, 1));
   MC.push_back(stMC("MC_WW"                    ,     2.783E1  , -1, -1, 1));
   MC.push_back(stMC("MC_WZ"                    ,     1.47E1   , -1, -1, 1));
}

void GetInputFiles(std::vector<std::string>& inputFiles, std::string SampleName, int period=0){
//  std::string BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_3/11_08_03/";
//   std::string BaseDirectory = "dcache:/pnfs/cms/WAX/11/store/user/jchen/11_08_03_HSCP2011EDM/";
   std::string BaseDirectory = "/uscmst1b_scratch/lpc1/lpcphys/jchen/HSCPEDM_08_02_11/";
   if(SampleName=="Data"){
     inputFiles.push_back(BaseDirectory + "Data_RunA_160404_163869.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_165001_166033.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_166034_166500.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_166501_166893.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_166894_167151.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_167153_167913.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_170826_171500.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_171501_172619.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_172620_172790.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_172791_172802.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_172803_172900.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_172901_173243.root");
   }else if(SampleName.find("MC_",0)<std::string::npos){
     inputFiles.push_back(BaseDirectory + SampleName + ".root");
   }else{
     if (period==0) inputFiles.push_back(BaseDirectory + SampleName + ".root");
     if (period==1) inputFiles.push_back(BaseDirectory + SampleName + "BX1.root");
   }
}

#endif

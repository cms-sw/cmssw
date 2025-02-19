
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
#define SID_D08K100   49
#define SID_D08K121   50
#define SID_D08K182   51
#define SID_D08K242   52
#define SID_D08K302   53
#define SID_D08K350   54
#define SID_D08K370   55
#define SID_D08K390   56
#define SID_D08K395   57
#define SID_D08K400   58
#define SID_D08K410   59
#define SID_D08K420   60
#define SID_D08K500   61
#define SID_D12K100   62
#define SID_D12K182   63
#define SID_D12K302   64
#define SID_D12K500   65
#define SID_D12K530   66
#define SID_D12K570   67
#define SID_D12K590   68
#define SID_D12K595   69
#define SID_D12K600   70
#define SID_D12K610   71
#define SID_D12K620   72
#define SID_D12K700   73
#define SID_D16K100   74
#define SID_D16K182   75
#define SID_D16K302   76
#define SID_D16K500   77
#define SID_D16K700   78
#define SID_D16K730   79
#define SID_D16K770   80
#define SID_D16K790   81
#define SID_D16K795   82
#define SID_D16K800   83
#define SID_D16K820   84
#define SID_D16K900   85


int                  RunningPeriods = 2;
double               IntegratedLuminosity = 4679;//2125; //2080; //1912; //1947; //1631; //976.204518023; //705.273820; //342.603275; //204.160928; //191.04;
double               IntegratedLuminosityBeforeTriggerChange = 353.494; // Total luminosity taken before RPC L1 trigger change (went into effect on run 165970)
float                Event_Weight = 1;
int                  MaxEntry = 10000;


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
  signals.push_back(stSignal("Gluino", "Gluino300S", "Gluino300S"    , "#tilde{g} 300S"                 , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino600", "Gluino600"    , "#tilde{g} 600"                 , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino600S", "Gluino600S"    , "#tilde{g} 600S"                 , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal("Gluino", "Gluino1100", "Gluino1100"  , "#tilde{g} 1100"                ,1100,  1, 1,   0.0038600) ); //NLO 
  signals.push_back(stSignal("Gluino", "Gluino1100S", "Gluino1100S"  , "#tilde{g} 1100S"                ,1100,  1, 1,   0.0038600) ); //NLO 

  signals.push_back(stSignal("Stau"  , "GMStau100", "stau_M-100"    , "GMSB #tilde{#tau}_{1} 100"     , 100,  1, 1,   1.3398) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau100S", "stau_M-100S"    , "GMSB #tilde{#tau}_{1} 100S"     , 100,  1, 1,   1.3398) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau200", "stau_M-200"    , "GMSB #tilde{#tau}_{1} 200"     , 200,  1, 1,   0.0118093) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau200S", "stau_M-200S"    , "GMSB #tilde{#tau}_{1} 200S"     , 200,  1, 1,   0.0118093) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau308", "stau_M-308"    , "GMSB #tilde{#tau}_{1} 308"     , 308,  1, 1,  0.00098447 ) ); //NLO
  signals.push_back(stSignal("Stau"  , "GMStau308S", "stau_M-308S"    , "GMSB #tilde{#tau}_{1} 308S"     , 308,  1, 1,  0.00098447 ) ); //NLO

  signals.push_back(stSignal("Stau"  , "PPStau100", "PPStau100", "Pair #tilde{#tau}_{1} 100"     , 100,  1, 1,   0.0382) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau100S", "PPStau100S", "Pair #tilde{#tau}_{1} 100S"     , 100,  1, 1,   0.0382) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau200", "PPStau200", "Pair #tilde{#tau}_{1} 200"     , 200,  1, 1,   0.00247) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau200S", "PPStau200S", "Pair #tilde{#tau}_{1} 200S"     , 200,  1, 1,   0.00247) ); //NLO
  signals.push_back(stSignal("Stau"  , "PPStau308", "PPStau308", "Pair #tilde{#tau}_{1} 308"     , 308,  0, 1,  0.000353) ); //NLO  
  signals.push_back(stSignal("Stau"  , "PPStau308S", "PPStau308S", "Pair #tilde{#tau}_{1} 308S"     , 308,  0, 1,  0.000353) ); //NLO  


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
   MC.push_back(stMC("MC_TTJets"                ,     9.400E1  , -1, -1, 1));
   MC.push_back(stMC("MC_QCD_Pt-15to30"         ,     8.16E8  , -1, -1, 0));
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
//   std::string BaseDirectory = "dcache:/pnfs/cms/WAX/11/store/user/jchen/11_09_13_HSCP2011EDM/";
  std::string BaseDirectory = "/uscmst1b_scratch/lpc1/lpcphys/jchen/HSCPEDM_11_01_11/";
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
     inputFiles.push_back(BaseDirectory + "Data_RunA_173244_173692.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_175860_176099.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_176100_176309.root");
   }else if(SampleName.find("MC_",0)<std::string::npos){
     inputFiles.push_back(BaseDirectory + SampleName + ".root");
   }else{
     if (period==0) inputFiles.push_back(BaseDirectory + SampleName + ".root");
     if (period==1) inputFiles.push_back(BaseDirectory + SampleName + "BX1.root");
   }
}

#endif

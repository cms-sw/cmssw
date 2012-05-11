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
#define SID_GL1200    9
#define SID_GL300N    10
#define SID_GL400N    11
#define SID_GL500N    12
#define SID_GL600N    13
#define SID_GL700N    14
#define SID_GL800N    15
#define SID_GL900N    16
#define SID_GL1000N   17
#define SID_GL1100N   18
#define SID_GL1200N   19
#define SID_ST130     20
#define SID_ST200     21
#define SID_ST300     22
#define SID_ST400     23
#define SID_ST500     24
#define SID_ST600     25
#define SID_ST700     26
#define SID_ST800     27
#define SID_ST130N    28
#define SID_ST200N    29
#define SID_ST300N    30
#define SID_ST400N    31
#define SID_ST500N    32
#define SID_ST600N    33
#define SID_ST700N    34
#define SID_ST800N    35
#define SID_GS100     36
#define SID_GS126     37
#define SID_GS156     38
#define SID_GS200     39
#define SID_GS247     40
#define SID_GS308     41
#define SID_GS370     42
#define SID_GS432     43
#define SID_GS494     44
#define SID_PS100     45
#define SID_PS126     46
#define SID_PS156     47
#define SID_PS200     48
#define SID_PS247     49
#define SID_PS308     50
#define SID_D08K100   51
#define SID_D08K121   52
#define SID_D08K182   53
#define SID_D08K242   54
#define SID_D08K302   55
#define SID_D08K350   56
#define SID_D08K370   57
#define SID_D08K390   58
#define SID_D08K395   59
#define SID_D08K400   60
#define SID_D08K410   61
#define SID_D08K420   62
#define SID_D08K500   63
#define SID_D12K100   64
#define SID_D12K182   65
#define SID_D12K302   66
#define SID_D12K500   67
#define SID_D12K530   68
#define SID_D12K570   69
#define SID_D12K590   70
#define SID_D12K595   71
#define SID_D12K600   72
#define SID_D12K610   73
#define SID_D12K620   74
#define SID_D12K700   75
#define SID_D16K100   76
#define SID_D16K182   77
#define SID_D16K302   78
#define SID_D16K500   79
#define SID_D16K700   80
#define SID_D16K730   81
#define SID_D16K770   82
#define SID_D16K790   83
#define SID_D16K795   84
#define SID_D16K800   85
#define SID_D16K820   86
#define SID_D16K900   87



//This code is there to enable/disable year dependent code
//#define ANALYSIS2011

#ifdef ANALYSIS2011
int                  RunningPeriods = 2;
double               IntegratedLuminosity = 4976; //3168; //2410;//2125; //2080; //1912; //1947; //1631; //976.204518023; //705.273820; //342.603275; //204.160928; //191.04;
double               IntegratedLuminosityBeforeTriggerChange = 355.227; //353.494; // Total luminosity taken before RPC L1 trigger change (went into effect on run 165970)
#else
int                  RunningPeriods = 1;
double               IntegratedLuminosity = 500;
double               IntegratedLuminosityBeforeTriggerChange = 0;
#endif


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
   int    NChargedHSCP;

   stSignal(); 
   stSignal(int NChargedHSCP_, std::string Type_, std::string Name_, std::string FileName_, std::string Legend_, double Mass_, bool MakePlot_, bool IsS4PileUp_, double XSec_){NChargedHSCP=NChargedHSCP_; Type=Type_; Name=Name_; FileName=FileName_; Legend=Legend_; Mass=Mass_; MakePlot=MakePlot_; IsS4PileUp=IsS4PileUp_;XSec=XSec_; }
};


void GetSignalDefinition(std::vector<stSignal>& signals, bool TkOnly=true){
#ifdef ANALYSIS2011
  //2011 7TeV Signals
  signals.push_back(stSignal(-1,"Gluino", "Gluino300", "Gluino300"    , "MC - #tilde{g} 300 GeV/#font[12]{c}^{2}"                 , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino400", "Gluino400"    , "MC - #tilde{g} 400 GeV/#font[12]{c}^{2}"                 , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino500", "Gluino500"    , "MC - #tilde{g} 500 GeV/#font[12]{c}^{2}"                 , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino600", "Gluino600"    , "MC - #tilde{g} 600 GeV/#font[12]{c}^{2}"                 , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino700", "Gluino700"    , "MC - #tilde{g} 700 GeV/#font[12]{c}^{2}"                 , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino800", "Gluino800"    , "MC - #tilde{g} 800 GeV/#font[12]{c}^{2}"                 , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino900", "Gluino900"    , "MC - #tilde{g} 900 GeV/#font[12]{c}^{2}"                 , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino1000", "Gluino1000"  , "MC - #tilde{g} 1000 GeV/#font[12]{c}^{2}"                ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino1100", "Gluino1100"  , "MC - #tilde{g} 1100 GeV/#font[12]{c}^{2}"                ,1100,  1, 1,   0.0038600) ); //NLO 
  signals.push_back(stSignal(-1,"Gluino", "Gluino1200", "Gluino1200"  , "MC - #tilde{g} 1200 GeV/#font[12]{c}^{2}"                ,1200,  1, 1,   0.0015400) ); //NLO

  signals.push_back(stSignal( 0,"Gluino", "Gluino300_NC0", "Gluino300"    , "MC - #tilde{g} 300 GeV/#font[12]{c}^{2}"                 , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino400_NC0", "Gluino400"    , "MC - #tilde{g} 400 GeV/#font[12]{c}^{2}"                 , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino500_NC0", "Gluino500"    , "MC - #tilde{g} 500 GeV/#font[12]{c}^{2}"                 , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino600_NC0", "Gluino600"    , "MC - #tilde{g} 600 GeV/#font[12]{c}^{2}"                 , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino700_NC0", "Gluino700"    , "MC - #tilde{g} 700 GeV/#font[12]{c}^{2}"                 , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino800_NC0", "Gluino800"    , "MC - #tilde{g} 800 GeV/#font[12]{c}^{2}"                 , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino900_NC0", "Gluino900"    , "MC - #tilde{g} 900 GeV/#font[12]{c}^{2}"                 , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino1000_NC0", "Gluino1000"  , "MC - #tilde{g} 1000 GeV/#font[12]{c}^{2}"                ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino1100_NC0", "Gluino1100"  , "MC - #tilde{g} 1100 GeV/#font[12]{c}^{2}"                ,1100,  1, 1,   0.0038600) ); //NLO 
  signals.push_back(stSignal( 0,"Gluino", "Gluino1200_NC0", "Gluino1200"  , "MC - #tilde{g} 1200 GeV/#font[12]{c}^{2}"                ,1200,  1, 1,   0.0015400) ); //NLO

  signals.push_back(stSignal( 1,"Gluino", "Gluino300_NC1", "Gluino300"    , "MC - #tilde{g} 300 GeV/#font[12]{c}^{2}"                 , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino400_NC1", "Gluino400"    , "MC - #tilde{g} 400 GeV/#font[12]{c}^{2}"                 , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino500_NC1", "Gluino500"    , "MC - #tilde{g} 500 GeV/#font[12]{c}^{2}"                 , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino600_NC1", "Gluino600"    , "MC - #tilde{g} 600 GeV/#font[12]{c}^{2}"                 , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino700_NC1", "Gluino700"    , "MC - #tilde{g} 700 GeV/#font[12]{c}^{2}"                 , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino800_NC1", "Gluino800"    , "MC - #tilde{g} 800 GeV/#font[12]{c}^{2}"                 , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino900_NC1", "Gluino900"    , "MC - #tilde{g} 900 GeV/#font[12]{c}^{2}"                 , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino1000_NC1", "Gluino1000"  , "MC - #tilde{g} 1000 GeV/#font[12]{c}^{2}"                ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino1100_NC1", "Gluino1100"  , "MC - #tilde{g} 1100 GeV/#font[12]{c}^{2}"                ,1100,  1, 1,   0.0038600) ); //NLO 
  signals.push_back(stSignal( 1,"Gluino", "Gluino1200_NC1", "Gluino1200"  , "MC - #tilde{g} 1200 GeV/#font[12]{c}^{2}"                ,1200,  1, 1,   0.0015400) ); //NLO

  signals.push_back(stSignal( 2,"Gluino", "Gluino300_NC2", "Gluino300"    , "MC - #tilde{g} 300 GeV/#font[12]{c}^{2}"                 , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino400_NC2", "Gluino400"    , "MC - #tilde{g} 400 GeV/#font[12]{c}^{2}"                 , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino500_NC2", "Gluino500"    , "MC - #tilde{g} 500 GeV/#font[12]{c}^{2}"                 , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino600_NC2", "Gluino600"    , "MC - #tilde{g} 600 GeV/#font[12]{c}^{2}"                 , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino700_NC2", "Gluino700"    , "MC - #tilde{g} 700 GeV/#font[12]{c}^{2}"                 , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino800_NC2", "Gluino800"    , "MC - #tilde{g} 800 GeV/#font[12]{c}^{2}"                 , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino900_NC2", "Gluino900"    , "MC - #tilde{g} 900 GeV/#font[12]{c}^{2}"                 , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino1000_NC2", "Gluino1000"  , "MC - #tilde{g} 1000 GeV/#font[12]{c}^{2}"                ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino1100_NC2", "Gluino1100"  , "MC - #tilde{g} 1100 GeV/#font[12]{c}^{2}"                ,1100,  1, 1,   0.0038600) ); //NLO 
  signals.push_back(stSignal( 2,"Gluino", "Gluino1200_NC2", "Gluino1200"  , "MC - #tilde{g} 1200 GeV/#font[12]{c}^{2}"                ,1200,  1, 1,   0.0015400) ); //NLO

  if(TkOnly) {
  signals.push_back(stSignal(-1,"Gluino", "Gluino300N", "Gluino300N"   , "#tilde{g} 300 GeV/#font[12]{c}^{2} CS"              , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino400N", "Gluino400N"   , "#tilde{g} 400 GeV/#font[12]{c}^{2} CS"              , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino500N", "Gluino500N"   , "#tilde{g} 500 GeV/#font[12]{c}^{2} CS"              , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino600N", "Gluino600N"   , "#tilde{g} 600 GeV/#font[12]{c}^{2} CS"              , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino700N", "Gluino700N"   , "#tilde{g} 700 GeV/#font[12]{c}^{2} CS"              , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino800N", "Gluino800N"   , "#tilde{g} 800 GeV/#font[12]{c}^{2} CS"              , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino900N", "Gluino900N"   , "#tilde{g} 900 GeV/#font[12]{c}^{2} CS"              , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino1000N", "Gluino1000N" , "#tilde{g} 1000 GeV/#font[12]{c}^{2} CS"             ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino1100N", "Gluino1100N" , "#tilde{g} 1100 GeV/#font[12]{c}^{2} CS"             ,1100,  1, 1,   0.0038600) ); //NLO
  signals.push_back(stSignal(-1,"Gluino", "Gluino1200N", "Gluino1200N" , "#tilde{g} 1200 GeV/#font[12]{c}^{2} CS"             ,1200,  1, 1,   0.0015400) ); //NLO

  signals.push_back(stSignal( 0,"Gluino", "Gluino300N_NC0", "Gluino300N"   , "#tilde{g} 300 GeV/#font[12]{c}^{2} CS"              , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino400N_NC0", "Gluino400N"   , "#tilde{g} 400 GeV/#font[12]{c}^{2} CS"              , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino500N_NC0", "Gluino500N"   , "#tilde{g} 500 GeV/#font[12]{c}^{2} CS"              , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino600N_NC0", "Gluino600N"   , "#tilde{g} 600 GeV/#font[12]{c}^{2} CS"              , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino700N_NC0", "Gluino700N"   , "#tilde{g} 700 GeV/#font[12]{c}^{2} CS"              , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino800N_NC0", "Gluino800N"   , "#tilde{g} 800 GeV/#font[12]{c}^{2} CS"              , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino900N_NC0", "Gluino900N"   , "#tilde{g} 900 GeV/#font[12]{c}^{2} CS"              , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino1000N_NC0", "Gluino1000N" , "#tilde{g} 1000 GeV/#font[12]{c}^{2} CS"             ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino1100N_NC0", "Gluino1100N" , "#tilde{g} 1100 GeV/#font[12]{c}^{2} CS"             ,1100,  1, 1,   0.0038600) ); //NLO
  signals.push_back(stSignal( 0,"Gluino", "Gluino1200N_NC0", "Gluino1200N" , "#tilde{g} 1200 GeV/#font[12]{c}^{2} CS"             ,1200,  1, 1,   0.0015400) ); //NLO

  signals.push_back(stSignal( 1,"Gluino", "Gluino300N_NC1", "Gluino300N"   , "#tilde{g} 300 GeV/#font[12]{c}^{2} CS"              , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino400N_NC1", "Gluino400N"   , "#tilde{g} 400 GeV/#font[12]{c}^{2} CS"              , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino500N_NC1", "Gluino500N"   , "#tilde{g} 500 GeV/#font[12]{c}^{2} CS"              , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino600N_NC1", "Gluino600N"   , "#tilde{g} 600 GeV/#font[12]{c}^{2} CS"              , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino700N_NC1", "Gluino700N"   , "#tilde{g} 700 GeV/#font[12]{c}^{2} CS"              , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino800N_NC1", "Gluino800N"   , "#tilde{g} 800 GeV/#font[12]{c}^{2} CS"              , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino900N_NC1", "Gluino900N"   , "#tilde{g} 900 GeV/#font[12]{c}^{2} CS"              , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino1000N_NC1", "Gluino1000N" , "#tilde{g} 1000 GeV/#font[12]{c}^{2} CS"             ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino1100N_NC1", "Gluino1100N" , "#tilde{g} 1100 GeV/#font[12]{c}^{2} CS"             ,1100,  1, 1,   0.0038600) ); //NLO
  signals.push_back(stSignal( 1,"Gluino", "Gluino1200N_NC1", "Gluino1200N" , "#tilde{g} 1200 GeV/#font[12]{c}^{2} CS"             ,1200,  1, 1,   0.0015400) ); //NLO

  signals.push_back(stSignal( 2,"Gluino", "Gluino300N_NC2", "Gluino300N"   , "#tilde{g} 300 GeV/#font[12]{c}^{2} CS"              , 300,  1, 1,  65.800000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino400N_NC2", "Gluino400N"   , "#tilde{g} 400 GeV/#font[12]{c}^{2} CS"              , 400,  1, 1,   11.20000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino500N_NC2", "Gluino500N"   , "#tilde{g} 500 GeV/#font[12]{c}^{2} CS"              , 500,  1, 1,   2.540000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino600N_NC2", "Gluino600N"   , "#tilde{g} 600 GeV/#font[12]{c}^{2} CS"              , 600,  1, 1,   0.693000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino700N_NC2", "Gluino700N"   , "#tilde{g} 700 GeV/#font[12]{c}^{2} CS"              , 700,  1, 1,   0.214000) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino800N_NC2", "Gluino800N"   , "#tilde{g} 800 GeV/#font[12]{c}^{2} CS"              , 800,  1, 1,   0.072500) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino900N_NC2", "Gluino900N"   , "#tilde{g} 900 GeV/#font[12]{c}^{2} CS"              , 900,  1, 1,   0.026200) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino1000N_NC2", "Gluino1000N" , "#tilde{g} 1000 GeV/#font[12]{c}^{2} CS"             ,1000,  1, 1,   0.0098700) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino1100N_NC2", "Gluino1100N" , "#tilde{g} 1100 GeV/#font[12]{c}^{2} CS"             ,1100,  1, 1,   0.0038600) ); //NLO
  signals.push_back(stSignal( 2,"Gluino", "Gluino1200N_NC2", "Gluino1200N" , "#tilde{g} 1200 GeV/#font[12]{c}^{2} CS"             ,1200,  1, 1,   0.0015400) ); //NLO

  }

  signals.push_back(stSignal(-1,"Stop"  , "Stop130", "Stop130"      , "#tilde{t}_{1} 130 GeV/#font[12]{c}^{2}"             , 130,  1, 1, 120.000000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop200", "Stop200"      , "#tilde{t}_{1} 200 GeV/#font[12]{c}^{2}"             , 200,  1, 1,  13.000000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop300", "Stop300"      , "#tilde{t}_{1} 300 GeV/#font[12]{c}^{2}"             , 300,  1, 1,   1.310000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop400", "Stop400"      , "#tilde{t}_{1} 400 GeV/#font[12]{c}^{2}"             , 400,  1, 1,   0.218000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop500", "Stop500"      , "#tilde{t}_{1} 500 GeV/#font[12]{c}^{2}"             , 500,  0, 1,  0.047800) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop600", "Stop600"      , "#tilde{t}_{1} 600 GeV/#font[12]{c}^{2}"             , 600,  1, 1,   0.012500) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop700", "Stop700"      , "#tilde{t}_{1} 700 GeV/#font[12]{c}^{2}"             , 700,  1, 1,   0.003560) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop800", "Stop800"      , "#tilde{t}_{1} 800 GeV/#font[12]{c}^{2}"             , 800,  1, 1,   0.001140) ); //NLO
  if(TkOnly) {
  signals.push_back(stSignal(-1,"Stop"  , "Stop130N", "Stop130N"      , "#tilde{t}_{1} 130 GeV/#font[12]{c}^{2} CS"          , 130,  1, 1, 120.000000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop200N", "Stop200N"     , "#tilde{t}_{1} 200 GeV/#font[12]{c}^{2} CS"          , 200,  1, 1,  13.000000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop300N", "Stop300N"     , "#tilde{t}_{1} 300 GeV/#font[12]{c}^{2} CS"          , 300,  1, 1,   1.310000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop400N", "Stop400N"     , "#tilde{t}_{1} 400 GeV/#font[12]{c}^{2} CS"          , 400,  1, 1,   0.218000) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop500N", "Stop500N"     , "#tilde{t}_{1} 500 GeV/#font[12]{c}^{2} CS"          , 500,  0, 1,  0.047800) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop600N", "Stop600N"     , "#tilde{t}_{1} 600 GeV/#font[12]{c}^{2} CS"          , 600,  1, 1,   0.012500) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop700N", "Stop700N"     , "#tilde{t}_{1} 700 GeV/#font[12]{c}^{2} CS"          , 700,  1, 1,   0.003560) ); //NLO
  signals.push_back(stSignal(-1,"Stop"  , "Stop800N", "Stop800N"     , "#tilde{t}_{1} 800 GeV/#font[12]{c}^{2} CS"          , 800,  1, 1,   0.001140) ); //NLO
  }

  signals.push_back(stSignal(-1,"Stau"  , "GMStau100", "GMStau100"    , "MC - GMSB #tilde{#tau}_{1} 100 GeV/#font[12]{c}^{2}"     , 100,  1, 1,   1.3398) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau126", "GMStau126"    , "MC - GMSB #tilde{#tau}_{1} 126 GeV/#font[12]{c}^{2}"     , 126,  1, 1,   0.274591) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau156", "GMStau156"    , "MC - GMSB #tilde{#tau}_{1} 156 GeV/#font[12]{c}^{2}"     , 156,  0, 1,  0.0645953) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau200", "GMStau200"    , "MC - GMSB #tilde{#tau}_{1} 200 GeV/#font[12]{c}^{2}"     , 200,  1, 1,   0.0118093) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau247", "GMStau247"    , "MC - GMSB #tilde{#tau}_{1} 247 GeV/#font[12]{c}^{2}"     , 247,  0, 1,  0.00342512) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau308", "GMStau308"    , "MC - GMSB #tilde{#tau}_{1} 308 GeV/#font[12]{c}^{2}"     , 308,  1, 1,  0.00098447 ) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau370", "GMStau370"    , "MC - GMSB #tilde{#tau}_{1} 370 GeV/#font[12]{c}^{2}"     , 370,  1, 1,   0.000353388) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau432", "GMStau432"    , "MC - GMSB #tilde{#tau}_{1} 432 GeV/#font[12]{c}^{2}"     , 432,  1, 1,   0.000141817) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "GMStau494", "GMStau494"    , "MC - GMSB #tilde{#tau}_{1} 494 GeV/#font[12]{c}^{2}"     , 494,  1, 1,   0.00006177) ); //NLO

  signals.push_back(stSignal(-1,"Stau"  , "PPStau100", "PPStau100", "Pair #tilde{#tau}_{1} 100 GeV/#font[12]{c}^{2}"     , 100,  1, 1,   0.0382) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "PPStau126", "PPStau126", "Pair #tilde{#tau}_{1} 126 GeV/#font[12]{c}^{2}"     , 126,  0, 1,  0.0161) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "PPStau156", "PPStau156", "Pair #tilde{#tau}_{1} 156 GeV/#font[12]{c}^{2}"     , 156,  0, 1,  0.00704) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "PPStau200", "PPStau200", "Pair #tilde{#tau}_{1} 200 GeV/#font[12]{c}^{2}"     , 200,  1, 1,   0.00247) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "PPStau247", "PPStau247", "Pair #tilde{#tau}_{1} 247 GeV/#font[12]{c}^{2}"     , 247,  0, 1,  0.00101) ); //NLO
  signals.push_back(stSignal(-1,"Stau"  , "PPStau308", "PPStau308", "Pair #tilde{#tau}_{1} 308 GeV/#font[12]{c}^{2}"     , 308,  0, 1,  0.000353) ); //NLO  

  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK100" , "DCRho08HyperK100"    , "DICHAMP #tilde{K} 100 GeV/#font[12]{c}^{2}"  , 100,  1, 1,   1.405000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK121" , "DCRho08HyperK121"    , "DICHAMP #tilde{K} 121 GeV/#font[12]{c}^{2}"  , 121,  1, 1,   0.979000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK182" , "DCRho08HyperK182"    , "DICHAMP #tilde{K} 182 GeV/#font[12]{c}^{2}"  , 182,  0, 1,   0.560000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK242" , "DCRho08HyperK242"    , "DICHAMP #tilde{K} 242 GeV/#font[12]{c}^{2}"  , 242,  0, 1,   0.489000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK302" , "DCRho08HyperK302"    , "DICHAMP #tilde{K} 302 GeV/#font[12]{c}^{2}"  , 302,  1, 1,   0.463000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK350" , "DCRho08HyperK350"    , "DICHAMP #tilde{K} 350 GeV/#font[12]{c}^{2}"  , 350,  1, 1,   0.473000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK370" , "DCRho08HyperK370"    , "DICHAMP #tilde{K} 370 GeV/#font[12]{c}^{2}"  , 370,  1, 1,   0.48288105) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK390" , "DCRho08HyperK390"    , "DICHAMP #tilde{K} 390 GeV/#font[12]{c}^{2}"  , 390,  1, 1,   0.47132496) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK395" , "DCRho08HyperK395"    , "DICHAMP #tilde{K} 395 GeV/#font[12]{c}^{2}"  , 395,  1, 1,   0.420000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK400" , "DCRho08HyperK400"    , "DICHAMP #tilde{K} 400 GeV/#font[12]{c}^{2}"  , 400,  1, 1,   0.473000) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK410" , "DCRho08HyperK410"    , "DICHAMP #tilde{K} 410 GeV/#font[12]{c}^{2}"  , 410,  1, 1,   0.0060812129) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK420" , "DCRho08HyperK420"    , "DICHAMP #tilde{K} 420 GeV/#font[12]{c}^{2}"  , 420,  1, 1,   0.003500) ); //LO
  signals.push_back(stSignal(-1,"DCRho08HyperK" , "DCRho08HyperK500" , "DCRho08HyperK500"    , "DICHAMP #tilde{K} 500 GeV/#font[12]{c}^{2}"  , 500,  1, 1,   0.0002849) ); //LO
  
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK100" , "DCRho12HyperK100"  , "DICHAMP #tilde{K} 100 GeV/#font[12]{c}^{2}"  , 100,  1, 1, 0.8339415992) ); //LO        
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK182" , "DCRho12HyperK182"  , "DICHAMP #tilde{K} 182 GeV/#font[12]{c}^{2}"  , 182,  1, 1, 0.168096952140) ); //LO 
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK302" , "DCRho12HyperK302"  , "DICHAMP #tilde{K} 302 GeV/#font[12]{c}^{2}"  , 302,  1, 1, 0.079554948387) ); //LO      
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK500" , "DCRho12HyperK500"  , "DICHAMP #tilde{K} 500 GeV/#font[12]{c}^{2}"  , 500,  1, 1, 0.063996737) ); //LO         
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK530" , "DCRho12HyperK530"  , "DICHAMP #tilde{K} 530 GeV/#font[12]{c}^{2}"  , 530,  1, 1, 0.064943882) ); //LO         
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK570" , "DCRho12HyperK570"  , "DICHAMP #tilde{K} 570 GeV/#font[12]{c}^{2}"  , 570,  1, 1, 0.0662920530) ); //LO        
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK590" , "DCRho12HyperK590"  , "DICHAMP #tilde{K} 590 GeV/#font[12]{c}^{2}"  , 590,  1, 1, 0.060748383) ); //LO         
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK595" , "DCRho12HyperK595"  , "DICHAMP #tilde{K} 595 GeV/#font[12]{c}^{2}"  , 595,  1, 1, 0.04968409) ); //LO          
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK600" , "DCRho12HyperK600"  , "DICHAMP #tilde{K} 600 GeV/#font[12]{c}^{2}"  , 600,  1, 1, 0.0026232721237) ); //LO     
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK610" , "DCRho12HyperK610"  , "DICHAMP #tilde{K} 610 GeV/#font[12]{c}^{2}"  , 610,  1, 1, 0.00127431) ); //LO          
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK620" , "DCRho12HyperK620"  , "DICHAMP #tilde{K} 620 GeV/#font[12]{c}^{2}"  , 620,  1, 1, 0.00056965104319) ); //LO    
  signals.push_back(stSignal(-1,"DCRho12HyperK" , "DCRho12HyperK700" , "DCRho12HyperK700"  , "DICHAMP #tilde{K} 700 GeV/#font[12]{c}^{2}"  , 700,  1, 1, 0.00006122886211) ); //LO     

  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK100" , "DCRho16HyperK100"  , "DICHAMP #tilde{K} 100 GeV/#font[12]{c}^{2}"  , 100,  1, 1, 0.711518686800) ); //LO       
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK182" , "DCRho16HyperK182"  , "DICHAMP #tilde{K} 182 GeV/#font[12]{c}^{2}"  , 182,  1, 1, 0.089726059780) ); //LO       
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK302" , "DCRho16HyperK302"  , "DICHAMP #tilde{K} 302 GeV/#font[12]{c}^{2}"  , 302,  1, 1, 0.019769637301) ); //LO       
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK500" , "DCRho16HyperK500"  , "DICHAMP #tilde{K} 500 GeV/#font[12]{c}^{2}"  , 500,  1, 1, 0.0063302286576) ); //LO      
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK700" , "DCRho16HyperK700"  , "DICHAMP #tilde{K} 700 GeV/#font[12]{c}^{2}"  , 700,  1, 1, 0.002536779850) ); //LO       
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK730" , "DCRho16HyperK730"  , "DICHAMP #tilde{K} 730 GeV/#font[12]{c}^{2}"  , 730,  1, 1, 0.00213454921) ); //LO
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK770" , "DCRho16HyperK770"  , "DICHAMP #tilde{K} 770 GeV/#font[12]{c}^{2}"  , 770,  1, 1, 0.001737551) ); //LO 
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK790" , "DCRho16HyperK790"  , "DICHAMP #tilde{K} 790 GeV/#font[12]{c}^{2}"  , 790,  1, 1, 0.00161578593) ); //LO
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK795" , "DCRho16HyperK795"  , "DICHAMP #tilde{K} 795 GeV/#font[12]{c}^{2}"  , 795,  1, 1, 0.00153513713) ); //LO
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK800" , "DCRho16HyperK800"  , "DICHAMP #tilde{K} 800 GeV/#font[12]{c}^{2}"  , 800,  1, 1, 0.000256086965) ); //LO
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK810" , "DCRho16HyperK810"  , "DICHAMP #tilde{K} 810 GeV/#font[12]{c}^{2}"  , 810,  1, 1, 0.000140664) ); //LO 
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK820" , "DCRho16HyperK820"  , "DICHAMP #tilde{K} 820 GeV/#font[12]{c}^{2}"  , 820,  1, 1, 0.000097929923655) ); //LO 
  signals.push_back(stSignal(-1,"DCRho16HyperK" , "DCRho16HyperK900" , "DCRho16HyperK900"  , "DICHAMP #tilde{K} 900 GeV/#font[12]{c}^{2}"  , 900,  1, 1, 0.000013146066) ); //LO       
#endif
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
#ifdef ANALYSIS2011
     //2011 7TeV MC
   MC.push_back(stMC("MC_DYToTauTau"            ,     1.300E3  , -1, -1, 0));
   MC.push_back(stMC("MC_DYToMuMu"              ,     1.300E3  , -1, -1, 0));
   MC.push_back(stMC("MC_WJetsToLNu"            ,     2.777E4  , -1, -1, 1));
   MC.push_back(stMC("MC_TTJets"                ,     9.400E1  , -1, -1, 1));
 //MC.push_back(stMC("MC_QCD_Pt-15to30"         ,     8.16E8  , -1, -1, 0));
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
#endif
}

void GetInputFiles(std::vector<std::string>& inputFiles, std::string SampleName, int period=0){
//  std::string BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_3/11_08_03/";
  //std::string BaseDirectory = "dcache:/pnfs/cms/WAX/11/store/user/jchen/11_09_13_HSCP2011EDM/";
  //std::string BaseDirectory = "/uscmst1b_scratch/lpc1/lpcphys/jchen/HSCPEDM_11_01_11/";
//  std::string BaseDirectory = "root://eoscms//eos/cms/store/cmst3/user/querten/12_04_17_HSCP_EDM/";
  std::string BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_3/11_11_01/";
   if(SampleName=="Data"){
#ifdef ANALYSIS2011
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
     inputFiles.push_back(BaseDirectory + "Data_RunA_176467_176800.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_176801_177053.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_177074_177783.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_177788_178380.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_178420_179411.root");
     inputFiles.push_back(BaseDirectory + "Data_RunA_179434_180252.root");
#else
     inputFiles.push_back(BaseDirectory + "Data_190XXX.root");
     inputFiles.push_back(BaseDirectory + "Data_191XXX.root");
#endif
   }else if(SampleName.find("MC_",0)<std::string::npos){
     inputFiles.push_back(BaseDirectory + SampleName + ".root");
   }else{
     if (period==0) inputFiles.push_back(BaseDirectory + SampleName + ".root");
     if (period==1) inputFiles.push_back(BaseDirectory + SampleName + "BX1.root");
   }
}

#endif

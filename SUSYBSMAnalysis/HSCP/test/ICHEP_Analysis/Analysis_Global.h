
#ifndef HSCP_ANALYSIS_GLOBAL
#define HSCP_ANALYSIS_GLOBAL

std::map<string,double>  CrossSections;



//const char*        EstimLeg        = "dE/dx estimator (MeV/cm)";
//const char*        DiscrLeg        = "dE/dx discriminator";
const char*        EstimLeg        = "I_{h} (MeV/cm)";
const char*        DiscrLeg        = "I_{as}";
const unsigned int dEdxLabelSize   = 12;
const char*        dEdxLabel    [] = {"dedxCNPHarm2", "dedxCNPTru40", "dedxCNPMed", "dedxSTCNPHarm2", "dedxSTCNPTru40", "dedxSTCNPMed", "dedxProd", "dedxSmi", "dedxASmi", "dedxSTProd", "dedxSTSmi", "dedxSTASmi"};
const double       dEdxK_Data   [] = {2.5857        , 2.4496        , 2.8284      , 2.5857        , 2.4496        , 2.8284};
const double       dEdxC_Data   [] = {2.5497        , 2.2364        , 2.2459      , 2.5497        , 2.2364        , 2.2459};
const double       dEdxK_MC     [] = {2.5404        , 2.5272        , 2.5853      , 2.5404        , 2.5272        , 2.5853};
const double       dEdxC_MC     [] = {2.6433        , 2.2315        , 1.6376      , 2.6433        , 2.2315        , 1.6376};
const double       dEdxUpLim    [] = {40            , 40            , 40          , 40            , 40            , 40      , 1       , 1        , 1       , 1       , 1       , 1       };
const bool         dEdxIsDiscrim[] = {false         , false         , false       , false         , false         , false   , true    , true     , true    , true    , true    , true    };
const char*        dEdxLegend   [] = {EstimLeg      , EstimLeg      , EstimLeg    , EstimLeg      , EstimLeg      , EstimLeg, DiscrLeg, DiscrLeg , DiscrLeg, DiscrLeg, DiscrLeg, DiscrLeg};

int                dEdxSeleIndex;
int                dEdxMassIndex;

double             SelectionCutPt = -1;
double             SelectionCutI  = 1E-4;
double             DefaultCutPt   = 10;
double             DefaultCutP    = 0;
double             DefaultCutI    = 0;
double             CutPt[40*6];
double             CutI [40*6];

double             PtHistoUpperBound   = 2000;
double             MassHistoUpperBound = 2000;

float              GlobalMaxDZ   =   2.00;
float              GlobalMaxDXY  =   0.25;
float              GlobalMaxChi2 =  10.00;
int                GlobalMinQual =   2;
unsigned int       GlobalMinNOH  =   1;
unsigned int       GlobalMinNOM  =   3;
double             GlobalMaxPterr=   0.15;
//double             GlobalMinPt   =   7.50;
double             GlobalMinPt   =   15.00;
double             GlobalMinI    =   0.0;
float              GlobalMaxEta  =  2.5; 

double		   MinCandidateMass = 100;

char               SplitMode        = 2;   // 0 = No decomposition in Hit/Eta intervals
                                        // 1 = Decomposition in Hit Intervals, but not in Eta intervals
                                        // 2 = Decomposition in both Hit AND Eta Intervals

char		   TypeMode         = 0; //0 = All Candidates
					 //1 = Muon Candidates	

bool	           AbsolutePredictiction = true;//false;

#endif

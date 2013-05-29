
#ifndef HSCP_ANALYSIS_GLOBAL
#define HSCP_ANALYSIS_GLOBAL

std::string             dEdxS_Label     = "dedxASmi";
double             dEdxS_UpLim     = 1.0;
std::string             dEdxS_Legend    = "I_{as}";
std::string             dEdxM_Label     = "dedxHarm2";
double             dEdxM_UpLim     = 30.0;
std::string             dEdxM_Legend    = "I_{h} (MeV/cm)";
double             dEdxK_Data      = 2.43;//25857;
double             dEdxC_Data      = 2.92;//2.5497;
double             dEdxK_MC        = 2.40;//2.5404;
double             dEdxC_MC        = 2.95;//2.6433;

std::string             TOF_Label       = "combined";


double             PtHistoUpperBound   = 2000;
double             MassHistoUpperBound = 2000;

float              GlobalMaxV3D  =   2.00;
float              GlobalMaxDZ   =   2.00;
float              GlobalMaxDXY  =   2.00;//0.25;
float              GlobalMaxChi2 =   10.0;
int                GlobalMinQual =   2;
unsigned int       GlobalMinNOH  =   1;
unsigned int       GlobalMinNOM  =   8;
double             GlobalMinNDOF =   8;
double             GlobalMaxPterr=   0.35;
double             GlobalMaxTIsol = 5;
double             GlobalMaxEIsol = 0.25;
double             GlobalMinPt   =   25.00;
double             GlobalMinI    =   0.0;
double             GlobalMinTOF  =   1.0;
float              GlobalMaxEta  =  2.5; 

double		   MinCandidateMass = 100;

char		   TypeMode         = 0; //0 = All Candidates
					 //1 = Muon Candidates	



void InitdEdx(std::string dEdxS_Label_){
   if(dEdxS_Label_=="dedxASmi" || dEdxS_Label_=="dedxNPASmi"){
      dEdxS_UpLim  = 1.0;
      dEdxS_Legend = "I_{as}";
   }else if(dEdxS_Label_=="dedxProd" || dEdxS_Label_=="dedxNPProd"){
      dEdxS_UpLim  = 1.0;
      dEdxS_Legend = "I_{prod}";
   }else{
      dEdxS_UpLim  = 30.0;
      dEdxS_Legend = "I_{h} (MeV/cm)";
   }
}


#endif

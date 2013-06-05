
#ifndef HSCP_ANALYSIS_GLOBAL
#define HSCP_ANALYSIS_GLOBAL

std::string        dEdxS_Label     = "dedxASmi";
double             dEdxS_UpLim     = 1.0;
std::string        dEdxS_Legend    = "I_{as}";
std::string        dEdxM_Label     = "dedxHarm2";
double             dEdxM_UpLim     = 15.0;
std::string        dEdxM_Legend    = "I_{h} (MeV/cm)";
double             dEdxK_Data      = 2.529; //2.67;//25857;
double             dEdxC_Data      = 2.772; //2.44;//2.5497;
double             dEdxK_MC        = 2.529; //2.67;//2.5404;
double             dEdxC_MC        = 2.772; //2.44;//2.6433;

std::string        TOF_Label       = "combined";
std::string        TOFdt_Label     = "dt";
std::string        TOFcsc_Label    = "csc";

double             PtHistoUpperBound   = 1200;
double             MassHistoUpperBound = 2000;
int		   MassNBins           = 200;

float              GlobalMaxV3D  =   0.50;
float              GlobalMaxDZ   =   2.00;
float              GlobalMaxDXY  =   2.00;//0.25;
float              GlobalMaxChi2 =   5.0;
int                GlobalMinQual =   2;
unsigned int       GlobalMinNOH  =   11;
unsigned int       GlobalMinNOM  =   6;
double             GlobalMinNDOF =   8;
double             GlobalMinNDOFDT  =  6;
double             GlobalMinNDOFCSC =  6;
double             GlobalMaxTOFErr =   0.07;
double             GlobalMaxPterr=   0.25;
double             GlobalMaxTIsol = 50;
double             GlobalMaxEIsol = 0.30;
double             GlobalMinPt   =   45.00;
double             GlobalMinIs   =   0.0;
double             GlobalMinIm   =   3.0;
double             GlobalMinTOF  =   1.0;
float              GlobalMaxEta  =  1.5; 

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

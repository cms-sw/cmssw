
#include "Analysis_Samples.h"


struct stPlots {
   string Name;
   TH1D*  Mass;//[NSUBSAMPLE];

   double WN_TotalE;       double UN_TotalE;
   double WN_TotalTE;      double UN_TotalTE;
   double WN_Total;	   double UN_Total;
   double WN_V3D;          double UN_V3D;
   double WN_DZ;           double UN_DZ;
   double WN_DXY;          double UN_DXY;
   double WN_Chi2;         double UN_Chi2;
   double WN_Qual;         double UN_Qual;
   double WN_Hits;         double UN_Hits;
   double WN_nDof;         double UN_nDof;
   double WN_Pterr;        double UN_Pterr;
   double WN_MPt;          double UN_MPt;
   double WN_MI;           double UN_MI;
   double WN_MTOF;         double UN_MTOF;
   double WN_CIsol;	   double UN_CIsol;
   double WN_TIsol;	   double UN_TIsol;
   double WN_EIsol;	   double UN_EIsol;

   double WN_Pt;	   double UN_Pt;
   double WN_I;		   double UN_I;
   double WN_TOF;	   double UN_TOF;


   double WN_HSCPE;        double UN_HSCPE;

   double WN_TOF_SYSTA;      double UN_TOF_SYSTA;
   double WN_HSCPE_SYSTA;  double UN_HSCPE_SYSTA;
   double WN_TOF_SYSTB;      double UN_TOF_SYSTB;
   double WN_HSCPE_SYSTB;  double UN_HSCPE_SYSTB;


   double MeanICut;
   double MeanPtCut;

   TH1D*  BS_V3D;	   TH1D*  AS_V3D;
   TH1D*  BS_DZ;	   TH1D*  AS_DZ;
   TH1D*  BS_DXY;          TH1D*  AS_DXY;
   TH1D*  BS_Chi2;	   TH1D*  AS_Chi2;
   TH1D*  BS_Qual;         TH1D*  AS_Qual;
   TH1D*  BS_Hits;         TH1D*  AS_Hits;
   TH1D*  BS_nDof;         TH1D*  AS_nDof;
   TH1D*  BS_Pterr;        TH1D*  AS_Pterr;
   TH1D*  BS_MPt;          TH1D*  AS_MPt;
   TH1D*  BS_MIs;          TH1D*  AS_MIs;
   TH1D*  BS_MIm;          TH1D*  AS_MIm;
   TH1D*  BS_MTOF;         TH1D*  AS_MTOF;
   TH1D*  BS_CIsol;        TH1D*  AS_CIsol;
   TH1D*  BS_TIsol;        TH1D*  AS_TIsol;
   TH1D*  BS_EIsol;        TH1D*  AS_EIsol;

   TH1D*  BS_P; 	   TH1D*  AS_P;
   TH1D*  BS_Pt;	   TH1D*  AS_Pt;
   TH1D*  BS_Is;	   TH1D*  AS_Is;
   TH1D*  BS_Im;           TH1D*  AS_Im;
   TH1D*  BS_TOF;          TH1D*  AS_TOF;

   TH2D*  BS_EtaIs;        TH2D*  AS_EtaIs;
   TH2D*  BS_EtaIm;        TH2D*  AS_EtaIm;
   TH2D*  BS_EtaP;	   TH2D*  AS_EtaP;
   TH2D*  BS_EtaPt;	   TH2D*  AS_EtaPt;
   TH2D*  BS_PIs;	   TH2D*  AS_PIs;
   TH2D*  BS_PIm;          TH2D*  AS_PIm;
   TH2D*  BS_PtIs;         TH2D*  AS_PtIs;
   TH2D*  BS_PtIm;         TH2D*  AS_PtIm;

   TH2D*  BS_TOFIs;        TH2D*  AS_TOFIs;  
   TH2D*  BS_TOFIm;        TH2D*  AS_TOFIm;   
};

void stPlots_Init(stPlots& st, string BaseName, bool SkipSelectionPlot=false)
{
   st.Name = BaseName;
   st.WN_TotalE = 0;     st.UN_TotalE = 0;
   st.WN_TotalTE= 0;     st.UN_TotalTE= 0;
   st.WN_Total  = 0;     st.UN_Total  = 0;
   st.WN_V3D    = 0;     st.UN_V3D    = 0;
   st.WN_DZ     = 0;     st.UN_DZ     = 0;
   st.WN_DXY    = 0;     st.UN_DXY    = 0;
   st.WN_Chi2   = 0;     st.UN_Chi2   = 0;
   st.WN_Qual   = 0;     st.UN_Qual   = 0;
   st.WN_Hits   = 0;     st.UN_Hits   = 0;
   st.WN_nDof   = 0;     st.UN_nDof   = 0;
   st.WN_Pterr  = 0;     st.UN_Pterr  = 0;
   st.WN_CIsol  = 0;     st.UN_CIsol  = 0;
   st.WN_TIsol  = 0;     st.UN_TIsol  = 0;
   st.WN_EIsol  = 0;     st.UN_EIsol  = 0;
   st.WN_MPt    = 0;     st.UN_MPt    = 0;
   st.WN_MI     = 0;     st.UN_MI     = 0;
   st.WN_MTOF   = 0;     st.UN_MTOF   = 0;
   st.WN_Pt     = 0;     st.UN_Pt     = 0;
   st.WN_I      = 0;     st.UN_I      = 0;
   st.WN_TOF    = 0;     st.UN_TOF    = 0;
   st.WN_HSCPE  = 0;     st.UN_HSCPE  = 0;

   st.WN_TOF_SYSTA     = 0; st.UN_TOF_SYSTA     = 0;
   st.WN_HSCPE_SYSTA = 0; st.UN_HSCPE_SYSTA = 0;
   st.WN_TOF_SYSTB     = 0; st.UN_TOF_SYSTB     = 0;
   st.WN_HSCPE_SYSTB = 0; st.UN_HSCPE_SYSTB = 0;

   st.MeanICut  = 0;
   st.MeanPtCut = 0;


   string Name;
//   for(unsigned int i=0;i<NSUBSAMPLE;i++){
//         if(i>0 && SkipSelectionPlot)continue;
//         Name = BaseName + "_Mass" + GetNameFromIndex(i); st.Mass[i]  = new TH1D(Name.c_str(), Name.c_str(), 400, 0, MassHistoUpperBound);   st.Mass[i]->Sumw2();
//   }
   Name = BaseName + "_Mass"; st.Mass  = new TH1D(Name.c_str(), Name.c_str(), 400, 0, MassHistoUpperBound);   st.Mass->Sumw2();

   if(SkipSelectionPlot)return;

   Name = BaseName + "_BS_V3D"  ; st.BS_V3D   = new TH1D(Name.c_str(), Name.c_str(),  50, 0,  2);  st.BS_V3D->Sumw2();
   Name = BaseName + "_AS_V3D"  ; st.AS_V3D   = new TH1D(Name.c_str(), Name.c_str(),  50, 0,  2);  st.AS_V3D->Sumw2();

   Name = BaseName + "_BS_DZ"   ; st.BS_DZ    = new TH1D(Name.c_str(), Name.c_str(),  50, 0,  5);  st.BS_DZ->Sumw2();
   Name = BaseName + "_AS_DZ"   ; st.AS_DZ    = new TH1D(Name.c_str(), Name.c_str(),  50, 0,  5);  st.AS_DZ->Sumw2();

   Name = BaseName + "_BS_DXY"	; st.BS_DXY   = new TH1D(Name.c_str(), Name.c_str(),  50, 0,   1);  st.BS_DXY->Sumw2();        
   Name = BaseName + "_AS_DXY"	; st.AS_DXY   = new TH1D(Name.c_str(), Name.c_str(),  50, 0,   1);  st.AS_DXY->Sumw2();

   Name = BaseName + "_BS_Chi2"	; st.BS_Chi2  = new TH1D(Name.c_str(), Name.c_str(),   50, 0,  5);  st.BS_Chi2->Sumw2();
   Name = BaseName + "_AS_Chi2"	; st.AS_Chi2  = new TH1D(Name.c_str(), Name.c_str(),   50, 0,  5);  st.AS_Chi2->Sumw2();

   Name = BaseName + "_BS_Qual" ; st.BS_Qual  = new TH1D(Name.c_str(), Name.c_str(),   20, 0,  20);  st.BS_Qual->Sumw2();
   Name = BaseName + "_AS_Qual" ; st.AS_Qual  = new TH1D(Name.c_str(), Name.c_str(),   20, 0,  20);  st.AS_Qual->Sumw2();

   Name = BaseName + "_BS_Hits" ; st.BS_Hits  = new TH1D(Name.c_str(), Name.c_str(),   40, 0,  40);  st.BS_Hits->Sumw2();
   Name = BaseName + "_AS_Hits" ; st.AS_Hits  = new TH1D(Name.c_str(), Name.c_str(),   40, 0,  40);  st.AS_Hits->Sumw2();

   Name = BaseName + "_BS_nDof" ; st.BS_nDof  = new TH1D(Name.c_str(), Name.c_str(),   40, 0,  40);  st.BS_nDof->Sumw2();
   Name = BaseName + "_AS_nDof" ; st.AS_nDof  = new TH1D(Name.c_str(), Name.c_str(),   40, 0,  40);  st.AS_nDof->Sumw2();

   Name = BaseName + "_BS_PtErr"; st.BS_Pterr = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 3);  st.BS_Pterr->Sumw2();
   Name = BaseName + "_AS_PtErr"; st.AS_Pterr = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 3);  st.AS_Pterr->Sumw2();

   Name = BaseName + "_BS_MPt"  ; st.BS_MPt   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound); st.BS_MPt->Sumw2();
   Name = BaseName + "_AS_MPt"  ; st.AS_MPt   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound); st.AS_MPt->Sumw2();
   
   Name = BaseName + "_BS_MIs"  ; st.BS_MIs   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxS_UpLim); st.BS_MIs->Sumw2();
   Name = BaseName + "_AS_MIs"  ; st.AS_MIs   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxS_UpLim); st.AS_MIs->Sumw2();

   Name = BaseName + "_BS_MIm"  ; st.BS_MIm   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxM_UpLim); st.BS_MIm->Sumw2();
   Name = BaseName + "_AS_MIm"  ; st.AS_MIm   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxM_UpLim); st.AS_MIm->Sumw2();

   Name = BaseName + "_BS_MTOF" ; st.BS_MTOF  = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 20); st.BS_MTOF->Sumw2();
   Name = BaseName + "_AS_MTOF" ; st.AS_MTOF  = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 20); st.AS_MTOF->Sumw2();

   Name = BaseName + "_BS_CIsol"; st.BS_CIsol = new TH1D(Name.c_str(), Name.c_str(), 20, 0, 20); st.BS_CIsol->Sumw2();
   Name = BaseName + "_AS_CIsol"; st.AS_CIsol = new TH1D(Name.c_str(), Name.c_str(), 20, 0, 20); st.AS_CIsol->Sumw2();

   Name = BaseName + "_BS_TIsol"; st.BS_TIsol = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 1); st.BS_TIsol->Sumw2();
   Name = BaseName + "_AS_TIsol"; st.AS_TIsol = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 1); st.AS_TIsol->Sumw2();

   Name = BaseName + "_BS_EIsol"; st.BS_EIsol = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 1.5); st.BS_EIsol->Sumw2();
   Name = BaseName + "_AS_EIsol"; st.AS_EIsol = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 1.5); st.AS_EIsol->Sumw2();

   Name = BaseName + "_BS_P"    ; st.BS_P     = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, PtHistoUpperBound); st.BS_P->Sumw2();
   Name = BaseName + "_AS_P"    ; st.AS_P     = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, PtHistoUpperBound); st.AS_P->Sumw2();

   Name = BaseName + "_BS_Pt"   ; st.BS_Pt    = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, PtHistoUpperBound); st.BS_Pt->Sumw2();
   Name = BaseName + "_AS_Pt"   ; st.AS_Pt    = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, PtHistoUpperBound); st.AS_Pt->Sumw2();

   Name = BaseName + "_BS_Is"   ; st.BS_Is    = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, dEdxS_UpLim); st.BS_Is->Sumw2();
   Name = BaseName + "_AS_Is"   ; st.AS_Is    = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, dEdxS_UpLim); st.AS_Is->Sumw2();

   Name = BaseName + "_BS_Im"   ; st.BS_Im    = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, dEdxM_UpLim); st.BS_Im->Sumw2();
   Name = BaseName + "_AS_Im"   ; st.AS_Im    = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, dEdxM_UpLim); st.AS_Im->Sumw2();

   Name = BaseName + "_BS_TOF"  ; st.BS_TOF   = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, 20); st.BS_TOF->Sumw2();
   Name = BaseName + "_AS_TOF"  ; st.AS_TOF   = new TH1D(Name.c_str(), Name.c_str(), 1000, 0, 20); st.AS_TOF->Sumw2();

   Name = BaseName + "_BS_EtaIs"; st.BS_EtaIs = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, dEdxS_UpLim);
   Name = BaseName + "_AS_EtaIs"; st.AS_EtaIs = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, dEdxS_UpLim);

   Name = BaseName + "_BS_EtaIm"; st.BS_EtaIm = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, dEdxM_UpLim);
   Name = BaseName + "_AS_EtaIm"; st.AS_EtaIm = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, dEdxM_UpLim);

   Name = BaseName + "_BS_EtaP" ; st.BS_EtaP  = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, PtHistoUpperBound);
   Name = BaseName + "_AS_EtaP" ; st.AS_EtaP  = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, PtHistoUpperBound);

   Name = BaseName + "_BS_EtaPt"; st.BS_EtaPt = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, PtHistoUpperBound);
   Name = BaseName + "_AS_EtaPt"; st.AS_EtaPt = new TH2D(Name.c_str(), Name.c_str(), 50,-3, 3, 100, 0, PtHistoUpperBound);

   Name = BaseName + "_BS_PIs"  ; st.BS_PIs   = new TH2D(Name.c_str(), Name.c_str(), 200, 0, PtHistoUpperBound, 200, 0, dEdxS_UpLim);
   Name = BaseName + "_AS_PIs"  ; st.AS_PIs   = new TH2D(Name.c_str(), Name.c_str(), 200, 0, PtHistoUpperBound, 200, 0, dEdxS_UpLim);

   Name = BaseName + "_BS_PIm"  ; st.BS_PIm   = new TH2D(Name.c_str(), Name.c_str(), 200, 0, PtHistoUpperBound, 200, 0, dEdxM_UpLim);
   Name = BaseName + "_AS_PIm"  ; st.AS_PIm   = new TH2D(Name.c_str(), Name.c_str(), 200, 0, PtHistoUpperBound, 200, 0, dEdxM_UpLim);

   Name = BaseName + "_BS_PtIs" ; st.BS_PtIs  = new TH2D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound, 100, 0, dEdxS_UpLim);
   Name = BaseName + "_AS_PtIs" ; st.AS_PtIs  = new TH2D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound, 100, 0, dEdxS_UpLim);

   Name = BaseName + "_BS_PtIm" ; st.BS_PtIm  = new TH2D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound, 100, 0, dEdxM_UpLim);
   Name = BaseName + "_AS_PtIm" ; st.AS_PtIm  = new TH2D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound, 100, 0, dEdxM_UpLim);

   Name = BaseName + "_BS_TOFIs"; st.BS_TOFIs = new TH2D(Name.c_str(), Name.c_str(), 200, 0, 20, 200, 0, dEdxS_UpLim);
   Name = BaseName + "_AS_TOFIs"; st.AS_TOFIs = new TH2D(Name.c_str(), Name.c_str(), 200, 0, 20, 200, 0, dEdxS_UpLim);

   Name = BaseName + "_BS_TOFIm"; st.BS_TOFIm = new TH2D(Name.c_str(), Name.c_str(), 200, 0, 20, 200, 0, dEdxM_UpLim);
   Name = BaseName + "_AS_TOFIm"; st.AS_TOFIm = new TH2D(Name.c_str(), Name.c_str(), 200, 0, 20, 200, 0, dEdxM_UpLim);
}


void stPlots_InitFromFile(stPlots& st, string BaseName, TFile* InputFile)
{
   st.Name = BaseName;

   st.BS_V3D    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_V3D");
   st.AS_V3D    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_V3D");
   st.BS_DZ     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_DZ");
   st.AS_DZ     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_DZ");
   st.BS_DXY    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_DXY");
   st.AS_DXY    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_DXY");
   st.BS_Chi2   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_Chi2");
   st.AS_Chi2   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_Chi2");
   st.BS_Qual   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_Qual");
   st.AS_Qual   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_Qual");
   st.BS_Hits   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_Hits");
   st.AS_Hits   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_Hits");
   st.BS_nDof   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_nDof");
   st.AS_nDof   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_nDof");
   st.BS_Pterr  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_PtErr");
   st.AS_Pterr  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_PtErr");
   st.BS_MPt    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_MPt");
   st.AS_MPt    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_MPt");
   st.BS_MIm    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_MIm");
   st.AS_MIm    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_MIm");
   st.BS_MIs    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_MIs");
   st.AS_MIs    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_MIs");
   st.BS_MTOF   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_MTOF");
   st.AS_MTOF   = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_MTOF");
   st.BS_CIsol  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_CIsol");
   st.AS_CIsol  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_CIsol");
   st.BS_TIsol  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_TIsol");
   st.AS_TIsol  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_TIsol");
   st.BS_EIsol  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_EIsol");
   st.AS_EIsol  = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_EIsol");
   st.BS_P      = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_P");
   st.AS_P      = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_P");
   st.BS_Pt     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_Pt");
   st.AS_Pt     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_Pt");
   st.BS_Im     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_Im");
   st.AS_Im     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_Im");
   st.BS_Is     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_Is");
   st.AS_Is     = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_Is");
   st.BS_TOF    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_BS_TOF");
   st.AS_TOF    = (TH1D*)GetObjectFromPath(InputFile,  BaseName + "_AS_TOF");
   st.BS_EtaIs  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_EtaIs");
   st.AS_EtaIs  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_EtaIs");
   st.BS_EtaIm  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_EtaIm");
   st.AS_EtaIm  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_EtaIm");
   st.BS_EtaP   = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_EtaP");
   st.AS_EtaP   = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_EtaP");
   st.BS_EtaPt  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_EtaPt");
   st.AS_EtaPt  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_EtaPt");
   st.BS_PIs    = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_PIs");
   st.AS_PIs    = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_PIs");
   st.BS_PIm    = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_PIm");
   st.AS_PIm    = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_PIm");
   st.BS_PtIs   = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_PtIs");
   st.AS_PtIs   = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_PtIs");
   st.BS_PtIm   = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_PtIm");
   st.AS_PtIm   = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_PtIm");
   st.BS_TOFIs  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_TOFIs");
   st.AS_TOFIs  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_TOFIs");
   st.BS_TOFIm  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_BS_TOFIm");
   st.AS_TOFIm  = (TH2D*)GetObjectFromPath(InputFile,  BaseName + "_AS_TOFIm");

}

void stPlots_Clear(stPlots& st)
{
   delete st.BS_V3D;          delete st.AS_V3D;
   delete st.BS_DZ;           delete st.AS_DZ;
   delete st.BS_DXY;          delete st.AS_DXY;
   delete st.BS_Chi2;         delete st.AS_Chi2;
   delete st.BS_Qual;         delete st.AS_Qual;
   delete st.BS_Hits;         delete st.AS_Hits;
   delete st.BS_nDof;         delete st.AS_nDof;
   delete st.BS_Pterr;        delete st.AS_Pterr;
   delete st.BS_MPt;          delete st.AS_MPt;
   delete st.BS_MIs;          delete st.AS_MIs;
   delete st.BS_MIm;          delete st.AS_MIm;
   delete st.BS_MTOF;         delete st.AS_MTOF;
   delete st.BS_CIsol;        delete st.AS_CIsol;
   delete st.BS_TIsol;        delete st.AS_TIsol;
   delete st.BS_EIsol;        delete st.AS_EIsol;
   delete st.BS_P;            delete st.AS_P;
   delete st.BS_Pt;           delete st.AS_Pt;
   delete st.BS_Is;           delete st.AS_Is;
   delete st.BS_Im;           delete st.AS_Im;
   delete st.BS_TOF;          delete st.AS_TOF;
   delete st.BS_EtaIs;        delete st.AS_EtaIs;
   delete st.BS_EtaIm;        delete st.AS_EtaIm;
   delete st.BS_EtaP;         delete st.AS_EtaP;
   delete st.BS_EtaPt;        delete st.AS_EtaPt;
   delete st.BS_PIs;          delete st.AS_PIs;
   delete st.BS_PIm;          delete st.AS_PIm;
   delete st.BS_PtIs;         delete st.AS_PtIs;
   delete st.BS_PtIm;         delete st.AS_PtIm;
   delete st.BS_TOFIs;        delete st.AS_TOFIs;
   delete st.BS_TOFIm;        delete st.AS_TOFIm;
}



void stPlots_Dump(stPlots& st, FILE* pFile){
   fprintf(pFile,"--------------------\n");
   fprintf(pFile,"#Events                      weighted (unweighted) = %4.2E (%4.2E)\n",st.WN_TotalE,st.UN_TotalE);
   fprintf(pFile,"#Triggered Events            weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_TotalTE,st.UN_TotalTE,st.WN_TotalTE/st.WN_TotalE,st.UN_TotalTE/st.UN_TotalE);
   fprintf(pFile,"#Tracks                      weighted (unweighted) = %4.2E (%4.2E)\n",st.WN_Total,st.UN_Total);
   fprintf(pFile,"#Tracks passing Hits   cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Hits, st.UN_Hits, st.WN_Hits/st.WN_Total, st.UN_Hits/st.UN_Total);
   fprintf(pFile,"#Tracks passing nDof   cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_nDof, st.UN_nDof, st.WN_nDof/st.WN_Hits,  st.UN_nDof/st.UN_Hits );
   fprintf(pFile,"#Tracks passing Qual   cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Qual, st.UN_Qual, st.WN_Qual/st.WN_nDof,  st.UN_Qual/st.UN_nDof );
   fprintf(pFile,"#Tracks passing Chi2   cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Chi2, st.UN_Chi2, st.WN_Chi2/st.WN_Qual,  st.UN_Chi2/st.UN_Qual );
   fprintf(pFile,"#Tracks passing Min Pt cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_MPt,  st.UN_MPt,  st.WN_MPt /st.WN_Chi2,  st.UN_MPt /st.UN_Chi2 );
   fprintf(pFile,"#Tracks passing Min I  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_MI,   st.UN_MI,   st.WN_MI  /st.WN_MPt,   st.UN_MI  /st.UN_MPt  );
   fprintf(pFile,"#Tracks passing Min TOFcuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_MTOF, st.UN_MTOF, st.WN_MTOF/st.WN_MI,    st.UN_MTOF/st.UN_MI    );
   fprintf(pFile,"#Tracks passing V3D    cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_V3D,  st.UN_V3D,  st.WN_V3D /st.WN_MI,    st.UN_V3D /st.UN_MI   );
   fprintf(pFile,"#Tracks passing dZ     cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_DZ ,  st.UN_DZ ,  st.WN_DZ  /st.WN_V3D,   st.UN_DZ  /st.UN_V3D  );
   fprintf(pFile,"#Tracks passing dXY    cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_DXY,  st.UN_DXY,  st.WN_DXY /st.WN_DZ,    st.UN_DXY /st.UN_DZ   );
   fprintf(pFile,"#Tracks passing CIsol  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_CIsol,st.UN_CIsol,st.WN_CIsol/st.WN_DXY,  st.UN_CIsol/st.UN_DXY );
   fprintf(pFile,"#Tracks passing TIsol  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_TIsol,st.UN_TIsol,st.WN_TIsol/st.WN_CIsol,st.UN_TIsol/st.UN_CIsol );
   fprintf(pFile,"#Tracks passing EIsol  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_EIsol,st.UN_EIsol,st.WN_EIsol/st.WN_TIsol,st.UN_EIsol/st.UN_TIsol );
   fprintf(pFile,"#Tracks passing PtErr  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Pterr,st.UN_Pterr,st.WN_Pterr/st.WN_EIsol, st.UN_Pterr/st.UN_EIsol);
   fprintf(pFile,"#Tracks passing Basic  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Pterr,st.UN_Pterr,st.WN_Pterr/st.WN_Total,st.UN_Pterr/st.UN_Total);
   fprintf(pFile,"#Tracks passing Pt     cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Pt,   st.UN_Pt,   st.WN_Pt  /st.WN_MI,    st.UN_Pt  /st.UN_MI   );
   fprintf(pFile,"#Tracks passing I      cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_I,    st.UN_I,    st.WN_I   /st.WN_Pt,    st.UN_I   /st.UN_Pt   );
   fprintf(pFile,"#Tracks passing TOF    cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_TOF,  st.UN_TOF,  st.WN_TOF /st.WN_I,     st.UN_TOF /st.UN_I    );
   fprintf(pFile,"#Tracks passing selection    weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_TOF,  st.UN_TOF,  st.WN_TOF /st.WN_Total, st.UN_TOF /st.WN_Total);   
   fprintf(pFile,"--------------------\n");
   fprintf(pFile,"HSCP Detection Efficiency Before Trigger                           Eff=%4.3E (%4.3E)\n",st.WN_TOF /(2*st.WN_TotalE),  st.UN_TOF /(2*st.UN_TotalE) );
   fprintf(pFile,"HSCP Detection Efficiency After  Trigger                           Eff=%4.3E (%4.3E)\n",st.WN_TOF /(2*st.WN_TotalTE), st.UN_TOF /(2*st.UN_TotalTE));
   fprintf(pFile,"#HSCPTrack per HSCPEvent (with at least one HSCPTrack)             Eff=%4.3E (%4.3E)\n",st.WN_TOF /(  st.WN_HSCPE),   st.UN_TOF /(  st.UN_HSCPE  ));
   fprintf(pFile,"--------------------\n");
}


void stPlots_Draw(stPlots& st, string SavePath, string LegendTitle)
{
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;
   TCanvas* c1;
/*
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_DZ;                    legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_DZ;                    legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dz (cm)", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dz", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_DXY;                   legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_DXY;                   legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dxy (cm)", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dxy", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Chi2;                  legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Chi2;                  legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#chi^{2}/ndof", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Chi2", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Qual;                  legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Qual;                  legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Track quality", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Quality", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Hits;                  legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Hits;                  legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Track #Hits", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Hits", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Pterr;                 legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Pterr;                 legend.push_back("After  Cut");
   st.AS_Pterr->SetMinimum(1);
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Track p_{T} Err / p_{T}", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pterr", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_MPt;                   legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_MPt;                   legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Track p_{T} (GeV/c)", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P"); 
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MPt", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_MIs;                   legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_MIs;                   legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true); 
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIs", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_MIm;                   legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_MIm;                   legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIm", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_CIsol;                 legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_CIsol;                 legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Count", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolC", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_TIsol;                 legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_TIsol;                 legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Tk SumPt (GeV/c)", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolT", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_EIsol;                 legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_EIsol;                 legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Ecal Energy / p", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolE", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_HIsol;                 legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_HIsol;                 legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Hcal Energy / p", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolH", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Pt->Rebin(100,"TMPA"); legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Pt->Rebin(100,"TMPB"); legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Track p_{T} (GeV/c)", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pt", true);
   delete  Histos[0]; delete Histos[1];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Is->Rebin(100,"TMPA"); legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Is->Rebin(100,"TMPB"); legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Is", true);
   delete  Histos[0]; delete Histos[1];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Im->Rebin(100,"TMPA"); legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Im->Rebin(100,"TMPB"); legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Im", true);
   delete  Histos[0]; delete Histos[1];
   delete c1;
*/
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_EtaIs;                 legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaIs_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_EtaIs;                 legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaIs_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_EtaIm;                 legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", dEdxM_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaIm_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_EtaIm;                 legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", dEdxM_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaIm_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_EtaP;                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "p (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaP_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_EtaP;                  legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "p (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaP_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_EtaPt;                 legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "p_{T} (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaPt_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_EtaPt;                 legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "p_{T} (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaPt_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_PIs;                   legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p (GeV/c)", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PIs_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_PIm;                   legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p (GeV/c)", dEdxM_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PIm_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_PtIs;                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p_{T} (GeV/c)", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PtIs_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_PtIm;                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p_{T} (GeV/c)", dEdxM_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PtIm_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_PIs;                   legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p (GeV/c)", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PIs_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_PIm;                   legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p (GeV/c)", dEdxM_Legend.c_str(), 0,600, 0,25, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PIm_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_PtIs;                  legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p_{T} (GeV/c)", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PtIs_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_PtIm;                  legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "p_{T} (GeV/c)", dEdxM_Legend.c_str(), 0,600, 0,25, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"PtIm_AS", true);
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_TOFIs;                 legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "#beta", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOFIs_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_TOFIm;                 legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "#beta", dEdxM_Legend.c_str(), 0,0, 0,15, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOFIm_BS", true);
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_TOFIs;                 legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "#beta", dEdxS_Legend.c_str(), 0,0, 0,0, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOFIs_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_TOFIm;                 legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "#beta", dEdxM_Legend.c_str(), 0,0, 0,15, false);
   c1->SetLogz(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOFIm_AS", true);
   delete c1;






   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
//   Histos[0] = ((TH1*)(st.BS_EtaIs->ProjectionY("H1_PY",0                                    ,st.BS_EtaIs->GetXaxis()->FindBin(0.9)) ))->Rebin(8); legend.push_back("0.0 < |#eta| < 0.9");
//   Histos[1] = ((TH1*)(st.BS_EtaIs->ProjectionY("H2_PY",st.BS_EtaIs->GetXaxis()->FindBin(0.9),st.BS_EtaIs->GetXaxis()->FindBin(1.4)) ))->Rebin(8); legend.push_back("0.9 < |#eta| < 1.4");
//   Histos[2] = ((TH1*)(st.BS_EtaIs->ProjectionY("H3_PY",st.BS_EtaIs->GetXaxis()->FindBin(1.4),st.BS_EtaIs->GetXaxis()->FindBin(2.5)) ))->Rebin(8); legend.push_back("1.4 < |#eta| < 2.5");
   Histos[0] = (TH1*)st.BS_EtaIs->ProjectionY("H1_PY",0                                    ,st.BS_EtaIs->GetXaxis()->FindBin(0.9)); legend.push_back("0.0 < |#eta| < 0.9");
   Histos[1] = (TH1*)st.BS_EtaIs->ProjectionY("H2_PY",st.BS_EtaIs->GetXaxis()->FindBin(0.9),st.BS_EtaIs->GetXaxis()->FindBin(1.4)); legend.push_back("0.9 < |#eta| < 1.4");
   Histos[2] = (TH1*)st.BS_EtaIs->ProjectionY("H3_PY",st.BS_EtaIs->GetXaxis()->FindBin(1.4),st.BS_EtaIs->GetXaxis()->FindBin(2.5)); legend.push_back("1.4 < |#eta| < 2.5");
   Histos[0] = ((TH1D*)Histos[0])->Rebin(1);
   Histos[1] = ((TH1D*)Histos[1])->Rebin(1);
   Histos[2] = ((TH1D*)Histos[2])->Rebin(1);

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"EtaIs_BS_Projected", true);
   delete  Histos[0]; delete Histos[1];  delete Histos[2];
   delete c1;
}





void stPlots_DrawComparison(stPlots& st1, stPlots& st2, stPlots& st3, stSignal signal, string SavePath, string LegendTitle)
{  
   string SignalLeg  = signal.Legend;

   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_V3D->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_V3D->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_V3D->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "V3D (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"V3D_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_V3D->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_V3D->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_V3D->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "V3D (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"V3D_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_DZ->Clone();          legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_DZ->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_DZ->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dz (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dz_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_DZ->Clone();          legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_DZ->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_DZ->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dz (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dz_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_DXY->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_DXY->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_DXY->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dxy (cm)", "arbitrary units", 0,2, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dxy_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_DXY->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_DXY->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_DXY->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dxy (cm)", "arbitrary units", 0,2, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dxy_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Chi2->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Chi2->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Chi2->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#chi^{2}/ndof", "arbitrary units", 0,10, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Chi2_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Chi2->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Chi2->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Chi2->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#chi^{2}/ndof", "arbitrary units", 0,10, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Chi2_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Qual->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Qual->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Qual->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "quality", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Quality_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Qual->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Qual->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Qual->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Track quality", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Quality_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Hits->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Hits->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Hits->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#Hits", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Hits_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Hits->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Hits->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Hits->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#Hits", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Hits_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_nDof->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_nDof->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_nDof->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "TOF_{nDof}", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"nDof_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_nDof->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_nDof->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_nDof->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "TOF_{nDof}", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"nDof_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;




   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Pterr->Clone();       legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Pterr->Clone();       legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Pterr->Clone();       legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} Err / p_{T}", "arbitrary units", 0,1, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pterr_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Pterr->Clone();       legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Pterr->Clone();       legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Pterr->Clone();       legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} Err / p_{T}", "arbitrary units", 0,1, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pterr_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_MPt->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_MPt->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_MPt->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MPt_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_MPt->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_MPt->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_MPt->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MPt_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_MIs->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_MIs->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_MIs->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MI_BSs", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_MIm->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_MIm->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_MIm->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIm_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_MIs->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_MIs->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_MIs->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIs_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_MTOF->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_MTOF->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_MTOF->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MTOF_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_MTOF->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_MTOF->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_MTOF->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MTOF_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_MIm->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_MIm->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_MIm->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIm_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_CIsol->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_CIsol->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_CIsol->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track Count", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolC_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_CIsol->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_CIsol->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_CIsol->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track Count", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolC_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_TIsol->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_TIsol->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_TIsol->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track SumPt (GeV/c)", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolT_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_TIsol->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_TIsol->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_TIsol->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track SumPt (GeV/c)", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolT_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_EIsol->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_EIsol->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_EIsol->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: (Ecal + Hcal) Energy / p", "arbitrary units", 0,30, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolE_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_EIsol->Clone();        legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_EIsol->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_EIsol->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: (Ecal + Hcal) Energy / p", "arbitrary units", 0,30, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolE_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Is->Rebin(10,"TMPA");legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Is->Rebin(10,"TMPB");legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Is->Rebin(10,"TMPC");legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Is_BS");
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Im->Rebin(10,"TMPA");legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Im->Rebin(10,"TMPB");legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Im->Rebin(10,"TMPC");legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Im_BS");
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Is->Rebin(10,"TMPA");legend.push_back(SignalLeg);	if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Is->Rebin(10,"TMPB");legend.push_back("MC");	if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Is->Rebin(10,"TMPC");legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Is_AS");
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Im->Rebin(10,"TMPA");legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Im->Rebin(10,"TMPB");legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Im->Rebin(10,"TMPC");legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Im_AS");
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Pt->Rebin(10,"TMPA");legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Pt->Rebin(10,"TMPB");legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Pt->Rebin(10,"TMPC");legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pt_BS");
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Pt->Rebin(10,"TMPA");legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Pt->Rebin(10,"TMPB");legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Pt->Rebin(10,"TMPC");legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pt_AS");
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;




   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_TOF->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_TOF->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_TOF->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOF_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;
   
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_TOF->Clone();         legend.push_back(SignalLeg);  if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_TOF->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_TOF->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOF_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;
}





void stPlots_DrawComparison(string SavePath, string LegendTitle, stPlots* st1, stPlots* st2, stPlots* st3=NULL, stPlots* st4=NULL, stPlots* st5=NULL, stPlots* st6=NULL, stPlots* st7=NULL)
{ 
  std::vector<string> lg;
  std::vector<stPlots*> st;
  if(st1)st.push_back(st1); 
  if(st2)st.push_back(st2);   
  if(st3)st.push_back(st3);   
  if(st4)st.push_back(st4);
  if(st5)st.push_back(st5);
  if(st6)st.push_back(st6);
  if(st7)st.push_back(st7);


  std::vector<stSignal> signals;
  GetSignalDefinition(signals);
  for(unsigned int i=0;i<st.size();i++){
     int Index = -1;
     for(unsigned int s=0;s<signals.size();s++){
        if(signals[s].Name==st[i]->Name){Index=s;break;}
     }
     if(Index==-1){lg.push_back(st[i]->Name);}else{lg.push_back(signals[Index].Legend);}
  }
   
   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_V3D->Clone();      legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral());   }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "V3D (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"V3D_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_V3D->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "V3D (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"V3D_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_DZ->Clone();          legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dz (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dz_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_DZ->Clone();          legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dz (cm)", "arbitrary units", 0,5, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dz_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_DXY->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dxy (cm)", "arbitrary units", 0,2, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dxy_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_DXY->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "dxy (cm)", "arbitrary units", 0,2, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"dxy_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_Chi2->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#chi^{2}/ndof", "arbitrary units", 0,10, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Chi2_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_Chi2->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#chi^{2}/ndof", "arbitrary units", 0,10, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Chi2_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_Qual->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "quality", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Quality_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_Qual->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Track quality", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Quality_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_Hits->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#Hits", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Hits_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_Hits->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#Hits", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Hits_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_nDof->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "TOF_{nDof}", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"nDof_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_nDof->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "TOF_{nDof}", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"nDof_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;




   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_Pterr->Clone();       legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} Err / p_{T}", "arbitrary units", 0,1, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pterr_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_Pterr->Clone();       legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} Err / p_{T}", "arbitrary units", 0,1, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pterr_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_MPt->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MPt_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_MPt->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MPt_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_MIs->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MI_BSs", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_MIm->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIm_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_MIs->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIs_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_MTOF->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MTOF_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_MTOF->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MTOF_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_MIm->Clone();         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"MIm_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_CIsol->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track Count", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolC_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_CIsol->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track Count", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolC_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_TIsol->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track SumPt (GeV/c)", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolT_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_TIsol->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: Track SumPt (GeV/c)", "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolT_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_EIsol->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: (Ecal + Hcal) Energy / p", "arbitrary units", 0,30, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolE_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_EIsol->Clone();        legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Isolation: (Ecal + Hcal) Energy / p", "arbitrary units", 0,30, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"IsolE_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_Is->Rebin(10,"TMPA");legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Is_BS");
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_Im->Rebin(10,"TMPA");legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Im_BS");
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_Is->Rebin(10,"TMPA");legend.push_back(lg[i]);	if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend.c_str(), "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Is_AS");
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_Im->Rebin(10,"TMPA");legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend.c_str(), "arbitrary units", 0,20, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Im_AS");
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_Pt->Rebin(10,"TMPA");legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pt_BS");
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_Pt->Rebin(10,"TMPA");legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "p_{T} (GeV/c)", "arbitrary units", 0,1250, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Pt_AS");
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;




   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->BS_TOF->Rebin(10,"TMPA");;         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); }
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P",0.35);
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOF_BS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;
   
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   for(unsigned int i=0;i<st.size();i++){
   Histos[i] = (TH1*)st[i]->AS_TOF->Rebin(10,"TMPA");         legend.push_back(lg[i]);  if(Histos[i]->Integral()>0) Histos[i]->Scale(1.0/Histos[i]->Integral()); } 
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#beta", "arbitrary units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"P", 0.35);
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOF_AS", true);
   for(unsigned int i=0;i<st.size();i++){delete Histos[i];}
   delete c1;
}


/////////////////////////// STRUCTURES DECLARATION /////////////////////////////

struct stPlots {
   TH1D*  BS_DZ;	   TH1D*  AS_DZ;
   TH1D*  BS_DXY;          TH1D*  AS_DXY;
   TH1D*  BS_Chi2;	   TH1D*  AS_Chi2;
   TH1D*  BS_Qual;         TH1D*  AS_Qual;
   TH1D*  BS_Hits;         TH1D*  AS_Hits;
   TH1D*  BS_Pterr;        TH1D*  AS_Pterr;
   TH1D*  BS_MPt;          TH1D*  AS_MPt;
   TH1D*  BS_MI;           TH1D*  AS_MI;
   TH1D*  BS_Pt;	   TH1D*  AS_Pt;
   TH1D*  BS_I;		   TH1D*  AS_I;

   TH2D*  BS_EtaP;	   TH2D*  AS_EtaP;
   TH2D*  BS_EtaPt;	   TH2D*  AS_EtaPt;
   TH2D*  BS_PI;	   TH2D*  AS_PI;

   double WN_TotalE;       double UN_TotalE;
   double WN_TotalTE;      double UN_TotalTE;
   double WN_Total;	   double UN_Total;
   double WN_DZ;           double UN_DZ;
   double WN_DXY;          double UN_DXY;
   double WN_Chi2;         double UN_Chi2;
   double WN_Qual;         double UN_Qual;
   double WN_Hits;         double UN_Hits;
   double WN_Pterr;        double UN_Pterr;
   double WN_MPt;          double UN_MPt;
   double WN_MI;           double UN_MI;
   double WN_Pt;	   double UN_Pt;
   double WN_I;		   double UN_I;
};


/////////////////////////// FUNCTION DECLARATION /////////////////////////////

void stPlots_Init(stPlots& st, string BaseName)
{
   st.WN_TotalE = 0;     st.UN_TotalE = 0;
   st.WN_TotalTE= 0;     st.UN_TotalTE= 0;
   st.WN_Total  = 0;     st.UN_Total  = 0;
   st.WN_DZ     = 0;     st.UN_DZ     = 0;
   st.WN_DXY    = 0;     st.UN_DXY    = 0;
   st.WN_Chi2   = 0;     st.UN_Chi2   = 0;
   st.WN_Qual   = 0;     st.UN_Qual   = 0;
   st.WN_Hits   = 0;     st.UN_Hits   = 0;
   st.WN_Pterr  = 0;     st.UN_Pterr  = 0;
   st.WN_MPt    = 0;     st.UN_MPt    = 0;
   st.WN_MI     = 0;     st.UN_MI     = 0;
   st.WN_Pt     = 0;     st.UN_Pt     = 0;
   st.WN_I      = 0;     st.UN_I      = 0;


   string Name;
   Name = BaseName + "_BS_DZ"   ; st.BS_DZ    = new TH1D(Name.c_str(), Name.c_str(),  100, 0,  25);
   Name = BaseName + "_AS_DZ"   ; st.AS_DZ    = new TH1D(Name.c_str(), Name.c_str(),  100, 0,  25);

   Name = BaseName + "_BS_DXY"	; st.BS_DXY   = new TH1D(Name.c_str(), Name.c_str(),  100, 0,   5);
   Name = BaseName + "_AS_DXY"	; st.AS_DXY   = new TH1D(Name.c_str(), Name.c_str(),  100, 0,   5);

   Name = BaseName + "_BS_Chi2"	; st.BS_Chi2  = new TH1D(Name.c_str(), Name.c_str(),   50, 0,  50);
   Name = BaseName + "_AS_Chi2"	; st.AS_Chi2  = new TH1D(Name.c_str(), Name.c_str(),   50, 0,  50);

   Name = BaseName + "_BS_Qual" ; st.BS_Qual  = new TH1D(Name.c_str(), Name.c_str(),   20, 0,  20);
   Name = BaseName + "_AS_Qual" ; st.AS_Qual  = new TH1D(Name.c_str(), Name.c_str(),   20, 0,  20);

   Name = BaseName + "_BS_Hits" ; st.BS_Hits  = new TH1D(Name.c_str(), Name.c_str(),   40, 0,  40);
   Name = BaseName + "_AS_Hits" ; st.AS_Hits  = new TH1D(Name.c_str(), Name.c_str(),   40, 0,  40);

   Name = BaseName + "_BS_PtErr"; st.BS_Pterr = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 3);
   Name = BaseName + "_AS_PtErr"; st.AS_Pterr = new TH1D(Name.c_str(), Name.c_str(), 100, 0, 3);

   Name = BaseName + "_BS_MPt"  ; st.BS_MPt   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound);
   Name = BaseName + "_AS_MPt"  ; st.AS_MPt   = new TH1D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound);
   
   Name = BaseName + "_BS_MI"   ; st.BS_MI    = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxUpLim[dEdxSeleIndex]);
   Name = BaseName + "_AS_MI"   ; st.AS_MI    = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxUpLim[dEdxSeleIndex]);

   Name = BaseName + "_BS_Pt"   ; st.BS_Pt    = new TH1D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound);
   Name = BaseName + "_AS_Pt"   ; st.AS_Pt    = new TH1D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound);

   Name = BaseName + "_BS_I"    ; st.BS_I     = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxUpLim[dEdxSeleIndex]);
   Name = BaseName + "_AS_I"    ; st.AS_I     = new TH1D(Name.c_str(), Name.c_str(), 100, 0, dEdxUpLim[dEdxSeleIndex]);

   Name = BaseName + "_BS_EtaP" ; st.BS_EtaP  = new TH2D(Name.c_str(), Name.c_str(), 100,-3, 3, 100, 0, PtHistoUpperBound);
   Name = BaseName + "_AS_EtaP" ; st.AS_EtaP  = new TH2D(Name.c_str(), Name.c_str(), 100,-3, 3, 100, 0, PtHistoUpperBound);

   Name = BaseName + "_BS_EtaPt"; st.BS_EtaPt = new TH2D(Name.c_str(), Name.c_str(), 100,-3, 3, 100, 0, PtHistoUpperBound);
   Name = BaseName + "_AS_EtaPt"; st.AS_EtaPt = new TH2D(Name.c_str(), Name.c_str(), 100,-3, 3, 100, 0, PtHistoUpperBound);

   Name = BaseName + "_BS_PI"   ; st.BS_PI    = new TH2D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound, 100, 0, dEdxUpLim[dEdxSeleIndex]);
   Name = BaseName + "_AS_PI"   ; st.AS_PI    = new TH2D(Name.c_str(), Name.c_str(), 100, 0, PtHistoUpperBound, 100, 0, dEdxUpLim[dEdxSeleIndex]);
}


void stPlots_Clear(stPlots& st)
{
   delete st.BS_DZ;           delete st.AS_DZ;
   delete st.BS_DXY;          delete st.AS_DXY;
   delete st.BS_Chi2;         delete st.AS_Chi2;
   delete st.BS_Qual;         delete st.AS_Qual;
   delete st.BS_Hits;         delete st.AS_Hits;
   delete st.BS_Pterr;        delete st.AS_Pterr;
   delete st.BS_MPt;          delete st.AS_MPt;
   delete st.BS_MI;           delete st.AS_MI;
   delete st.BS_Pt;           delete st.AS_Pt;
   delete st.BS_I;            delete st.AS_I;
   delete st.BS_EtaP;         delete st.AS_EtaP;
   delete st.BS_EtaPt;        delete st.AS_EtaPt;
   delete st.BS_PI;           delete st.AS_PI;
}


void stPlots_Dump(stPlots& st, FILE* pFile){
   fprintf(pFile,"--------------------\n");
   fprintf(pFile,"#Events                      weighted (unweighted) = %4.2E (%4.2E)\n",st.WN_TotalE,st.UN_TotalE);
   fprintf(pFile,"#Triggered Events            weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_TotalTE,st.UN_TotalTE,st.WN_TotalTE/st.WN_TotalE,st.UN_TotalTE/st.UN_TotalE);
   fprintf(pFile,"#Tracks                      weighted (unweighted) = %4.2E (%4.2E)\n",st.WN_Total,st.UN_Total);
   fprintf(pFile,"#Tracks passing dZ     cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_DZ ,  st.UN_DZ ,  st.WN_DZ  /st.WN_Total, st.UN_DZ  /st.UN_Total);
   fprintf(pFile,"#Tracks passing dXY    cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_DXY,  st.UN_DXY,  st.WN_DXY /st.WN_DZ,    st.UN_DXY /st.UN_DZ   );
   fprintf(pFile,"#Tracks passing Chi2   cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Chi2, st.UN_Chi2, st.WN_Chi2/st.WN_DXY,   st.UN_Chi2/st.UN_DXY  );
   fprintf(pFile,"#Tracks passing Qual   cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Qual, st.UN_Qual, st.WN_Qual/st.WN_Chi2,  st.UN_Qual/st.UN_Chi2 );
   fprintf(pFile,"#Tracks passing Hits   cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Hits, st.UN_Hits, st.WN_Hits/st.WN_Qual,  st.UN_Hits/st.UN_Qual );
   fprintf(pFile,"#Tracks passing PtErr  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Pterr,st.UN_Pterr,st.WN_Pterr/st.WN_Hits,  st.UN_Pterr /st.UN_Hits );
   fprintf(pFile,"#Tracks passing Min Pt cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_MPt,  st.UN_MPt,  st.WN_MPt /st.WN_Pterr, st.UN_MPt /st.UN_Pterr);
   fprintf(pFile,"#Tracks passing Min I  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_MI,   st.UN_MI,   st.WN_MI  /st.WN_MPt,   st.UN_MI  /st.UN_MPt  );
   fprintf(pFile,"#Tracks passing Basic  cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_MI,   st.UN_MI,   st.WN_MI  /st.WN_Total, st.UN_MI  /st.UN_Total);
   fprintf(pFile,"#Tracks passing Pt     cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_Pt,   st.UN_Pt,   st.WN_Pt  /st.WN_MI,    st.UN_Pt  /st.UN_MI   );
   fprintf(pFile,"#Tracks passing I      cuts  weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_I,    st.UN_I,    st.WN_I   /st.WN_Pt,    st.UN_I   /st.UN_Pt   );
   fprintf(pFile,"#Tracks passing selection    weighted (unweighted) = %4.2E (%4.2E) Eff=%4.3E (%4.3E)\n",st.WN_I,    st.UN_I,    st.WN_I   /st.WN_Total, st.UN_I   /st.WN_Total);   
   fprintf(pFile,"--------------------\n");
}


void stPlots_Draw(stPlots& st, string SavePath)
{
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;
   TCanvas* c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_DZ;                    legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_DZ;                    legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "dz", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"dz");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_DXY;                   legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_DXY;                   legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "dxy", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"dxy", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Chi2;                  legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Chi2;                  legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "#chi^{2}/ndof", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Chi2", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Qual;                  legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Qual;                  legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track quality", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Quality", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Hits;                  legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Hits;                  legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track #Hits", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Hits", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Pterr;                 legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Pterr;                 legend.push_back("After  Cut");
   st.AS_Pterr->SetMinimum(1);
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Pt Err / Track Pt", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Pterr", true);
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_MPt;                   legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_MPt;                   legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Pt", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L"); 
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"MPt", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_MI;                    legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_MI;                    legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Ionization Info", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true); 
   SaveCanvas(c1,SavePath,"MI", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_Pt;                    legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_Pt;                    legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Pt", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Pt", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_I;                     legend.push_back("Before Cut");
   Histos[1] = (TH1*)st.AS_I;                     legend.push_back("After  Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Ionization Info", "#Tracks", 0,0, 0,0, false);
   DrawLegend(Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"I", true);
   delete c1;



   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_EtaP;                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "P (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   SaveCanvas(c1,SavePath,"EtaP_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_EtaP;                  legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "P (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   SaveCanvas(c1,SavePath,"EtaP_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_EtaPt;                 legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "Pt (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   SaveCanvas(c1,SavePath,"EtaPt_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_EtaPt;                 legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#eta", "Pt (GeV/c)", 0,0, 0,0, false);
   c1->SetLogz(true);
   SaveCanvas(c1,SavePath,"EtaPt_AS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.BS_PI;                    legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "P (GeV/c)", "Track Ionization Info", 0,0, 0,0, false);
   c1->SetLogz(true);
   SaveCanvas(c1,SavePath,"PI_BS", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st.AS_PI;                    legend.push_back("After Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ", "P (GeV/c)", "Track Ionization Info", 0,0, 0,0, false);
   c1->SetLogz(true);
   SaveCanvas(c1,SavePath,"PI_AS", true);
   delete c1;
}





void stPlots_DrawComparison(stPlots& st1, stPlots& st2, stPlots& st3, string SavePath)
{  
   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_DZ->Clone();          legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_DZ->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_DZ->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "dz", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"dz_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_DZ->Clone();          legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_DZ->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_DZ->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "dz", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"dz_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_DXY->Clone();         legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_DXY->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_DXY->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "dxy", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"dxy_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_DXY->Clone();         legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_DXY->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_DXY->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "dxy", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"dxy_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Chi2->Clone();        legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Chi2->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Chi2->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "#chi^{2}/ndof", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Chi2_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Chi2->Clone();        legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Chi2->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Chi2->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "#chi^{2}/ndof", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Chi2_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Qual->Clone();        legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Qual->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Qual->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track quality", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Quality_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Qual->Clone();        legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Qual->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Qual->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track quality", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Quality_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Hits->Clone();        legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Hits->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Hits->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track #Hits", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Hits_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Hits->Clone();        legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Hits->Clone();        legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Hits->Clone();        legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track #Hits", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Hits_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Pterr->Clone();       legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Pterr->Clone();       legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Pterr->Clone();       legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Pt Err / Track Pt", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Pterr_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Pterr->Clone();       legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Pterr->Clone();       legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Pterr->Clone();       legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Pt Err / Track Pt", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Pterr_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_MPt->Clone();         legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_MPt->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_MPt->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Pt", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"MPt_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_MPt->Clone();         legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_MPt->Clone();         legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_MPt->Clone();         legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Pt", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"MPt_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_MI->Clone();          legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_MI->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_MI->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Ionization Info", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"MI_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_MI->Clone();          legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_MI->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_MI->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Ionization Info", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"MI_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_I->Clone();           legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_I->Clone();           legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_I->Clone();           legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Ionization Info", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"I_BS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_I->Clone();           legend.push_back("Sign");	if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_I->Clone();           legend.push_back("MC");	if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_I->Clone();           legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Track Ionization Info", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"I_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.BS_Pt->Clone();          legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.BS_Pt->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.BS_Pt->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Pt (GeV/c)", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Pt_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)st1.AS_Pt->Clone();          legend.push_back("Sign");     if(Histos[0]->Integral()>0) Histos[0]->Scale(1.0/Histos[0]->Integral());
   Histos[1] = (TH1*)st2.AS_Pt->Clone();          legend.push_back("MC");       if(Histos[1]->Integral()>0) Histos[1]->Scale(1.0/Histos[1]->Integral());
   Histos[2] = (TH1*)st3.AS_Pt->Clone();          legend.push_back("Data");     if(Histos[2]->Integral()>0) Histos[2]->Scale(1.0/Histos[2]->Integral());
   DrawSuperposedHistos((TH1**)Histos, legend, "",  "Pt (GeV/c)", "#Tracks", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"","L");
   c1->SetLogy(true);
   SaveCanvas(c1,SavePath,"Pt_AS", true);
   delete Histos[0]; delete Histos[1]; delete Histos[2];
   delete c1;
}


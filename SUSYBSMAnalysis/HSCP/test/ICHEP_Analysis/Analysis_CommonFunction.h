
#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"



double CutFromEfficiency(TH1* Histo, double Efficiency, bool DoesKeepLeft=false)
{
   if(DoesKeepLeft){  Efficiency = 1 - Efficiency;  }

   char Buffer[1024];
   sprintf(Buffer,"%s_NTracks",Histo->GetName());
   TH1D* Temp = new TH1D(Buffer,Buffer, Histo->GetXaxis()->GetNbins(), Histo->GetXaxis()->GetXmin(), Histo->GetXaxis()->GetXmax());

   double Entries  = Histo->Integral(0,Histo->GetNbinsX()+1);
   Temp->SetBinContent(0,Entries);

   double Integral = 0;
   for(int i=0;i<=Histo->GetXaxis()->GetNbins()+1;i++){
      Integral += Histo->GetBinContent(i);
      if(Integral>Entries)Integral = Entries;
       Temp->SetBinContent(i,   Entries - Integral);      
   }

   unsigned int CutPosition = Temp->GetXaxis()->GetNbins()+1;
   for(int c=0;c<=Temp->GetXaxis()->GetNbins()+1;c++){
      if(Temp->GetBinContent(c)/Entries <= Efficiency){ CutPosition = c;  break; }
   }
   delete Temp;

   if(DoesKeepLeft){
      return Histo->GetXaxis()->GetBinLowEdge(CutPosition);
   }else{
      return Histo->GetXaxis()->GetBinUpEdge(CutPosition);
   }
}

double Efficiency(TH1* Histo, double CutX){
   double Entries  = Histo->Integral(0,Histo->GetNbinsX()+1);
   double Integral = Histo->Integral(Histo->GetXaxis()->FindBin(CutX),Histo->GetNbinsX()+1);
   return Integral/Entries;
}

double Efficiency(TH2* Histo, double CutX, double CutY){
   double Entries  = Histo->Integral(0,Histo->GetNbinsX()+1, 0,Histo->GetNbinsY()+1);
   double Integral = Histo->Integral(Histo->GetXaxis()->FindBin(CutX),Histo->GetNbinsX()+1, Histo->GetYaxis()->FindBin(CutY),Histo->GetNbinsY()+1);
   return Integral/Entries;
}


double EfficiencyAndError(TH1* Histo, double CutX, double& error){
   double Entries  = Histo->Integral(0,Histo->GetNbinsX()+1);
   double Integral = 0;
          error    = 0;
   for(Int_t binx = Histo->GetXaxis()->FindBin(CutX); binx<= Histo->GetNbinsX()+1; ++binx){
      Integral += Histo->GetBinContent(binx);
      error    += Histo->GetBinError(binx)*Histo->GetBinError(binx);
   }
   error = sqrt(error);
   error /= Entries;
   return Integral/Entries;
}




double GetEventInRange(double min, double max, TH1D* hist){
  int binMin = hist->GetXaxis()->FindBin(min);
  int binMax = hist->GetXaxis()->FindBin(max);
  return hist->Integral(binMin,binMax);
}

void FillArray(int HitIndex, int EtaIndex, double* Array, double value){
   Array[ 0                 ] +=  value;
   Array[           EtaIndex] +=  value;
   Array[HitIndex           ] +=  value;
   Array[HitIndex + EtaIndex] +=  value;
}

void FillHisto(int HitIndex, int EtaIndex, TH1D** Histo, double value, double weight){
   Histo[ 0                 ]->Fill(value,weight);
   Histo[           EtaIndex]->Fill(value,weight);
   Histo[HitIndex           ]->Fill(value,weight);
   Histo[HitIndex + EtaIndex]->Fill(value,weight);
}

void FillHisto(int HitIndex, int EtaIndex, TH2D** Histo, double value1, double value2, double weight){
   Histo[ 0                 ]->Fill(value1,value2,weight);
   Histo[           EtaIndex]->Fill(value1,value2,weight);
   Histo[HitIndex           ]->Fill(value1,value2,weight);
   Histo[HitIndex + EtaIndex]->Fill(value1,value2,weight);
}

double GetMass(double P, double I, bool MC=false){
   if(MC){
      const double& K = dEdxK_MC[dEdxMassIndex];
      const double& C = dEdxC_MC[dEdxMassIndex];

      return sqrt((I-C)/K)*P;
   }else{
      const double& K = dEdxK_Data[dEdxMassIndex];
      const double& C = dEdxC_Data[dEdxMassIndex];

      return sqrt((I-C)/K)*P;
   }

   return -1;
}

TF1* GetMassLine(double M, bool MC=false)
{
   double K;   double C;
   if(MC){
      K = dEdxK_MC[dEdxMassIndex];
      C = dEdxC_MC[dEdxMassIndex];
   }else{
      K = dEdxK_Data[dEdxMassIndex];
      C = dEdxC_Data[dEdxMassIndex];
   }

   double BetaMax = 0.9;
   double PMax = sqrt((BetaMax*BetaMax*M*M)/(1-BetaMax*BetaMax));

   double BetaMin = 0.2;
   double PMin = sqrt((BetaMin*BetaMin*M*M)/(1-BetaMin*BetaMin));

   TF1* MassLine = new TF1("MassLine","[2] + ([0]*[0]*[1])/(x*x)", PMin, PMax);
   MassLine->SetParName  (0,"M");
   MassLine->SetParName  (1,"K");
   MassLine->SetParName  (2,"C");
   MassLine->SetParameter(0, M);
   MassLine->SetParameter(1, K);
   MassLine->SetParameter(2, C);
   MassLine->SetLineWidth(2);
   return MassLine;
}

void GetIndices(int NOM, double Eta, int& HitIndex, int& EtaIndex)
{
   HitIndex = NOM*6;
   EtaIndex = 0;

         if(fabs(Eta)<0.5)EtaIndex = 1;
   else  if(fabs(Eta)<1.0)EtaIndex = 2;
   else  if(fabs(Eta)<1.5)EtaIndex = 3;
   else  if(fabs(Eta)<2.0)EtaIndex = 4;
   else                   EtaIndex = 5;
}

int GetCutIndex(int HitIndex, int EtaIndex){
   int CutIndex;
   if(SplitMode==0){
      CutIndex = 0;
   }else if(SplitMode==1){
      CutIndex = HitIndex;
   }else{
      CutIndex = HitIndex + EtaIndex;
   }
   return CutIndex;
}

void GetNameFromIndex(char* NameExt, int index)
{
      unsigned int Hit = index/6;
      unsigned int Eta = index%6;
      if(Hit>=1)sprintf(NameExt,"%s_SSHit%02i",NameExt,Hit);
      if(Eta==1)sprintf(NameExt,"%s_Eta00to05",NameExt);
      if(Eta==2)sprintf(NameExt,"%s_Eta05to10",NameExt);
      if(Eta==3)sprintf(NameExt,"%s_Eta10to15",NameExt);
      if(Eta==4)sprintf(NameExt,"%s_Eta15to20",NameExt);
      if(Eta==5)sprintf(NameExt,"%s_Eta20to25",NameExt);
}

double deltaR(double eta1, double phi1, double eta2, double phi2) {
   double deta = eta1 - eta2;
   double dphi = phi1 - phi2;
   while (dphi >   M_PI) dphi -= 2*M_PI;
   while (dphi <= -M_PI) dphi += 2*M_PI;
   return sqrt(deta*deta + dphi*dphi);
}


void GetPredictionRescale(string InputPattern, double& Rescale, double& RMS, bool ForceRecompute=false)
{
   size_t CutIndex = InputPattern.find("/Type");
   InputPattern    = InputPattern.substr(0,CutIndex+7);
   string Input    = InputPattern + "PredictionRescale.txt";


   FILE* pFile = fopen(Input.c_str(),"r");
   if(pFile && !ForceRecompute){
      float tmp1, tmp2;
      fscanf(pFile,"Rescale=%f RMS=%f\n",&tmp1,&tmp2);
      Rescale = tmp1;
      RMS = tmp2;
      fclose(pFile);
   }else{
      Rescale = 0;
      RMS     = 0;
      int    NPoints = 0;

      std::vector<double> DValue;
      std::vector<double> PValue;
  
      for(float WP_Pt=0;WP_Pt>=-5;WP_Pt-=0.5f){
      for(float WP_I =0;WP_I >=-5;WP_I -=0.5f){
         char Buffer[2048];
         sprintf(Buffer,"%sWPPt%+03i/WPI%+03i/DumpHistos.root",InputPattern.c_str(),(int)(10*WP_Pt),(int)(10*WP_I));
         TFile* InputFile = new TFile(Buffer); 
         if(!InputFile || InputFile->IsZombie() || !InputFile->IsOpen() || InputFile->TestBit(TFile::kRecovered) )continue;

         double d=0, p=0;//, m=0;
         TH1D* Hd = (TH1D*)GetObjectFromPath(InputFile, "Mass_Data");if(Hd){d=GetEventInRange(0,75,Hd);delete Hd;}
         TH1D* Hp = (TH1D*)GetObjectFromPath(InputFile, "Mass_Pred");if(Hp){p=GetEventInRange(0,75,Hp);delete Hp;}
//       TH1D* Hm = (TH1D*)GetObjectFromPath(InputFile, "Mass_MCTr");if(Hm){m=GetEventInRange(0,75,Hm);delete Hm;}

//       if(!(d!=d) && p>0 && d>10 && (WP_Pt+WP_I)<=-3){
//         if(!(d!=d) && p>0 && d>20 && (WP_Pt+WP_I)<=-3){
         if(!(d!=d) && p>0 && d>20 && (WP_Pt+WP_I)<=-2){
            DValue.push_back(d);
            PValue.push_back(p);
            printf("%6.2f %6.2f (eff=%6.2E) --> %f  (d=%6.2E)\n",WP_Pt,WP_I, pow(10,WP_Pt+WP_I),d/p, d);
            Rescale += (d/p);
            NPoints++;
         }
         InputFile->Close();
      }}
      printf("----------------------------\n");
      Rescale /= NPoints;

      for(unsigned int i=0;i<DValue.size();i++){
          RMS += pow( (DValue[i]/(PValue[i]*Rescale)) - 1.0 ,2);
      }
      RMS /= NPoints;
      RMS = sqrt(RMS);

      FILE* pFile = fopen(Input.c_str(),"w");
      if(!pFile)return;
      fprintf(pFile,"Rescale=%6.2f RMS=%6.2f\n",Rescale,RMS);
      fclose(pFile);
   }
   printf("Mean Rescale = %f   RMS = %f\n",Rescale, RMS);
}






#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"



///////////////////////////////////////////////////////////////////////////////////////  STUFF RELATED TO THE SUBSAMPLES


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




double GetEventInRange(double min, double max, TH1D* hist, double& error){
  int binMin = hist->GetXaxis()->FindBin(min);
  int binMax = hist->GetXaxis()->FindBin(max);
  error = 0; for(int i=binMin;i<binMax;i++){ error += pow(hist->GetBinError(i),2); }  error = sqrt(error);
  return hist->Integral(binMin,binMax);
}

double GetMassFromBeta(double P, double beta){
   double gamma = 1/sqrt(1-beta*beta);
   return P/(beta*gamma);
} 

double GetTOFMass(double P, double TOF){
   return GetMassFromBeta(P, 1/TOF);
}


double GetIBeta(double I, bool MC=false){
   double K, C;

   if(MC){
      K = dEdxK_MC;
      C = dEdxC_MC;
   }else{
      K = dEdxK_Data;
      C = dEdxC_Data;
   }

   double a = K / (I-C);
   double b2 = a / (a+1);

   if(b2<0)return -1*sqrt(b2);
   return sqrt(b2);
}



double GetMass(double P, double I, bool MC=false){
   if(MC){
      const double& K = dEdxK_MC;
      const double& C = dEdxC_MC;

      if(I-C<0)return -1;
      return sqrt((I-C)/K)*P;
   }else{
      const double& K = dEdxK_Data;
      const double& C = dEdxC_Data;

      if(I-C<0)return -1;
      return sqrt((I-C)/K)*P;
   }

   return -1;
}

TF1* GetMassLine(double M, bool MC=false)
{
   double K;   double C;
   if(MC){
      K = dEdxK_MC;
      C = dEdxC_MC;
   }else{
      K = dEdxK_Data;
      C = dEdxC_Data;
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



double deltaR(double eta1, double phi1, double eta2, double phi2) {
   double deta = eta1 - eta2;
   double dphi = phi1 - phi2;
   while (dphi >   M_PI) dphi -= 2*M_PI;
   while (dphi <= -M_PI) dphi += 2*M_PI;
   return sqrt(deta*deta + dphi*dphi);
}


std::string LegendFromType(const std::string& InputPattern){
   if(InputPattern.find("Type0",0)<std::string::npos){
      return std::string("Tracker - Only");
   }else if(InputPattern.find("Type1",0)<std::string::npos){
      return std::string("Tracker + Muon");
   }else{
      return std::string("Tracker + TOF");
   }
}


void GetPredictionRescale(std::string InputPattern, double& Rescale, double& RMS, bool ForceRecompute=false)
{
   size_t CutIndex = InputPattern.find("/Type");
   InputPattern    = InputPattern.substr(0,CutIndex+7);
   std::string Input    = InputPattern + "PredictionRescale.txt";


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
         double error =0;
         TH1D* Hd = (TH1D*)GetObjectFromPath(InputFile, "Mass_Data");if(Hd){d=GetEventInRange(0,75,Hd,error);delete Hd;}
         TH1D* Hp = (TH1D*)GetObjectFromPath(InputFile, "Mass_Pred");if(Hp){p=GetEventInRange(0,75,Hp,error);delete Hp;}
//       TH1D* Hm = (TH1D*)GetObjectFromPath(InputFile, "Mass_MCTr");if(Hm){m=GetEventInRange(0,75,Hm);delete Hm;}

//       if(!(d!=d) && p>0 && d>10 && (WP_Pt+WP_I)<=-3){
//         if(!(d!=d) && p>0 && d>20 && (WP_Pt+WP_I)<=-3){
         if(!(d!=d) && p>0 && d>20 && (WP_Pt+WP_I)<=-2){
//         if(!(d!=d) && p>0 && d>500 && (WP_Pt+WP_I)<=-2){
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

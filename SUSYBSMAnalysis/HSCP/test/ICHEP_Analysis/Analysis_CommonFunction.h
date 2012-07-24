// Original Author:  Loic Quertenmont

#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////// 
// general purpose code 

// return the TypeMode from a string inputPattern
int TypeFromPattern(const std::string& InputPattern){
   if(InputPattern.find("Type0",0)<std::string::npos){       return 0;
   }else if(InputPattern.find("Type1",0)<std::string::npos){ return 1;
   }else{                                                    return 2;
   }
}

// define the legend corresponding to a Type
std::string LegendFromType(const std::string& InputPattern){
   switch(TypeFromPattern(InputPattern)){
      case 0:  return std::string("Tracker - Only"); break;
      case 1:  return std::string("Tracker + Muon"); break;
      case 2:  return std::string("Tracker + TOF" ); break;
      default : std::string("unknown");
   }
   return std::string("unknown");
}

// compute deltaR between two point (eta,phi) (eta,phi)
double deltaR(double eta1, double phi1, double eta2, double phi2) {
   double deta = eta1 - eta2;
   double dphi = phi1 - phi2;
   while (dphi >   M_PI) dphi -= 2*M_PI;
   while (dphi <= -M_PI) dphi += 2*M_PI;
   return sqrt(deta*deta + dphi*dphi);
}

// function to go form a TH3 plot with cut index on the x axis to a  TH2
TH2D* GetCutIndexSliceFromTH3(TH3D* tmp, unsigned int CutIndex, string Name="zy"){
   tmp->GetXaxis()->SetRange(CutIndex+1,CutIndex+1);
   return (TH2D*)tmp->Project3D(Name.c_str());
}

// function to go form a TH2 plot with cut index on the x axis to a  TH1
TH1D* GetCutIndexSliceFromTH2(TH2D* tmp, unsigned int CutIndex, string Name="_py"){
   return tmp->ProjectionY(Name.c_str(),CutIndex+1,CutIndex+1);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////// 
// Genertic code for beta/mass reconstruction starting from p&dEdx or p&TOF

// compute mass out of a beta and momentum value
double GetMassFromBeta(double P, double beta){
   double gamma = 1/sqrt(1-beta*beta);
   return P/(beta*gamma);
} 

// compute mass out of a momentum and tof value
double GetTOFMass(double P, double TOF){
   return GetMassFromBeta(P, 1/TOF);
}

// estimate beta from a dEdx value, if dEdx is below the allowed threshold returns a negative beta value
double GetIBeta(double I, bool MC){
   double& K = dEdxK_Data;
   double& C = dEdxC_Data;
   if(MC){ K = dEdxK_MC;
           C = dEdxC_MC;  }

   double a = K / (I-C);
   double b2 = a / (a+1);
   if(b2<0)return -1*sqrt(b2);
   return sqrt(b2);
}

// compute mass out of a momentum and dEdx value
double GetMass(double P, double I, bool MC){
   double& K = dEdxK_Data;
   double& C = dEdxC_Data;
   if(MC){ K = dEdxK_MC;
           C = dEdxC_MC;  }

   if(I-C<0)return -1;
   return sqrt((I-C)/K)*P;
}

// return a TF1 corresponding to a mass line in the momentum vs dEdx 2D plane
TF1* GetMassLine(double M, bool MC){
   double& K = dEdxK_Data;
   double& C = dEdxC_Data;
   if(MC){ K = dEdxK_MC;
           C = dEdxC_MC;  }

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








////////////////////////////////////////////////////////////////////////////////////////////////////////// 
// Functions below were used for the 2009 and 2010 papers, but are probably not used anymore 

// return the selection efficiency for a given disribution (TH1) and for a given cut... Nothing but the integral above the cut divided by the number of events
double Efficiency(TH1* Histo, double CutX){
   double Entries  = Histo->Integral(0,Histo->GetNbinsX()+1);
   double Integral = Histo->Integral(Histo->GetXaxis()->FindBin(CutX),Histo->GetNbinsX()+1);
   return Integral/Entries;
}

// same as before but also compute the error in the efficiency
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

// return the selection efficiency for a given disribution (TH2) and for a given recangular signal region ... Nothing but the integral above the cut divided by the number of events
double Efficiency(TH2* Histo, double CutX, double CutY){
   double Entries  = Histo->Integral(0,Histo->GetNbinsX()+1, 0,Histo->GetNbinsY()+1);
   double Integral = Histo->Integral(Histo->GetXaxis()->FindBin(CutX),Histo->GetNbinsX()+1, Histo->GetYaxis()->FindBin(CutY),Histo->GetNbinsY()+1);
   return Integral/Entries;
}

// return the number of entry in an histogram (and it's error) in a window defined by two cuts
double GetEventInRange(double min, double max, TH1D* hist, double& error){
  int binMin = hist->GetXaxis()->FindBin(min);
  int binMax = hist->GetXaxis()->FindBin(max);
  error = 0; for(int i=binMin;i<binMax;i++){ error += pow(hist->GetBinError(i),2); }  error = sqrt(error);
  return hist->Integral(binMin,binMax);
}

// not used anymore, was computing a scale factor between datadriven prediction and observation using the M<75GeV events
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

      pFile = fopen(Input.c_str(),"w");
      if(!pFile)return;
      fprintf(pFile,"Rescale=%6.2f RMS=%6.2f\n",Rescale,RMS);
      fclose(pFile);
   }
   printf("Mean Rescale = %f   RMS = %f\n",Rescale, RMS);
}

// find the intersection between two graphs... very useful to know what is the excluded mass range from an observed xsection limit and theoretical xsection
double FindIntersectionBetweenTwoGraphs(TGraph* obs, TGraph* th, double Min, double Max, double Step, double ThUncertainty=0, bool debug=false){

   double Intersection = -1;

   double ThShift = 1.0-ThUncertainty;
   double PreviousX = Min;
   double PreviousV = obs->Eval(PreviousX, 0, "") - (ThShift * th->Eval(PreviousX, 0, "")) ;
   if(PreviousV>0)return -1;
   for(double x=Min+=Step;x<Max;x+=Step){                 
      double V = obs->Eval(x, 0, "") - (ThShift * th->Eval(x, 0, "") );
      if(debug){
         printf("%7.2f --> Obs=%6.2E  Th=%6.2E",x,obs->Eval(x, 0, ""),ThShift * th->Eval(x, 0, ""));
         if(V>=0)printf("   X\n");
         else printf("\n");
      }
      if(V<0){
         PreviousX = x;
         PreviousV = V;
      }else{
         Intersection = PreviousX;
      }
   }
   return Intersection;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Genertic code related to samples processing in FWLITE --> functions below will be loaded only if FWLITE compiler variable is defined

#ifdef FWLITE
double DistToHSCP        (const susybsm::HSCParticle& hscp, const std::vector<reco::GenParticle>& genColl, int& IndexOfClosest);
int    HowManyChargedHSCP(const std::vector<reco::GenParticle>& genColl);
void   GetGenHSCPBeta    (const std::vector<reco::GenParticle>& genColl, double& beta1, double& beta2, bool onlyCharged=true);

// compute the distance between a "reconstructed" HSCP candidate and the closest geenrated HSCP
double DistToHSCP (const susybsm::HSCParticle& hscp, const std::vector<reco::GenParticle>& genColl, int& IndexOfClosest){
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;

   double RMin = 9999; IndexOfClosest=-1;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000)continue;    

      double dR = deltaR(track->eta(), track->phi(), genColl[g].eta(), genColl[g].phi());
      if(dR<RMin){RMin=dR;IndexOfClosest=g;}
   }
   return RMin;
}

// count the number of charged generated HSCP in the event --> this is needed to reweights the events for different gluino ball fraction starting from f=10% samples
int HowManyChargedHSCP (const std::vector<reco::GenParticle>& genColl){
   int toReturn = 0;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000)continue;
      if(AbsPdg==1000993 || AbsPdg==1009313 || AbsPdg==1009113 || AbsPdg==1009223 || AbsPdg==1009333 || AbsPdg==1092114 || AbsPdg==1093214 || AbsPdg==1093324)continue; //Skip neutral gluino RHadrons
      if(AbsPdg==1000622 || AbsPdg==1000642 || AbsPdg==1006113 || AbsPdg==1006311 || AbsPdg==1006313 || AbsPdg==1006333)continue;  //skip neutral stop RHadrons
      toReturn++;
   }
   return toReturn;
}

// returns the generated beta of the two firsts HSCP in the events
void  GetGenHSCPBeta (const std::vector<reco::GenParticle>& genColl, double& beta1, double& beta2, bool onlyCharged){
   beta1=-1; beta2=-1;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000)continue;
      if(onlyCharged && (AbsPdg==1000993 || AbsPdg==1009313 || AbsPdg==1009113 || AbsPdg==1009223 || AbsPdg==1009333 || AbsPdg==1092114 || AbsPdg==1093214 || AbsPdg==1093324))continue; //Skip neutral gluino RHadrons
      if(onlyCharged && (AbsPdg==1000622 || AbsPdg==1000642 || AbsPdg==1006113 || AbsPdg==1006311 || AbsPdg==1006313 || AbsPdg==1006333))continue;  //skip neutral stop RHadrons
      if(beta1<0){beta1=genColl[g].p()/genColl[g].energy();}else if(beta2<0){beta2=genColl[g].p()/genColl[g].energy();return;}
   }
}
#endif



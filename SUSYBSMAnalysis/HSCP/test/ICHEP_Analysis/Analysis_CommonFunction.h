
#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"



///////////////////////////////////////////////////////////////////////////////////////  STUFF RELATED TO THE SUBSAMPLES


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

void GetIndices(int NOM, double Eta, int& HitIndex, int& EtaIndex)
{
         if(fabs(Eta)<1.0)EtaIndex = 1;
   else  if(fabs(Eta)<2.0)EtaIndex = 2;
   else                   EtaIndex = 3;

         if(NOM<=8       )HitIndex = 1;
   else  if(NOM<=10      )HitIndex = 2;
   else  if(NOM<=12      )HitIndex = 3;
   else  if(NOM<=14      )HitIndex = 4;
   else  if(NOM<=16      )HitIndex = 5;
   else                   HitIndex = 6;
   
   HitIndex*=4;  //Multiply by the number of eta slices + 1, in order to allow EtaIndex+HitIndex summmation
}

string GetNameFromIndex(int index)
{
   char buffer[256]; buffer[0]='\0';
   unsigned int Hit = index/4;  //Divide by eta slices +1 in order to recover the hit and eta index.
   unsigned int Eta = index%4;

   if(Hit==1)sprintf(buffer,"%s_SSHit00-08",buffer);
   if(Hit==2)sprintf(buffer,"%s_SSHit09-10",buffer);
   if(Hit==3)sprintf(buffer,"%s_SSHit11-12",buffer);
   if(Hit==4)sprintf(buffer,"%s_SSHit13-14",buffer);
   if(Hit==5)sprintf(buffer,"%s_SSHit15-16",buffer);
   if(Hit==6)sprintf(buffer,"%s_SSHit17-99",buffer);
 
   if(Eta==1)sprintf(buffer,"%s_Eta00-10",buffer);
   if(Eta==2)sprintf(buffer,"%s_Eta10-20",buffer);
   if(Eta==3)sprintf(buffer,"%s_Eta20-25",buffer);
   return string(buffer);
}

bool isSubSampleExist(int i, bool extended=false){
   if(!extended){
      if(SplitMode==0 && i>0)return false;
      if(SplitMode==1 && (i==0             || i%NETASUBSAMPLE!=0))return false;
      if(SplitMode==2 && (i< NETASUBSAMPLE || i%NETASUBSAMPLE==0))return false;
   }else{
      if(SplitMode==0 && i>0)return false;
      if(SplitMode==1 && (i%NETASUBSAMPLE!=0))return false;
   }
   return true;
}

///////////////////////////////////////////////////////////////////////////////////////

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

   if(Temp->GetBinContent(CutPosition)<=0.0)CutPosition = Temp->GetXaxis()->GetNbins()+1;

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



double GetMass(double P, double I, bool MC=false){
   if(MC){
      const double& K = dEdxK_MC;
      const double& C = dEdxC_MC;

      return sqrt((I-C)/K)*P;
   }else{
      const double& K = dEdxK_Data;
      const double& C = dEdxC_Data;

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


string LegendFromType(const string& InputPattern){
   if(InputPattern.find("Type0",0)<string::npos){
      return string("Tracker - Only");
   }else if(InputPattern.find("Type1",0)<string::npos){
      return string("Tracker + Muon");
   }else{
      return string("Tracker + TOF");
   }
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




void MassPredictionFromABCD(string InputPattern, TH1D* Pred_Mass)
{
   int SplitMode=0;
   if(InputPattern.find("SplitMode1",0)<string::npos)SplitMode=1;
   if(InputPattern.find("SplitMode2",0)<string::npos)SplitMode=2;

   string Input     = InputPattern + "DumpHistos.root";
   TFile* InputFile = new TFile(Input.c_str());

   printf("Predicting (Finding Prob)    :");
   int TreeStep = (40*6)/50;if(TreeStep==0)TreeStep=1;
   int CountStep = 0;
   for(unsigned int i=0;i<40*6;i++){
      if(SplitMode==0 && i>0)continue;
      if(SplitMode==1 && (i==0 || i%6!=0))continue;
      if(SplitMode==2 && (i< 6 || i%6==0))continue;
      if(i%TreeStep==0 && CountStep<=50){printf(".");fflush(stdout);CountStep++;}

      char PredExt[1024];
      char DataExt[1024];
      sprintf(PredExt,"Pred%s",GetNameFromIndex(i).c_str());
      //GetNameFromIndex(PredExt, i);
      sprintf(DataExt,"Data%s",GetNameFromIndex(i).c_str());
      //GetNameFromIndex(DataExt, i);

      TH1D* Pred_MassSubSample = (TH1D*)Pred_Mass->Clone("subsamplesprediction");
      Pred_MassSubSample->Reset();

      TH1D* Pred_P    = (TH1D*)GetObjectFromPath(InputFile, string("P_"   ) + PredExt);
      TH1D* Pred_I    = (TH1D*)GetObjectFromPath(InputFile, string("I_"   ) + PredExt);
      TH2D* Data_PI_A = (TH2D*)GetObjectFromPath(InputFile, string("PI_A_") + DataExt);
      TH2D* Data_PI_B = (TH2D*)GetObjectFromPath(InputFile, string("PI_B_") + DataExt);
      TH2D* Data_PI_C = (TH2D*)GetObjectFromPath(InputFile, string("PI_C_") + DataExt);
      //TH2D* Data_PI_D = (TH2D*)GetObjectFromPath(InputFile, string("PI_D_") + DataExt);

      double N_A = Data_PI_A->Integral();	double N_Aerr = N_A;
      double N_B = Data_PI_B->Integral();	double N_Berr = N_B;
      double N_C = Data_PI_C->Integral();	double N_Cerr = N_C;
      //double N_D = Data_PI_D->Integral();	//double N_Derr = N_D;



      double IntegralP = Pred_P->Integral(0, Pred_P->GetNbinsX()+1);
      double IntegralI = Pred_I->Integral(0, Pred_I->GetNbinsX()+1);
      if(IntegralP>0)Pred_P->Scale(1.0/IntegralP);
      if(IntegralI>0)Pred_I->Scale(1.0/IntegralI);
      IntegralP = N_C;
      IntegralI = N_B;

//      printf("%6.2E %6.2E %6.2E %6.2E | %6.2E %6.2E\n",N_A,N_B,N_C,N_D, IntegralP, IntegralI);


      double N_A_L = N_A - sqrt(N_Aerr); if(N_A_L<0)N_A_L=0;
      double N_A_C = N_A;                if(N_A_C<0)N_A_C=0;
      double N_A_U = N_A + sqrt(N_Aerr); if(N_A_U<0)N_A_U=0;

      double N_B_L = N_B - sqrt(N_Berr); if(N_B_L<0)N_B_L=0;
      double N_B_C = N_B;                if(N_B_C<0)N_B_C=0;
      double N_B_U = N_B + sqrt(N_Berr); if(N_B_U<0)N_B_U=0;

      double N_C_L = N_C - sqrt(N_Cerr); if(N_C_L<0)N_C_L=0;
      double N_C_C = N_C;                if(N_C_C<0)N_C_C=0;
      double N_C_U = N_C + sqrt(N_Cerr); if(N_C_U<0)N_C_U=0;

      double NExpectedBckgEntriesC;
      double NExpectedBckgEntriesC2;

      if(N_A>0){
         NExpectedBckgEntriesC  = ((N_C*N_B)/N_A);
         NExpectedBckgEntriesC2 = sqrt((pow(N_B/N_A,2)*N_Cerr) + (pow(N_C/N_A,2)*N_Berr) + (pow((N_B*(N_C)/(N_A*N_A)),2)*N_Aerr));
      }else{
         NExpectedBckgEntriesC  = 0;
         NExpectedBckgEntriesC2 = 0;
      }

      //Loop on Mass Line
      for(int m=0;m<Pred_Mass->GetNbinsX()+1;m++){
         //Find which bins contributes to this particular mass bin
         std::vector<std::pair<int,int> > BinThatGivesThisMass;
         for(int x=1;x<Pred_P->GetNbinsX()+1;x++){
         for(int y=1;y<Pred_I->GetNbinsX()+1;y++){
            double Mass = GetMass( Pred_P->GetXaxis()->GetBinCenter(x) , Pred_I->GetXaxis()->GetBinCenter(y) );
            if(Mass>Pred_Mass->GetXaxis()->GetBinLowEdge(m) && Mass<Pred_Mass->GetXaxis()->GetBinUpEdge(m)){
               BinThatGivesThisMass.push_back(std::make_pair(x,y));
            }
         }}

         double MBinContent=0;
         double MBinError2 =0;

         //Loops on the bins that contribute to this mass bin.

	 /////////////////BEGINNING OF MODIFICATIONS BY GIACOMO ////////////////////////////////////////////////////////////////////
	   /// Variable ErrMassBin is the statistical error on the considered mass bin. To compute the statistical error on the prediction in [75, 2000] GeV, just use the same code below with variable BinThatGivesThisMass filled with all pairs of p-I bins that contribute to this interval.

	   //GGG
	   /// bx1 -->i1; by1-->j1; vx1 --> Ci1 ; vy1 --> Bj1 ; ****MISTAKE - MUST USE MBinContent INSTEAD ***NExpectedBckgEntriesC --> N ********** ; N_A[i] --> A ;         
	   double Err_Numer_ijSum=0.;  
	   double Err_Denom_ijSum=0.;
	   double Err_Numer_CorrelSum=0.;
	   double ErrSquared=0.;
	   double ErrMassBin=0.;
	   ////////

         for(unsigned int b1=0;b1<BinThatGivesThisMass.size();b1++){
            double bx1 = BinThatGivesThisMass[b1].first;
            double by1 = BinThatGivesThisMass[b1].second;
            double vx1 = Pred_P->GetBinContent(bx1);
            double vy1 = Pred_I->GetBinContent(by1);
            double ex1 = Pred_P->GetBinError(bx1);
            double ey1 = Pred_I->GetBinError(by1);
            double vz1 = vx1*vy1;

	    //GGG
	    double vxN1=vx1 *IntegralP; 
	    double vyN1=vy1 *IntegralI; 

	    Err_Numer_ijSum += (vxN1*vyN1*(vxN1+vyN1)); 
	    Err_Denom_ijSum += (vxN1*vyN1); // will square at the end of the loop 
	    /////////

            MBinContent += vz1*NExpectedBckgEntriesC;
//            Pred_PI[i]->SetBinContent(bx1,by1,vz1*NExpectedBckgEntriesC);

	    //GGG
            //Compute the errors with a covariance matrix (on the fly) --> Only vertical and horizontal term contributes.
	    ///bx2 -->i2; by2-->j2; vxN2 --> Ci2 ; vyN2 --> Bj2 ;
	    /////

            for(unsigned int b2=0;b2<BinThatGivesThisMass.size();b2++){
               double bx2 = BinThatGivesThisMass[b2].first;
               double by2 = BinThatGivesThisMass[b2].second;
               double vx2 = Pred_P->GetBinContent(bx2);
               double vy2 = Pred_I->GetBinContent(by2);
               //double ex2 = Pred_P[i]->GetBinError(bx2);
               //double ey2 = Pred_I[i]->GetBinError(by2);


	       //GGG
	       double vxN2=vx2*IntegralP; 
	       double vyN2=vy2*IntegralI; 

	       /////////




               if(bx1==bx2 && by1==by2){
                  //Correlation with itself!
                  MBinError2 += NExpectedBckgEntriesC2*(ex1*ex1+ey1*ey1)*vz1*vz1;
               }else if(by1==by2){
                  //Vertical term
                  MBinError2 += NExpectedBckgEntriesC2*vx1*vx2*ey1;

		  //GGG
		  Err_Numer_CorrelSum += vyN2*vxN1*vxN2;
		  /////

               }else if(bx1==bx2){
                  //Horizontal term
                  MBinError2 += NExpectedBckgEntriesC2*vy1*vy2*ex1;


		  //GGG
		  Err_Numer_CorrelSum += vxN2*vyN1*vyN2;
		  /////


               }else{
                  //Diagonal term... do nothing
               }
            }
//            printf("Interval %i --> M = %i --> %f +- %f, %f+-%f\n",i,m,vx1,ex1,vy1,ey1);
//            printf("Interval %i --> M = %i --> %i Bins concerned --> %fEntries -->  %f +- %f\n",i,m,BinThatGivesThisMass.size(),NExpectedBckgEntriesC,MBinContent,NExpectedBckgEntriesC*sqrt(MBinError));
         }

         Pred_MassSubSample->SetBinContent(m, MBinContent);
//         Pred_Mass[i]->SetBinError  (m, sqrt(MBinError2));

	 //GGG
	 // squared error on predicted background in considered  mass bin
	 if ( N_A != 0 && Err_Denom_ijSum != 0 ) ErrSquared= ( MBinContent * MBinContent)*(1./N_A + (Err_Numer_ijSum + Err_Numer_CorrelSum)/(Err_Denom_ijSum*Err_Denom_ijSum) );
	 //final statistical error
	 ErrMassBin = sqrt(ErrSquared);
         Pred_MassSubSample->SetBinError  (m, ErrMassBin);
	 /////////
	 /////////////////END OF MODIFICATIONS BY GIACOMO ////////////////////////////////////////////////////////////////////


         BinThatGivesThisMass.clear();
      }
//      if(SplitMode!=0)Pred_PI   [0]->Add(Pred_PI   [i],1);
      Pred_Mass->Add(Pred_MassSubSample,1);     
      delete Pred_MassSubSample;
   }
   printf("\n");
}


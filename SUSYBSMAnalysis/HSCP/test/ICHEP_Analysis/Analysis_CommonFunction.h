// Original Author:  Loic Quertenmont

#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////// 
// general purpose code 

// return the TypeMode from a string inputPattern
int TypeFromPattern(const std::string& InputPattern){
   if(InputPattern.find("Type0",0)<std::string::npos){       return 0;
   }else if(InputPattern.find("Type1",0)<std::string::npos){ return 1;
   }else if(InputPattern.find("Type2",0)<std::string::npos){ return 2;
   }else if(InputPattern.find("Type3",0)<std::string::npos){ return 3;
   }else if(InputPattern.find("Type4",0)<std::string::npos){ return 4;
   }else if(InputPattern.find("Type5",0)<std::string::npos){ return 5;
   }else{                                                    return 6;
   }
}

// define the legend corresponding to a Type
std::string LegendFromType(const std::string& InputPattern){
   switch(TypeFromPattern(InputPattern)){
      case 0:  return std::string("Tracker - Only"); break;
      case 1:  return std::string("Tracker + Muon"); break;
      case 2:  return std::string("Tracker + TOF" ); break;
      case 3:  return std::string("Muon - Only"); break;
      case 4:  return std::string("Q>1"); break;
      case 5:  return std::string("Q<1"); break;
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
   if(PreviousV>0){if(debug){printf("FindIntersectionBetweenTwoGraphs returns -1 because observed xsection is above th xsection for the first mass already : %f vs %f\n", obs->Eval(PreviousX, 0, ""), th->Eval(PreviousX, 0, ""));}return -1;}
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
// some functions that where defined in the nSigma.cc code of Greg Landsberg
// v1.1, updated by Greg Landsberg 5/21/09
//
// This Root code computes the probability for the expectd background Bkgr with the FRACTIONAL
// uncertainty sFfrac (i.e., B = Bkgr*(1 +/- sBfrac)) to fluctuate to or above the
// observed number of events nobs
//
// To find 3/5 sigma evidence/discovery points, one should use nobs = int(<S+B>),
// where <S+B> is the expected mean of the signal + background.
//
// Usage: nSigma(Double_t Bkgr, Int_t nobs, Double_t sBfrac) returns the one sided probability
// of an upward backround fluctuations, expressed in Gaussian sigmas. It is suggested to run
// this code in the compiled mode, i.e. .L nSigma.cc++
//
// 5 sigma corresponds to the p-value of 2.85E-7; 3 sigma corresponds to p-value of 1.35E-3
//---------------------------------------------------------------------------------------------

namespace nSigma{
   Double_t nSigma(Double_t Bkgr, Int_t nobs, Double_t sBfrac);
   Double_t Poisson(Double_t Mu, Int_t n);
   Double_t PoissonAve(Double_t Mu, Int_t n, Double_t ErrMu);
   Double_t Inner(Double_t *x, Double_t *par);
   Double_t ErfcInverse(Double_t x);

   static const Double_t Eps = 1.e-9;

   Double_t nSigma(Double_t Bkgr, Int_t nobs, Double_t sBfrac) {
           //caluculate poisson probability 
           Double_t probLess = 0.;
           Int_t i = nobs;
           Double_t eps = 0;
           do {
                   eps = 2.*PoissonAve(Bkgr, i++, sBfrac*Bkgr);
                   probLess += eps;
           } while (eps > 0.);
           return TMath::Sqrt(2.)*ErfcInverse(probLess);	
   }

   Double_t Poisson(Double_t Mu, Int_t n){
           Double_t logP;
           logP = -Mu + n*TMath::Log(Mu);
           for (Int_t i = 2; i <= n; i++) logP -= TMath::Log((Double_t) i);
           return TMath::Exp(logP);
   }

   Double_t PoissonAve(Double_t Mu, Int_t n, Double_t ErrMu) {
           Double_t par[3], retval;
           par[0]=Mu;  // background value
           par[1]=ErrMu;  // background error
           par[2]=n; // n
           TF1 *in = new TF1("Inner",Inner,0.,Mu + 5.*ErrMu,3);   
           Double_t low = Mu > 5.*ErrMu ? Mu - 5.*ErrMu : 0.;
           if (ErrMu < Eps) {
                   Double_t x[1];
                   x[0] = Mu;
                   par[1] = 1./sqrt(2.*TMath::Pi());
                   retval = Inner(x,par);
           } else retval = in->Integral(low,Mu+5.*ErrMu,par);
           delete in;
           return retval;
   }

   Double_t Inner(Double_t *x, Double_t *par){
       Double_t B, sB;
       B = par[0];
       sB = par[1];
       Int_t n = par[2];
       return 1./sqrt(2.*TMath::Pi())/sB*exp(-(x[0]-B)*(x[0]-B)/2./sB/sB)*Poisson(x[0],n);
   }

   Double_t ErfcInverse(Double_t x){
           Double_t xmin = 0., xmax = 20.;
           Double_t sig = xmin;
           if (x >=1) return sig;
           do {
                   Double_t erf = TMath::Erfc(sig);
                   if (erf > x) {    xmin = sig;
                                     sig = (sig+xmax)/2.;
                   } else {          xmax = sig;
                                     sig = (xmin + sig)/2.;
                   }
           } while (xmax - xmin > Eps);
           return sig;
   }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Genertic code related to samples processing in FWLITE --> functions below will be loaded only if FWLITE compiler variable is defined

#ifdef FWLITE
double DistToHSCP        (const susybsm::HSCParticle& hscp, const std::vector<reco::GenParticle>& genColl, int& IndexOfClosest);
int    HowManyChargedHSCP(const std::vector<reco::GenParticle>& genColl);
void   GetGenHSCPBeta    (const std::vector<reco::GenParticle>& genColl, double& beta1, double& beta2, bool onlyCharged=true);

// compute the distance between a "reconstructed" HSCP candidate and the closest geenrated HSCP
double DistToHSCP (const susybsm::HSCParticle& hscp, const std::vector<reco::GenParticle>& genColl, int& IndexOfClosest){
   reco::TrackRef   track;
   if(TypeMode!=3) track = hscp.trackRef();
   else {
     reco::MuonRef muon = hscp.muonRef();
     if(muon.isNull()) return false;
     track = muon->standAloneMuon();
   }
   if(track.isNull())return false;

   double RMin = 9999; IndexOfClosest=-1;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000 && AbsPdg!=17)continue;    
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
      if(AbsPdg<1000000 && AbsPdg!=17)continue;
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
      if(AbsPdg<1000000 && AbsPdg!=17)continue;
      if(onlyCharged && (AbsPdg==1000993 || AbsPdg==1009313 || AbsPdg==1009113 || AbsPdg==1009223 || AbsPdg==1009333 || AbsPdg==1092114 || AbsPdg==1093214 || AbsPdg==1093324))continue; //Skip neutral gluino RHadrons
      if(onlyCharged && (AbsPdg==1000622 || AbsPdg==1000642 || AbsPdg==1006113 || AbsPdg==1006311 || AbsPdg==1006313 || AbsPdg==1006333))continue;  //skip neutral stop RHadrons
      if(beta1<0){beta1=genColl[g].p()/genColl[g].energy();}else if(beta2<0){beta2=genColl[g].p()/genColl[g].energy();return;}
   }
}

#include "TVector3.h"
double deltaROpositeTrack(const susybsm::HSCParticleCollection& hscpColl, const susybsm::HSCParticle& hscp){
   reco::TrackRef track1=hscp.trackRef();

   double maxDr=-0.1;
   for(unsigned int c=0;c<hscpColl.size();c++){
      reco::TrackRef track2;
      if(!hscpColl[c].trackRef().isNull()){
         track2=hscpColl[c].trackRef();
      }else if(!hscpColl[c].muonRef().isNull() && hscpColl[c].muonRef()->combinedQuality().updatedSta){
         track2= hscpColl[c].muonRef()->standAloneMuon();
      }else{
         continue;
      }

      if(fabs(track1->pt()-track2->pt())<1 && deltaR(track1->eta(), track1->phi(), track2->eta(), track2->phi())<0.1)continue; //Skip same tracks
//      double dR = deltaR(-1*track1->eta(), M_PI+track1->phi(), track2->eta(), track2->phi());
      TVector3 v1 = TVector3(track1->momentum().x(), track1->momentum().y(), track1->momentum().z());
      TVector3 v2 = TVector3(track2->momentum().x(), track2->momentum().y(), track2->momentum().z());
      double dR = v1.Angle(v2);
      if(dR>maxDr)maxDr=dR;
   }
   return maxDr;
}

#endif


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Handfull class to check for duplicated events

class DuplicatesClass{
   private :
      typedef std::map<std::pair<unsigned int, unsigned int>, int > RunEventHashMap;
      RunEventHashMap map;
   public :
        DuplicatesClass(){}
        ~DuplicatesClass(){}
        void Clear(){map.clear();}
        bool isDuplicate(unsigned int Run, unsigned int Event){
	   RunEventHashMap::iterator it = map.find(std::make_pair(Run,Event));
           if(it==map.end()){
   	      map[std::make_pair(Run,Event)] = 1;
              return false;
           }else{
              map[std::make_pair(Run,Event)]++;
           }
           return true;
        }

        void printDuplicate(){
           printf("Duplicate event summary:\n##########################################");
           for(RunEventHashMap::iterator it = map.begin(); it != map.end(); it++){
              if(it->second>1)printf("Run %6i Event %10i is duplicated (%i times)\n",it->first.first, it->first.second, it->second);
           }          
           printf("##########################################");
        }
};
 

#ifdef FWLITE
bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut, int NObjectAboveThreshold, bool averageThreshold)
{
   unsigned int filterIndex = trEv.filterIndex(InputPath);
   //if(filterIndex<trEv.sizeFilters())printf("SELECTED INDEX =%i --> %s    XXX   %s\n",filterIndex,trEv.filterTag(filterIndex).label().c_str(), trEv.filterTag(filterIndex).process().c_str());

   if (filterIndex<trEv.sizeFilters()){
      const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
      const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
      const int nI(VIDS.size());
      const int nK(KEYS.size());
      assert(nI==nK);
      const int n(std::max(nI,nK));
      const trigger::TriggerObjectCollection& TOC(trEv.getObjects());


      if(!averageThreshold){
         int NObjectAboveThresholdObserved = 0;
         for (int i=0; i!=n; ++i) {
	   if(TOC[KEYS[i]].pt()> NewThreshold && fabs(TOC[KEYS[i]].eta())<etaCut) NObjectAboveThresholdObserved++;
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if(NObjectAboveThresholdObserved>=NObjectAboveThreshold)return true;

      }else{
         std::vector<double> ObjPt;

         for (int i=0; i!=n; ++i) {
            ObjPt.push_back(TOC[KEYS[i]].pt());
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if((int)(ObjPt.size())<NObjectAboveThreshold)return false;
         std::sort(ObjPt.begin(), ObjPt.end());

         double Average = 0;
         for(int i=0; i<NObjectAboveThreshold;i++){
            Average+= ObjPt[ObjPt.size()-1-i];
         }Average/=NObjectAboveThreshold;
         //cout << "AVERAGE = " << Average << endl;

         if(Average>NewThreshold)return true;
      }
   }
   return false;
}

reco::DeDxData* dEdxOnTheFly(const fwlite::ChainEvent& ev, const reco::TrackRef&   track, const reco::DeDxData* dedxSObj, bool reverseProb=false){
     fwlite::Handle<HSCPDeDxInfoValueMap> dEdxHitsH;
     dEdxHitsH.getByLabel(ev, "dedxHitInfo");
     if(!dEdxHitsH.isValid()){printf("Invalid dEdxHitInfo\n");return NULL;}
     const ValueMap<HSCPDeDxInfo>& dEdxHitMap = *dEdxHitsH.product();

     const HSCPDeDxInfo& hscpHitsInfo = dEdxHitMap.get((size_t)track.key());

     std::vector<double> vect_probs;
     for(unsigned int h=0;h<hscpHitsInfo.charge.size();h++){
        DetId detid(hscpHitsInfo.detIds[h]);  
        if(detid.subdetId()<3)continue; // skip pixels
        if(!hscpHitsInfo.shapetest[h])continue;

        //Remove hits close to the border
        //for unknown reasons, localx,localy, modwidth,modlength is not saved in all ntuples!
        //double absDistEdgeXNorm = 1-fabs(hscpHitsInfo.localx[h])/(hscpHitsInfo.modwidth [h]/2.0);
        //double absDistEdgeYNorm = 1-fabs(hscpHitsInfo.localy[h])/(hscpHitsInfo.modlength[h]/2.0);
        //if(detid.subdetId()==1 && (absDistEdgeXNorm<0.05  || absDistEdgeYNorm<0.01)) continue;
        //if(detid.subdetId()==2 && (absDistEdgeXNorm<0.05  || absDistEdgeYNorm<0.01)) continue; 
        //if(detid.subdetId()==3 && (absDistEdgeXNorm<0.005 || absDistEdgeYNorm<0.04)) continue;  
        //if(detid.subdetId()==4 && (absDistEdgeXNorm<0.005 || absDistEdgeYNorm<0.02)) continue;  
        //if(detid.subdetId()==5 && (absDistEdgeXNorm<0.005 || absDistEdgeYNorm<0.02 || absDistEdgeYNorm>0.97)) continue;
        //if(detid.subdetId()==6 && (absDistEdgeXNorm<0.005 || absDistEdgeYNorm<0.03 || absDistEdgeYNorm>0.8)) continue;

        vect_probs.push_back(reverseProb?1.0-hscpHitsInfo.probability[h]:hscpHitsInfo.probability[h]);
     }
     int size = vect_probs.size();

     //Prod
//     double P = 1;
//     for(int i=0;i<size;i++){
//        if(vect_probs[i]<=0.0001){P *= pow(0.0001       , 1.0/size);}
//        else                     {P *= pow(vect_probs[i], 1.0/size);}
//     }

     //Ias
     double P = 1.0/(12*size);
     std::sort(vect_probs.begin(), vect_probs.end(), std::less<double>() );
     for(int i=1;i<=size;i++){
        P += vect_probs[i-1] * pow(vect_probs[i-1] - ((2.0*i-1.0)/(2.0*size)),2);
     }
     P *= (3.0/size);


     if(size<=0)P=-1;

//     printf("%f vs %f (%i vs %i)\n",dedxSObj->dEdx(), P, dedxSObj->numberOfMeasurements(), size);

//                  dedxSObj = new DeDxData(1.0-dedxSObj->dEdx(), dedxSObj->numberOfSaturatedMeasurements(), dedxSObj->numberOfMeasurements());

     return new DeDxData(P, dedxSObj->numberOfSaturatedMeasurements(), size);
}



#endif

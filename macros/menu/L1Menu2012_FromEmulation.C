/*

 E.P., March 28, 2012.
 
 Run over events firing L1_ZeroBias.
 The logic of the L1 bits of the 2012 collision menu is emulated.
 Enter properly the prescale of L1_ZeroBias !!

*/


#include "L1Ntuple.h"
#include "hist.C"
#include "Style.C"

#include "TLegend.h"
#include "TMath.h"
#include "TText.h"
#include "TH2.h"
#include "TAxis.h"
#include "TString.h"

#include <iostream>
#include <vector>
#include <map>
#include <set>


#define NLUMIS 21 // the same for all currently used ntuples

// -- Run 179828, LS 140 - 160, PU=33:
#define LUMIREF 0.509
#define L1NtupleFileName "~heistera/scratch0/L1Ntuples/L1TreeL1Accept_179828_LS_140_160.root";

#define LUMI 50.

// -- Huge prescale value for seeds "for lower PU"
#define INFTY 10000

#define ZEROBIAS_PRESCALE 3


TH1F* h_Cross;
TH1F* h_Jets;
TH1F* h_Sums;
TH1F* h_Egamma;
TH1F* h_Muons;

TH1F* h_Block;
TH2F* cor_Block;

int NPAGS = 6;
TH2F* cor_PAGS;
TH1F* h_PAGS_pure;
TH1F* h_PAGS_shared;


// -- For the pure rates :

const int N128 = 128;	// could be > 128 for "test seeds"
int kOFFSET = 0;
bool TheTriggerBits[N128] ; 	// contains the emulated triggers for each event
TH1F* h_All;	// one bin for each trigger. Fill bin i if event fires trigger i.
TH1F* h_Pure;	// one bin for each trigger. Fill bin i if event fires trigger i and NO OTHER TRIGGER.

// --- Methods to scale L1 jets for new HF LUTs

// correction by 8% overall (from HCAL January 2012)
Double_t CorrectedL1FwdJetPtByFactor(Bool_t isFwdJet, Double_t JetPt)
{
	Double_t JetPtcorr = JetPt;
	if (isFwdJet) { JetPtcorr = JetPt*1.08; }
	return JetPtcorr;
}

// correction for HF bins (from HCAL January 2012)
Size_t   JetHFiEtabins   = 13;
Int_t    JetHFiEtabin[]  = {29,30,31,32,33,34,35,36,37,38,39,40,41};
Double_t JetHFiEtacorr[] = {0.982,0.962,0.952, 0.943,0.947,0.939, 0.938,0.935,0.934, 0.935,0.942,0.923,0.914};

Double_t CorrectedL1JetPtByHFtowers(Double_t JetiEta,Double_t JetPt)
{
	Double_t JetPtcorr   = JetPt;
	Int_t    iJetiEtabin = 0;
	for (iJetiEtabin=0; iJetiEtabin<JetHFiEtabins; iJetiEtabin++) {
		if (JetHFiEtabin[iJetiEtabin]==JetiEta) {
			JetPtcorr = JetPt * (1+(1-JetHFiEtacorr[iJetiEtabin]));
		}
	}
		return JetPtcorr;
}

// correction for RCT->GCT bins (from HCAL January 2012)
// HF from 29-41, first 3 HF trigger towers 3 iEtas, last highest eta HF trigger tower 4 iEtas; each trigger tower is 0.5 eta, RCT iEta from 0->21 (left->right)
Double_t JetRCTHFiEtacorr[]  = {0.965,0.943,0.936,0.929}; // from HF iEta=29 to 41 (smaller->higher HF iEta)

Double_t CorrectedL1JetPtByGCTregions(Double_t JetiEta,Double_t JetPt)
{
	Double_t JetPtcorr   = JetPt;

/*	if (JetiEta==0 || JetiEta==21) {
		JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[3]));
	}
	else if (JetiEta==1 || JetiEta==20) {
		JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[2]));
	}
	else if (JetiEta==2 || JetiEta==19) {
		JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[1]));
	}
	else if (JetiEta==3 || JetiEta==18) {
		JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[0]));
	}
*/
    return JetPtcorr;
}



// --- Methods from Mia for the correlation conditions

size_t PHIBINS = 18;
double PHIBIN[] = {10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350};

size_t ETABINS = 23;
double ETABIN[] = {-5.,-4.5,-4.,-3.5,
                   -3.,-2.172,-1.74,-1.392,-1.044,-0.696,-0.348,
                   0,
                   0.348,0.696,1.044,1.392,1.74,2.172,3.,
                   3.5,4.,4.5,5.};

size_t ETAMUBINS = 65;
double ETAMU[] = { -2.45,
-2.4,
-2.35,
-2.3,
-2.25,
-2.2,
-2.15,
-2.1,
-2.05,
-2,
-1.95,
-1.9,
-1.85,
-1.8,
-1.75,
-1.7,
-1.6,
-1.5,
-1.4,
-1.3,
-1.2,
-1.1,
-1,
-0.9,
-0.8,
-0.7,
-0.6,
-0.5,
-0.4,
-0.3,
-0.2,
-0.1,
0,
0.1,
0.2,
0.3,
0.4,
0.5,
0.6,
0.7,
0.8,
0.9,
1,
1.1,
1.2,
1.3,
1.4,
1.5,
1.6,
1.7,
1.75,
1.8,
1.85,
1.9,
1.95,
2,
2.05,
2.1,
2.15,
2.2,
2.25,
2.3,
2.35,
2.4,
2.45
}  ;


int etaMuIdx(double eta) {
  size_t etaIdx = 0.;
  for (size_t idx=0; idx<ETAMUBINS; idx++) {
    if (eta>=ETAMU[idx] and eta<ETAMU[idx+1])
      etaIdx = idx;
  }
  return int(etaIdx);
}


int etaINjetCoord(double eta){
  size_t etaIdx = 0.;
  for (size_t idx=0; idx<ETABINS; idx++) {
    if (eta>=ETABIN[idx] and eta<ETABIN[idx+1])
      etaIdx = idx;
  }
  return int(etaIdx);
}


double degree(double radian) {
  if (radian<0)
    return 360.+(radian/TMath::Pi()*180.);
  else
    return radian/TMath::Pi()*180.;
}

int phiINjetCoord(double phi) {
  size_t phiIdx = 0;
  double phidegree = degree(phi);
  for (size_t idx=0; idx<PHIBINS; idx++) {
    if (phidegree>=PHIBIN[idx] and phidegree<PHIBIN[idx+1])
      phiIdx = idx;
    else if (phidegree>=PHIBIN[PHIBINS-1] || phidegree<=PHIBIN[0])
      phiIdx = idx;
  }
   phiIdx = phiIdx + 1;
   if (phiIdx == 18)  phiIdx = 0;
  return int(phiIdx);
}

bool correlateInPhi(int jetphi, int muphi, int delta=1) {
 
  bool correlationINphi = fabs(muphi-jetphi)<fabs(2 +delta-1) || fabs(muphi-jetphi)>fabs(PHIBINS-2 - (delta-1) );
  return correlationINphi;

}


bool correlateInEta(int mueta, int jeteta, int delta=1) {
  bool correlationINeta = fabs(mueta-jeteta)<2 + delta-1;
 return correlationINeta;
}



void MyScale(TH1F* h, float scal) {

// -- to set the errors properly

  int nbins = h -> GetNbinsX();

  for (int i=1; i<= nbins; i++)  {
        float val = h -> GetBinContent(i);
        float er = sqrt(val);
        val = val * scal;
        er = er * scal;
        h -> SetBinContent(i,val);
        h -> SetBinError(i,er);
   }


}




// --------------------------------------------------------------------




// --------------------------------------------------------------------
//                       L1Menu2012 macro definition
// --------------------------------------------------------------------

class L1Menu2012 : public L1Ntuple
{
  public :

    //constructor    
    L1Menu2012(std::string filename) : L1Ntuple(filename) {}
    L1Menu2012() {}
    ~L1Menu2012() {}

    //main function macro : arguments can be adpated to your need

    void MyInit();
    void FillBits();

	std::map<std::string, int> Counts;
	std::map<std::string, int> Prescales;
        std::map<std::string, bool> Biased;

	std::map<std::string, float> WeightsPAGs;


    void InsertInMenu(string L1name, bool value);

	bool Cross();
	bool Jets();
	bool EGamma();
	bool Muons();
	bool Sums();

	bool dummy(string L1name);

// -- Cross
   bool Mu_EG(float mucut, float EGcut );
   bool MuOpen_EG(float mucut, float EGcut );
   bool Mu_JetCentral(float mucut, float jetcut );
   bool Mu_DoubleJetCentral(float mucut, float jetcut );
   bool Mu_JetCentral_LowerTauTh(float mucut, float jetcut, float taucut );
   bool Muer_JetCentral(float mucut, float jetcut );
   bool Muer_JetCentral_LowerTauTh(float mucut, float jetcut, float taucut );
   bool Mu_HTT(float mucut, float HTcut );
   bool Muer_ETM(float mucut, float ETMcut );
   bool EG_FwdJet(float EGcut, float FWcut ) ;
   bool EG_HT(float EGcut, float HTcut );
   bool EG_DoubleJetCentral(float EGcut, float jetcut );
   bool DoubleEG_HT(float EGcut, float HTcut );
   bool EGEta2p1_JetCentral(float EGcut, float jetcut);		// delta
   bool EGEta2p1_JetCentral_LowTauTh(float EGcut, float jetcut, float taucut);          // delta
   bool IsoEGEta2p1_JetCentral_LowTauTh(float EGcut, float jetcut, float taucut);          // delta
   bool EGEta2p1_DoubleJetCentral(float EGcut, float jetcut);	// delta
   bool EGEta2p1_DoubleJetCentral_TripleJetCentral(float EGcut, float jetcut2, float jetcut3);   

   bool HTT_HTM(float HTTcut, float HTMcut);
   bool JetCentral_ETM(float jetcut, float ETMcut);
   bool DoubleJetCentral_ETM(float jetcut1, float jetcut2, float ETMcut);
   bool DoubleMu_EG(float mucut, float EGcut );
   bool Mu_DoubleEG(float mucut, float EGcut);

   bool Muer_TripleJetCentral(float mucut, float jet1, float jet2, float jet3);
   bool Mia(float mucut, float jet1, float jet2);	// delta
   bool Mu_JetCentral_delta(float mucut, float ptcut, int IBIT);	// delta
   bool Mu_JetCentral_deltaOut(float mucut, float ptcut, int IBIT); // delta


// -- Jets 
     bool SingleJet(float cut);
     bool SingleJetCentral(float cut);
     bool DoubleJetCentral(float cut1, float cut2);
     bool DoubleJet_Eta1p7_deltaEta4(float cut1, float cut2);
     bool TripleJetCentral(float cut1, float cut2, float cut3);
     bool TripleJet_VBF(float cut1, float cut2, float cut3);

     bool QuadJetCentral(float cut1, float cut2, float cut3, float cut4);
     bool DoubleTauJetEta2p17(float cut1, float cut2);

// -- Sums
    bool ETT(float ETTcut);
    bool HTT(float HTTcut);
    bool ETM(float ETMcut);

// -- Egamma
     bool SingleEG(float cut);
     bool SingleEG_Eta2p1(float cut);
     bool SingleIsoEG_Eta2p1(float cut);

     bool DoubleEG(float cut1, float cut2);
     bool TripleEG(float cut1, float cut2, float cut3);

// -- Muons 
   bool SingleMu(float ptcut, int qualmin=4);
   bool SingleMuEta2p1(float ptcut);
   bool DoubleMu(float cut1, float cut2);	// on top of DoubleMu3
   bool DoubleMuHighQEtaCut(float ptcut, float etacut);
   bool TripleMu(float cut1, float cut2, float cut3, int qualmin);	// on top of DoubleMu3
   bool DoubleMuXOpen(float ptcut);	// on top of SingleMu7
   bool Onia(float ptcut1, float ptcut2, float etacut, int delta);   


	void Loop();


  private :

    //your private methods can be declared here

        bool PhysicsBits[128];
	bool first;

	int insert_ibin;
	bool insert_val[100];
	string insert_names[100];

	int NBITS_MUONS;
	int NBITS_EGAMMA;
	int NBITS_JETS;
	int NBITS_SUMS;
	int NBITS_CROSS;

	set<string> setTOP;
	set<string> setHIGGS;
	set<string> setEXO;
	set<string> setSMP;
	set<string> setBPH;
        set<string> setSUSY;

};


// ------------------------------------------------------------------

void L1Menu2012::InsertInMenu(string L1name, bool value) {

	bool post_prescale = false;

	int prescale = 1;

	map<string, int>::const_iterator it = Prescales.find(L1name);
	if (it == Prescales.end() ) {
	  cout << " --- NO PRESCALE DEFINED FOR " << L1name << " ---  SET P = 1 " << endl;
	}
	else {
           prescale = Prescales[L1name];
	}

	if (prescale >0) {
           Counts[L1name] ++;
           int n = Counts[L1name];
           if ( n % prescale == 0) post_prescale = value; 
	}

	insert_names[insert_ibin] = L1name;
	insert_val[insert_ibin] = post_prescale ;

	insert_ibin ++;

}



// ------------------------------------------------------------------

void L1Menu2012::FillBits() {

//      Fill the physics bits:

        for (int ibit=0; ibit < 128; ibit++) {
                PhysicsBits[ibit] = 0;
                if (ibit<64) {
                  PhysicsBits[ibit] = (gt_->tw1[2]>>ibit)&1;
                }
                else {
                  PhysicsBits[ibit] = (gt_->tw2[2]>>(ibit-64))&1;
                }
        }
        
        
}       

// ------------------------------------------------------------------

void L1Menu2012::MyInit() {


// --- The seeds per group

setTOP.insert("L1_HTT150") ;
setTOP.insert("L1_HTT175") ;
setTOP.insert("L1_HTT200") ;
setTOP.insert("L1_SingleEG18er") ;
setTOP.insert("L1_SingleIsoEG18er") ;
setTOP.insert("L1_SingleEG20") ;
setTOP.insert("L1_SingleIsoEG20er") ;
setTOP.insert("L1_SingleEG22") ;
setTOP.insert("L1_SingleEG24") ;
setTOP.insert("L1_SingleEG30") ;
setTOP.insert("L1_DoubleEG_13_7") ;
setTOP.insert("L1_QuadJetC36") ;
setTOP.insert("L1_QuadJetC40") ;
setTOP.insert("L1_Mu12_EG7") ;
setTOP.insert("L1_Mu3p5_EG12") ;
setTOP.insert("L1_SingleMu14er") ;
setTOP.insert("L1_SingleMu16er") ;
setTOP.insert("L1_SingleMu18er") ;
setTOP.insert("L1_SingleMu20er") ;
setTOP.insert("L1_SingleMu25er") ;
setTOP.insert("L1_DoubleMu_12_5") ;
setTOP.insert("L1_DoubleMu_10_3p5") ;


setHIGGS.insert("L1_ETM30") ;
setHIGGS.insert("L1_ETM36") ;
setHIGGS.insert("L1_ETM40") ;
setHIGGS.insert("L1_SingleEG7") ;
setHIGGS.insert("L1_SingleEG12") ;
setHIGGS.insert("L1_SingleEG18er") ;
setHIGGS.insert("L1_SingleIsoEG18er") ;
setHIGGS.insert("L1_SingleEG20") ;
setHIGGS.insert("L1_SingleIsoEG20er") ;
setHIGGS.insert("L1_SingleEG22") ;
setHIGGS.insert("L1_DoubleEG_13_7") ;
setHIGGS.insert("L1_TripleEG_12_7_5") ;
setHIGGS.insert("L1_SingleJet128") ;
setHIGGS.insert("L1_DoubleJetC36") ;
setHIGGS.insert("L1_DoubleJetC44_Eta1p74_WdEta4") ;
setHIGGS.insert("L1_DoubleJetC52") ;
setHIGGS.insert("L1_DoubleJetC56_Eta1p74_WdEta4") ;
setHIGGS.insert("L1_DoubleJetC56") ;
setHIGGS.insert("L1_DoubleJetC64") ;
setHIGGS.insert("L1_TripleJet_64_44_24_VBF") ;
setHIGGS.insert("L1_TripleJet_64_48_28_VBF") ;
setHIGGS.insert("L1_TripleJet_68_48_32_VBF") ;
setHIGGS.insert("L1_DoubleTauJet44er") ;
setHIGGS.insert("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12") ;
setHIGGS.insert("L1_Mu10er_JetC32") ;
setHIGGS.insert("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12") ;
setHIGGS.insert("L1_EG18er_JetC_Cen28_Tau20_dPhi1") ;
setHIGGS.insert("L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1") ;
setHIGGS.insert("L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1") ;
setHIGGS.insert("L1_EG18er_JetC_Cen36_Tau28_dPhi1") ;
setHIGGS.insert("L1_Mu12_EG7") ;
setHIGGS.insert("L1_MuOpen_EG12") ;
setHIGGS.insert("L1_Mu3p5_EG12") ;
setHIGGS.insert("L1_SingleMu3") ;
setHIGGS.insert("L1_SingleMu7") ;
setHIGGS.insert("L1_SingleMu18er") ;
setHIGGS.insert("L1_SingleMu20er") ;
setHIGGS.insert("L1_DoubleMu_10_Open") ;
setHIGGS.insert("L1_DoubleMu_10_3p5") ;

setSUSY.insert("L1_HTT100") ;
setSUSY.insert("L1_HTT125") ;
setSUSY.insert("L1_HTT150") ;
setSUSY.insert("L1_HTT175") ;
setSUSY.insert("L1_HTT200") ;
setSUSY.insert("L1_SingleEG20") ;
setSUSY.insert("L1_SingleIsoEG20er") ;
setSUSY.insert("L1_SingleEG22") ;
setSUSY.insert("L1_SingleEG24") ;
setSUSY.insert("L1_DoubleEG_13_7") ;
setSUSY.insert("L1_TripleEG7") ;
setSUSY.insert("L1_SingleJet128") ;
setSUSY.insert("L1_DoubleJetC52") ;
setSUSY.insert("L1_DoubleJetC56") ;
setSUSY.insert("L1_DoubleJetC64") ;
setSUSY.insert("L1_QuadJetC32") ;
setSUSY.insert("L1_QuadJetC36") ;
setSUSY.insert("L1_QuadJetC40") ;
setSUSY.insert("L1_Mu0_HTT50") ;
setSUSY.insert("L1_Mu0_HTT100") ;
setSUSY.insert("L1_Mu4_HTT125") ;
setSUSY.insert("L1_Mu8_DoubleJetC20") ;
setSUSY.insert("L1_DoubleEG6_HTT100") ;
setSUSY.insert("L1_DoubleEG6_HTT125") ;
setSUSY.insert("L1_EG8_DoubleJetC20") ;
setSUSY.insert("L1_Mu12_EG7") ;
setSUSY.insert("L1_MuOpen_EG12") ;
setSUSY.insert("L1_Mu3p5_EG12") ;
setSUSY.insert("L1_DoubleMu3p5_EG5") ;
setSUSY.insert("L1_DoubleMu5_EG5") ;
setSUSY.insert("L1_Mu5_DoubleEG5") ;
setSUSY.insert("L1_Mu5_DoubleEG6") ;
setSUSY.insert("L1_DoubleJetC36_ETM30") ;
setSUSY.insert("L1_DoubleJetC44_ETM30") ;
setSUSY.insert("L1_SingleMu14er") ;
setSUSY.insert("L1_SingleMu16er") ;
setSUSY.insert("L1_SingleMu18er") ;
setSUSY.insert("L1_SingleMu20er") ;
setSUSY.insert("L1_TripleMu0") ;
setSUSY.insert("L1_TripleMu0_HighQ") ;

setEXO.insert("L1_ETM36") ;
setEXO.insert("L1_ETM40") ;
setEXO.insert("L1_ETM50") ;
setEXO.insert("L1_ETM70") ;
setEXO.insert("L1_ETM100") ;
setEXO.insert("L1_HTT150") ;
setEXO.insert("L1_HTT175") ;
setEXO.insert("L1_HTT200") ;
setEXO.insert("L1_ETT300") ;
setEXO.insert("L1_SingleEG20") ;
setEXO.insert("L1_SingleIsoEG20er") ;
setEXO.insert("L1_SingleEG22") ;
setEXO.insert("L1_SingleEG24") ;
setEXO.insert("L1_SingleEG30") ;
setEXO.insert("L1_DoubleEG_13_7") ;
setEXO.insert("L1_SingleJet36") ;
setEXO.insert("L1_SingleJet52") ;
setEXO.insert("L1_SingleJet68") ;
setEXO.insert("L1_SingleJet92") ;
setEXO.insert("L1_SingleJet128") ;
setEXO.insert("L1_QuadJetC36") ;
setEXO.insert("L1_QuadJetC40") ;
setEXO.insert("L1_Mu12_EG7") ;
setEXO.insert("L1_Mu3p5_EG12") ;
setEXO.insert("L1_SingleMu16er") ;
setEXO.insert("L1_SingleMu18er") ;
setEXO.insert("L1_SingleMu20er") ;
setEXO.insert("L1_SingleMu25er") ;
setEXO.insert("L1_DoubleMu0er_HighQ") ;
setEXO.insert("L1_DoubleMu_10_3p5") ;
setEXO.insert("L1_TripleMu0_HighQ") ;
setEXO.insert("L1_SingleJetC32_NotBptxOR") ;
setEXO.insert("L1_SingleJetC20_NotBptxOR") ;
setEXO.insert("L1_SingleMu6_NotBptxOR") ;
setEXO.insert("L1_DoubleMuHighQ_3_3_Eta2p1_WdEta22");

setSMP.insert("L1_SingleEG7") ;
setSMP.insert("L1_SingleEG12") ;
setSMP.insert("L1_SingleIsoEG18er") ;
setSMP.insert("L1_SingleEG20") ;
setSMP.insert("L1_SingleIsoEG20er") ;
setSMP.insert("L1_SingleEG22") ;
setSMP.insert("L1_SingleEG24") ;
setSMP.insert("L1_SingleEG30") ;
setSMP.insert("L1_DoubleEG_13_7") ;
setSMP.insert("L1_SingleJet16") ;
setSMP.insert("L1_SingleJet36") ;
setSMP.insert("L1_SingleJet52") ;
setSMP.insert("L1_SingleJet68") ;
setSMP.insert("L1_SingleJet92") ;
setSMP.insert("L1_SingleJet128") ;
setSMP.insert("L1_EG22_ForJet24") ;
setSMP.insert("L1_EG22_ForJet32") ;
setSMP.insert("L1_Mu12_EG7") ;
setSMP.insert("L1_MuOpen_EG12") ;
setSMP.insert("L1_Mu3p5_EG12") ;
setSMP.insert("L1_SingleMu7") ;
setSMP.insert("L1_SingleMu16er") ;
setSMP.insert("L1_SingleMu18er") ;
setSMP.insert("L1_SingleMu20er") ;
setSMP.insert("L1_DoubleMu_12_5") ;
setSMP.insert("L1_DoubleMu_10_Open") ;
setSMP.insert("L1_DoubleMu_10_3p5") ;


setBPH.insert("L1_SingleMu3") ;
setBPH.insert("L1_DoubleMu0er_HighQ") ;
setBPH.insert("L1_DoubleMuHighQ_3_3_Eta2p1_WdEta22") ;
setBPH.insert("L1_DoubleMuHighQ_5_0_Eta2p1_WdEta22") ;
setBPH.insert("L1_TripleMu0_HighQ") ;



// -- LUMI70
if ( LUMI == 70 ) {

// -- Cross 
 Prescales["L1_Mu0_HTT50"] = INFTY;
 Prescales["L1_Mu0_HTT100"] = INFTY;
 Prescales["L1_Mu4_HTT125"] = 1;

 Prescales["L1_Mu12er_ETM20"] = 2;

 // Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = INFTY;
 Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 40;
 Prescales["L1_Mu10er_JetC32"] = INFTY ;
 Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

 Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 150 *2;  
 Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 2 *5;
 Prescales["L1_Mu8_DoubleJetC20"] = 200;
 // Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;
 Prescales["L1_EG22_ForJet24"] = 1;
 Prescales["L1_EG22_ForJet32"] = 1;
 Prescales["L1_DoubleEG6_HTT100"] = INFTY;
 Prescales["L1_DoubleEG6_HTT125"] = 1;

 Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = INFTY;
 Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
 Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
 Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = INFTY;

 Prescales["L1_EG8_DoubleJetC20"] = 1000 ;	


 Prescales["L1_Mu12_EG7"] = 1;

 Prescales["L1_DoubleMu3p5_EG5"] = INFTY ;
 Prescales["L1_DoubleMu5_EG5"] = 1;

 // Prescales["L1_Mu3p5_DoubleEG5"] = 0 ;
 Prescales["L1_Mu5_DoubleEG5"] = INFTY ;
 Prescales["L1_Mu5_DoubleEG6"] = 1;

 Prescales["L1_MuOpen_EG12"] = INFTY;
 Prescales["L1_Mu3p5_EG12"] = 1;
 Prescales["L1_DoubleJetC36_ETM30"] = INFTY;
 Prescales["L1_DoubleJetC44_ETM30"] = 1;

// -- Jets 
 Prescales["L1_SingleJet16"] = 0;
 Prescales["L1_SingleJet36"] = 6000;
 Prescales["L1_SingleJet52"] = 500;
 Prescales["L1_SingleJet68"] = 100;
 Prescales["L1_SingleJet92"] = 20 ;
 Prescales["L1_SingleJet128"] =1;

 Prescales["L1_DoubleJetC36"] = 320;

 // Prescales["L1_DoubleJet44_Central"] = 0;
 // Prescales["L1_DoubleJetC52"] = 40;
 // Prescales["L1_DoubleJetC56"] = 1;

 // Prescales["L1_DoubleJet44_Central"] = 0;

  Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
  Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 6;

 Prescales["L1_DoubleJetC52"] = 0;
 Prescales["L1_DoubleJetC56"] = INFTY;

 Prescales["L1_DoubleJetC64"] = 1;

 Prescales["L1_TripleJet_64_44_24_VBF"] = 0;
 Prescales["L1_TripleJet_64_48_28_VBF"] = INFTY;
 Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

 Prescales["L1_TripleJetC_52_28_28"] = 100;

 Prescales["L1_QuadJetC32"] = INFTY;		// better estimate on 178208
 Prescales["L1_QuadJetC36"] = INFTY;
 Prescales["L1_QuadJetC40"] = 1;

 Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
  Prescales["L1_ETM30"] = 300;
  Prescales["L1_ETM36"] = INFTY;
  Prescales["L1_ETM40"] = 1;
  Prescales["L1_ETM50"] = 1;
  Prescales["L1_ETM70"] = 1;
  Prescales["L1_ETM100"] = 1;
  
  Prescales["L1_HTT50"] = 0;
  // Prescales["L1_HTT100"] = 40;
  // Prescales["L1_HTT125"] = 20;

  Prescales["L1_HTT100"] = INFTY;
  Prescales["L1_HTT125"] = INFTY;
  Prescales["L1_HTT150"] = INFTY;
  // Prescales["L1_HTT175"] = INFTY;
  Prescales["L1_HTT175"] = 1;
  Prescales["L1_HTT200"] = 1;
  
  Prescales["L1_ETT300"] = 1;



// -- Egamma
 Prescales["L1_SingleEG5"] = 4500;
 Prescales["L1_SingleEG7"] = 800;
 Prescales["L1_SingleEG12"] = 300;
 Prescales["L1_SingleEG18er"] = 80; 
 Prescales["L1_SingleIsoEG18er"] = 10;
 Prescales["L1_SingleEG20"] = INFTY;
 Prescales["L1_SingleIsoEG20_Eta2p1"] = 0;
 Prescales["L1_SingleEG22"] = 1;
 Prescales["L1_SingleEG24"] = 1;
 Prescales["L1_SingleEG30"] = 1;
 
 // Prescales["L1_DoubleEG_15_5"] = 8;
 Prescales["L1_DoubleEG_13_7"] = 1;
 
 Prescales["L1_TripleEG7"] = 1;
 Prescales["L1_TripleEG_12_7_5"] = 1;



// -- Muons 

 Prescales["L1_SingleMu12"] = 0;

 Prescales["L1_SingleMu20"] = 150;
 Prescales["L1_SingleMu16"] = 150;
 Prescales["L1_SingleMu12"] = 300;
 Prescales["L1_SingleMu7"] = 600;
 Prescales["L1_SingleMu3"] = 4000;
 Prescales["L1_SingleMuOpen"] = 7000;
 Prescales["L1_DoubleMu0"] = 300;

 Prescales["L1_SingleMu25er"] = 1;
 Prescales["L1_SingleMu20er"] = 1;
 Prescales["L1_SingleMu14er"] = 75;
 Prescales["L1_SingleMu18er"] = 1;
 // Prescales["L1_SingleMu16er"] = INFTY;
 Prescales["L1_SingleMu16er"] = 1;
 Prescales["L1_DoubleMu_12_5"] = 1;
 Prescales["L1_DoubleMu5"] = 50;
 // Prescales["L1_DoubleMu0_HighQ_Eta1p7"] = 1;
 // Prescales["L1_DoubleMu3p5_HighQ_Eta2p1"] = 1;

  Prescales["L1_DoubleMu0er_HighQ"] = 50;
  Prescales["L1_DoubleMuHighQ_3_3_Eta2p1_WdEta22"] = 1;
  Prescales["L1_DoubleMuHighQ_5_0_Eta2p1_WdEta22"] = 1;

 Prescales["L1_DoubleMu_10_Open"] = INFTY ;
 Prescales["L1_DoubleMu_10_3p5"] = 1;
 Prescales["L1_TripleMu0"] = INFTY;
 Prescales["L1_TripleMu0_HighQ"] = 1;
 Prescales["L1_TripleMu_5_5_0"] = 1;

}


// -- LUMI70  EMERGENCY !
if ( LUMI == 70.001 ) {

// -- Cross 
 Prescales["L1_Mu0_HTT50"] = INFTY;
 Prescales["L1_Mu0_HTT100"] = INFTY;
 Prescales["L1_Mu4_HTT125"] = 1;

 Prescales["L1_Mu12er_ETM20"] = 2;

 Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 40;
 Prescales["L1_Mu10er_JetC32"] = INFTY ;
 Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

 Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 150 *2;
 Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 2 *5;
 Prescales["L1_Mu8_DoubleJetC20"] = 200;
 // Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;
 Prescales["L1_EG22_ForJet24"] = 1;
 Prescales["L1_EG22_ForJet32"] = 1;
 Prescales["L1_DoubleEG6_HTT100"] = INFTY;
 Prescales["L1_DoubleEG6_HTT125"] = 1;

 Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = INFTY;
 // Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = INFTY;
 Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = INFTY;
 Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
 Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = INFTY;

 Prescales["L1_EG8_DoubleJetC20"] = 1000 ;


 Prescales["L1_Mu12_EG7"] = 1;

 Prescales["L1_DoubleMu3p5_EG5"] = INFTY ;
 Prescales["L1_DoubleMu5_EG5"] = 1;

 // Prescales["L1_Mu3p5_DoubleEG5"] = 0 ;
 Prescales["L1_Mu5_DoubleEG5"] = INFTY ;
 Prescales["L1_Mu5_DoubleEG6"] = 1;

 Prescales["L1_MuOpen_EG12"] = INFTY;
 Prescales["L1_Mu3p5_EG12"] = 1;
 Prescales["L1_DoubleJetC36_ETM30"] = INFTY;
 Prescales["L1_DoubleJetC44_ETM30"] = 1;

// -- Jets 
 Prescales["L1_SingleJet16"] = 0;
 Prescales["L1_SingleJet36"] = 6000 *2;
 Prescales["L1_SingleJet52"] = 500 *2;
 Prescales["L1_SingleJet68"] = 100 *2;
 Prescales["L1_SingleJet92"] = 20 *2;
 Prescales["L1_SingleJet128"] =1;

 Prescales["L1_DoubleJetC36"] = 320 *2;

 // Prescales["L1_DoubleJet44_Central"] = 0;
 // Prescales["L1_DoubleJetC52"] = 40;
 // Prescales["L1_DoubleJetC56"] = 1;

 // Prescales["L1_DoubleJet44_Central"] = 0;
 Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;

 Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 8;
 Prescales["L1_DoubleJetC52"] = 0;
 Prescales["L1_DoubleJetC56"] = INFTY;

 Prescales["L1_DoubleJetC64"] = 1;

 Prescales["L1_TripleJet_64_44_24_VBF"] = 0;
 Prescales["L1_TripleJet_64_48_28_VBF"] = INFTY;
 Prescales["L1_TripleJet_68_48_32_VBF"] = 1;
 // Prescales["L1_TripleJet28_Central"] = 1000;         
 Prescales["L1_TripleJetC_52_28_28"] = 100 *2;

 Prescales["L1_QuadJetC32"] = INFTY;             // better estimate on 178208
 Prescales["L1_QuadJetC36"] = INFTY;
 Prescales["L1_QuadJetC40"] = 1;

 Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
  Prescales["L1_ETM30"] = 300;
  Prescales["L1_ETM36"] = INFTY;
  Prescales["L1_ETM40"] = 1;
  Prescales["L1_ETM50"] = 1;
  Prescales["L1_ETM70"] = 1;
  Prescales["L1_ETM100"] = 1;

  Prescales["L1_HTT50"] = 0;
  // Prescales["L1_HTT100"] = 40;
  // Prescales["L1_HTT125"] = 20;
  Prescales["L1_HTT100"] = INFTY;
  Prescales["L1_HTT125"] = INFTY;
  Prescales["L1_HTT150"] = INFTY;
  Prescales["L1_HTT175"] = INFTY;
  // Prescales["L1_HTT175"] = 1;
  Prescales["L1_HTT200"] = 1;

  Prescales["L1_ETT300"] = 1;
 
 
// -- Egamma
 Prescales["L1_SingleEG5"] = 4500;
 Prescales["L1_SingleEG7"] = 800;
 Prescales["L1_SingleEG12"] = 300;
 Prescales["L1_SingleEG18er"] = 80; 
 Prescales["L1_SingleIsoEG18er"] = 10;
 Prescales["L1_SingleEG20"] = INFTY;
 Prescales["L1_SingleIsoEG20_Eta2p1"] = 0;
 Prescales["L1_SingleEG22"] = 1;
 Prescales["L1_SingleEG24"] = 1;
 Prescales["L1_SingleEG30"] = 1;
 
 // Prescales["L1_DoubleEG_15_5"] = 8;
 Prescales["L1_DoubleEG_13_7"] = 1;

 Prescales["L1_TripleEG7"] = 1;
 Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

 Prescales["L1_SingleMu12"] = 0;
 
 Prescales["L1_SingleMu20"] = 150;
 Prescales["L1_SingleMu16"] = 150;
 Prescales["L1_SingleMu12"] = 300;
 Prescales["L1_SingleMu7"] = 600;
 Prescales["L1_SingleMu3"] = 4000;
 Prescales["L1_SingleMuOpen"] = 7000;
 Prescales["L1_DoubleMu0"] = 300;
 
 Prescales["L1_SingleMu25er"] = 1;
 Prescales["L1_SingleMu20er"] = 1;
 Prescales["L1_SingleMu14er"] = 75;
 // Prescales["L1_SingleMu18er"] = 1;
 Prescales["L1_SingleMu18er"] = INFTY;
 Prescales["L1_SingleMu16er"] = INFTY;
 // Prescales["L1_SingleMu16er"] = 1;
 Prescales["L1_DoubleMu_12_5"] = 1;
 Prescales["L1_DoubleMu5"] = 50; 
 // Prescales["L1_DoubleMu0_HighQ_Eta1p7"] = 1;
 // Prescales["L1_DoubleMu3p5_HighQ_Eta2p1"] = 1;
 Prescales["L1_DoubleMu0er_HighQ"] = 50;
 Prescales["L1_DoubleMuHighQ_3_3_Eta2p1_WdEta22"] = 1;
 Prescales["L1_DoubleMuHighQ_5_0_Eta2p1_WdEta22"] = 1;
 Prescales["L1_DoubleMu_10_Open"] = INFTY ;

 Prescales["L1_DoubleMu_10_3p5"] = 1;
 // Prescales["L1_DoubleMu_10_3p5"] = INFTY;
 Prescales["L1_TripleMu0"] = INFTY;
 Prescales["L1_TripleMu0_HighQ"] = 1;
 Prescales["L1_TripleMu_5_5_0"] = 1;

}


// -- LUMI50

if (LUMI == 50) {

// -- Cross 
 Prescales["L1_Mu0_HTT50"] = INFTY;
 Prescales["L1_Mu0_HTT100"] = 1;
 Prescales["L1_Mu4_HTT125"] = 1;

 Prescales["L1_Mu12er_ETM20"] = 1;

 Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 1;
 Prescales["L1_Mu10er_JetC32"] = 1 ;
 Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

 Prescales["L1_Mu3_JetC16_WdEtaPhi2"] =80;  
 Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 1;
 Prescales["L1_Mu8_DoubleJetC20"] = 100;
 // Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;

 Prescales["L1_EG22_ForJet24"] = 1;
 Prescales["L1_EG22_ForJet32"] = 1;
 Prescales["L1_DoubleEG6_HTT100"] = 1;
 Prescales["L1_DoubleEG6_HTT125"] = 1;

 Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 1;
 // Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = 1;
 Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
 Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
 Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 1;

 Prescales["L1_EG8_DoubleJetC20"] = 650 ;  

 Prescales["L1_Mu12_EG7"] = 1;

 Prescales["L1_DoubleMu3p5_EG5"] = 1;
 Prescales["L1_DoubleMu5_EG5"] = 1;

 // Prescales["L1_Mu3p5_DoubleEG5"] = 0;
 Prescales["L1_Mu5_DoubleEG5"] = 1 ;
 Prescales["L1_Mu5_DoubleEG6"] = 1;

 Prescales["L1_MuOpen_EG12"] = 1;
 Prescales["L1_Mu3p5_EG12"] = 1;

 Prescales["L1_DoubleJetC36_ETM30"] = 1;
 Prescales["L1_DoubleJetC44_ETM30"] = 1;


// -- Jets 
 Prescales["L1_SingleJet16"] = 0;
 Prescales["L1_SingleJet36"] = 2400;
 Prescales["L1_SingleJet52"] = 200;
 Prescales["L1_SingleJet68"] = 50;
 Prescales["L1_SingleJet92"] = 10 ;
 Prescales["L1_SingleJet128"] =1;

 Prescales["L1_DoubleJetC36"] = 160;
 Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 3;
 Prescales["L1_DoubleJetC52"] = INFTY;
 Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
 Prescales["L1_DoubleJetC56"] = 1;
 Prescales["L1_DoubleJetC64"] = 1;

 Prescales["L1_TripleJet_64_44_24_VBF"] = 1;
 Prescales["L1_TripleJet_64_48_28_VBF"] = 1;
 Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

 // Prescales["L1_TripleJet28_Central"] = 500; 
 Prescales["L1_TripleJetC_52_28_28"] = 40;

 Prescales["L1_QuadJetC32"] = INFTY;            
 Prescales["L1_QuadJetC36"] = 1;
 Prescales["L1_QuadJetC40"] = 1;

 Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
  Prescales["L1_ETM30"] = 100;
  Prescales["L1_ETM36"] = 1;
  Prescales["L1_ETM40"] = 1;
  Prescales["L1_ETM50"] = 1;
  Prescales["L1_ETM70"] = 1;
  Prescales["L1_ETM100"] = 1;

  Prescales["L1_HTT100"] = INFTY;
  Prescales["L1_HTT125"] = INFTY;
  Prescales["L1_HTT150"] = 1;
  Prescales["L1_HTT175"] = 1;
  Prescales["L1_HTT200"] = 1;

  Prescales["L1_ETT300"] = 1;


// -- Egamma
 Prescales["L1_SingleEG5"] = 3000;
 Prescales["L1_SingleEG7"] = 400;
 Prescales["L1_SingleEG12"] = 200;
 Prescales["L1_SingleEG18er"] = 10;
 // Prescales["L1_SingleIsoEG18er"] = 10;
 Prescales["L1_SingleIsoEG18er"] = 1;

 Prescales["L1_SingleEG20"] = 1;
 Prescales["L1_SingleIsoEG20_Eta2p1"] = 0;
 Prescales["L1_SingleEG22"] = 1;
 Prescales["L1_SingleEG24"] = 1;
 Prescales["L1_SingleEG30"] = 1;

 // Prescales["L1_DoubleEG_15_5"] = 1;
 Prescales["L1_DoubleEG_13_7"] = 1;

 Prescales["L1_TripleEG7"] = 1;
 Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

 Prescales["L1_SingleMu12"] = 0;
  
 Prescales["L1_SingleMu20"] = 100;
 Prescales["L1_SingleMu16"] = 100;
 Prescales["L1_SingleMu12"] = 200;
 Prescales["L1_SingleMu7"] = 400;
 Prescales["L1_SingleMu3"] = 2000;
 Prescales["L1_SingleMuOpen"] = 3500;
 Prescales["L1_DoubleMu0"] = 150;

 Prescales["L1_SingleMu25er"] = 1;
 Prescales["L1_SingleMu20er"] = 1;
 // Prescales["L1_SingleMu14er"] = 50;
 Prescales["L1_SingleMu14er"] =  1;
 Prescales["L1_SingleMu18er"] = 1;
 Prescales["L1_SingleMu16er"] = 1;
 Prescales["L1_DoubleMu_12_5"] = 1;
 Prescales["L1_DoubleMu5"] = 50;
 Prescales["L1_DoubleMu0er_HighQ"] = 1;
 Prescales["L1_DoubleMuHighQ_3_3_Eta2p1_WdEta22"] = 1;
 Prescales["L1_DoubleMuHighQ_5_0_Eta2p1_WdEta22"] = 1;
 Prescales["L1_DoubleMu_10_Open"] = 1 ;
 Prescales["L1_DoubleMu_10_3p5"] = 1;
 Prescales["L1_TripleMu0"] = 1;
 Prescales["L1_TripleMu0_HighQ"] = 1;
 Prescales["L1_TripleMu_5_5_0"] = 1;



}



// test : unprescale all seeds
/*
 for (map<string, int>::iterator it=Prescales.begin(); it != Prescales.end(); it++) {
        string name = it -> first;
        Prescales[name] = 1;
 }
*/


/*
// -- test  to see where we stand if we would get rid of all p'ed seeds
// -- (in 2011 we spent ~ 20% of the rate in monitoring / control p'ed seeds..)

 for (map<string, int>::iterator it=Prescales.begin(); it != Prescales.end(); it++) {
        string name = it -> first;
	int p = it -> second;
	if (p > 1 ) Prescales[name] = 0;
 }
*/


// -- Each seed gets a "weight" according to how many PAGS are using it

 for (map<string, int>::iterator it=Prescales.begin(); it != Prescales.end(); it++) {
        string name = it -> first;
	int UsedPernPAG = 0;
	if ( setTOP.count(name) > 0) UsedPernPAG ++;
        if ( setHIGGS.count(name) > 0) UsedPernPAG ++;
        if ( setSUSY.count(name) > 0) UsedPernPAG ++;
        if ( setEXO.count(name) > 0) UsedPernPAG ++;
        if ( setSMP.count(name) > 0) UsedPernPAG ++;
        if ( setBPH.count(name) > 0) UsedPernPAG ++;
	WeightsPAGs[name] = 1./(float)UsedPernPAG;
 }


 for (map<string, int>::iterator it=Prescales.begin(); it != Prescales.end(); it++) {
        string name = it -> first;
 	Counts[name] = 0;
        Biased[name] = false; 
 }


// -- The "Biased" table is only used for the final print-out
// -- set true for seeds for which the rate estimation is biased by
// -- the sample (because of the seeds enabled in the high PU run)
 
 // Biased["L1_TripleMu0"] = true;
 // Biased["L1_DoubleMu_10_Open"] = true;
 // Biased["L1_SingleEG5"] = true;
 // Biased["L1_TripleEG7"] = true;
 // Biased["L1_TripleEG_10_7_5"] = true;
 // Biased["L1_SingleJet36"] = true;
 // Biased["L1_DoubleJetC36"] = true;
 // Biased["L1_DoubleJetC44_Eta1p74_WdEta4"] = true;
 // Biased["L1_QuadJetC36"] = true;
 // Biased["L1_QuadJetC40"] = true;
 // Biased["L1_DoubleTauJet44er"] = true;
 // Biased["L1_Mu3p5_DoubleEG5"] = true;
 // Biased["L1_QuadJetC32"] = true;
 // Biased["L1_TripleJet28_Central"] = true;


 
}

// ------------------------------------------------------------------



        
// ------------------------------------------------------------------

// --------------------------------------------------------------------
//                             run function 
// --------------------------------------------------------------------

bool L1Menu2012::dummy(string L1name) {

	return false;
}


// --------------------------------------------------------------------

bool L1Menu2012::SingleMuEta2p1(float ptcut) {

        bool muon = false;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) { 
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > 2.1) continue;
          if (pt >= ptcut) muon = true;
        }

        bool ok = muon;
	return ok;

}

bool L1Menu2012::SingleMu(float ptcut, int qualmin) {

        bool muon = false;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < qualmin) continue;
          if (pt >= ptcut) muon = true;
        }

        bool ok = muon;
	return ok;

}

bool L1Menu2012::DoubleMuHighQEtaCut(float ptcut, float etacut) {

	int nmu=0;
        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > etacut) continue;
          if (pt >= ptcut) nmu ++;
        }

        bool ok = (nmu >= 2 ) ;
	return ok;

}


bool L1Menu2012::Onia(float ptcut1, float ptcut2, float etacut, int delta) {
          
        int Nmu = gmt_ -> N;
	int n1=0;
	int n2=0;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > etacut) continue;
          if (pt >= ptcut1) n1 ++;
	  if (pt >= ptcut2) n2++;
        }

        bool ok = (n1 >=1 && n2 >= 2 ) ;
        if (! ok) return false;

	// -- now the CORRELATION condition
	bool CORREL = false;

	// cout << " Onia, correl " << endl;

	for (int imu=0; imu < Nmu; imu++) {
	  int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > etacut) continue;
	  if (pt < ptcut1) continue;
	  int ieta1 = etaMuIdx(eta);

		for (int imu2=0; imu2 < Nmu; imu2++) {
		if (imu2 == imu) continue;
          	int bx2 = gmt_ -> CandBx[imu2];       //    BX = 0, +/- 1 or +/- 2
          	if (bx2 != 0) continue;
          	float pt2 = gmt_ -> Pt[imu2];       // the Pt  of the muon in GeV
          	int qual2 = gmt_ -> Qual[imu2];        // the quality of the muon
          	if ( qual2 < 4) continue;
          	float eta2 = gmt_  -> Eta[imu2];        // physical eta
          	if (fabs(eta2) > etacut) continue;
          	if (pt2 < ptcut2) continue;
		int ieta2 = etaMuIdx(eta2);

		float deta = ieta1 - ieta2; 
		// cout << "eta 1 2 delta " << ieta1 << " " << ieta2 << " " << deta << endl;
		if ( fabs(deta) <= delta)  CORREL = true;
 		// if (fabs ( eta - eta2) <=  1.7) CORREL = true; 
		}

	}

	return CORREL;

}



bool L1Menu2012::DoubleMu(float cut1, float cut2) {

        int n1=0;
	int n2=0;
        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          // if ( qual < 4) continue;
	  if (qual < 4  && qual != 3 ) continue;
          if (pt >= cut1) n1 ++;
          if (pt >= cut2) n2 ++;
        }

        bool ok = (n1 >= 1 && n2 >= 2 );
	return ok;
        
}

bool L1Menu2012::TripleMu(float cut1, float cut2, float cut3, int qualmin) {

        int n1=0;
        int n2=0;
	int n3=0;
        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < qualmin) continue;
          if (pt >= cut1) n1 ++;
          if (pt >= cut2) n2 ++;
          if (pt >= cut3) n3 ++;
        }

        bool ok = ( n1 >= 1 && n2 >= 2 && n3 >= 3 );
	return ok;

}


bool L1Menu2012::DoubleMuXOpen(float cut) {

        int n1=0;
        int n2=0;
        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
	  if ( (qual >= 5 || qual == 3 ) && pt >= cut ) n1 ++;
	  if ( pt >= 0 ) n2 ++;
        }

        bool ok = ( n1 >= 1 && n2 >= 2 );
	return ok;

}


bool L1Menu2012::Muons() {

 insert_ibin = 0;
 InsertInMenu("L1_SingleMuOpen",SingleMu(0.,0));
 InsertInMenu("L1_SingleMu3",SingleMu(3.));
 InsertInMenu("L1_SingleMu7",SingleMu(7.));
 InsertInMenu("L1_SingleMu12",SingleMu(12.));
 InsertInMenu("L1_SingleMu16",SingleMu(16.));
 InsertInMenu("L1_SingleMu20",SingleMu(20.));

 InsertInMenu("L1_SingleMu14er",SingleMuEta2p1(14.));
 InsertInMenu("L1_SingleMu16er",SingleMuEta2p1(16.));
 InsertInMenu("L1_SingleMu18er",SingleMuEta2p1(18.));
 InsertInMenu("L1_SingleMu20er",SingleMuEta2p1(20.));
 InsertInMenu("L1_SingleMu25er",SingleMuEta2p1(25.));
        
 InsertInMenu("L1_DoubleMu0",DoubleMu(0.,0.));
 InsertInMenu("L1_DoubleMu0er_HighQ",DoubleMuHighQEtaCut(0.,2.1));
 InsertInMenu("L1_DoubleMuHighQ_3_3_Eta2p1_WdEta22",Onia(3.,3.,2.1,22));
 InsertInMenu("L1_DoubleMuHighQ_5_0_Eta2p1_WdEta22",Onia(5.,0.,2.1,22));

 InsertInMenu("L1_DoubleMu5",DoubleMu(5.,5.));
 InsertInMenu("L1_DoubleMu_12_5",DoubleMu(12.,5.));
 InsertInMenu("L1_DoubleMu_10_Open",DoubleMuXOpen(10.));
 InsertInMenu("L1_DoubleMu_10_3p5",DoubleMu(10.,3.5));

 InsertInMenu("L1_TripleMu0",TripleMu(0.,0.,0.,3));
 InsertInMenu("L1_TripleMu0_HighQ",TripleMu(0.,0.,0.,4));
 InsertInMenu("L1_TripleMu_5_5_0",TripleMu(5.,5.,0.,3));


 int NN = insert_ibin;
// cout << " NN = " << NN << endl;

 int kOFFSET_old = kOFFSET;
 for (int k=0; k < NN; k++) {
        TheTriggerBits[k + kOFFSET_old] = insert_val[k];
 }
 kOFFSET += insert_ibin;

        
 if (first) {

	NBITS_MUONS = NN;

	for (int ibin=0; ibin < insert_ibin; ibin++) {
	  TString l1name = (TString)insert_names[ibin];
	  h_Muons -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
	}
	h_Muons -> GetXaxis() -> SetBinLabel(NN+1, "MUONS") ;

        for (int k=1; k <= kOFFSET - kOFFSET_old ; k++) {
           h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Muons -> GetXaxis() -> GetBinLabel(k) );
        }


 }

 bool res = false;
 for (int i=0; i < NN; i++) {
  res = res || insert_val[i] ;
  if (insert_val[i]) h_Muons -> Fill(i);
 }
 if (res) h_Muons -> Fill(NN);

 return res;
}


// ----------------------------------------------------------------------------



bool L1Menu2012::Mu_EG(float mucut, float EGcut ) {

        bool eg =false;
        bool muon = false;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {   
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) muon = true;
        }
        
                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        bool ok = muon && eg;
	return ok;
        
}

bool L1Menu2012::DoubleMu_EG(float mucut, float EGcut ) {

        bool eg =false;
        bool muon = false;
        int  Nmuons = 0;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          // if ( qual < 4) continue;
          if (qual < 4 && qual !=3 ) continue;
          if (pt >= mucut) Nmuons ++;
        }
	if (Nmuons >= 2) muon = true;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        bool ok = muon && eg;
	return ok;

}

bool L1Menu2012::Mu_DoubleEG(float mucut, float EGcut ) {

        bool eg =false;
        bool muon = false;
        int  Nmuons = 0;
	int Nelectrons = 0;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) Nmuons ++;
        }
        if (Nmuons >= 1) muon = true;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) Nelectrons ++;
                }  // end loop over EM objects
		if (Nelectrons >= 2) eg = true;

        bool ok = muon && eg;
	return ok;

}


// --------------------------------------------------------------------

bool L1Menu2012::MuOpen_EG(float mucut, float EGcut ) {

        bool eg =false;
        bool muon = false;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          if (pt >= mucut) muon = true;
        }

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        bool ok = muon && eg;
	return ok;

}



// --------------------------------------------------------------------

bool L1Menu2012::Mu_JetCentral(float mucut, float jetcut ) {

	bool jet=false;
	bool muon = false;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
		if (pt >= jetcut) jet = true;
	}

  	int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) muon = true;
        }

	bool ok = muon && jet;
	return ok;


}

// --------------------------------------------------------------------

bool L1Menu2012::Mu_DoubleJetCentral(float mucut, float jetcut ) {

        bool jet=false;
        bool muon = false;

	int n1 = 0;
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) n1 ++;
        }
	jet = ( n1 >= 2 );

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) muon = true;
        }

        bool ok = muon && jet;
	return ok;

}
// --------------------------------------------------------------------

bool L1Menu2012::Mu_JetCentral_LowerTauTh(float mucut, float jetcut, float taucut ) {

        bool jet=false;
	bool central = false;
	bool tau = false;
        bool muon = false;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
		bool isTauJet = gt_ -> Taujet[ue];
		if (! isTauJet) {  	// look at CentralJet
                	if (pt >= jetcut) central = true;
		}
		else   {		// look at TauJets
			if (pt >= taucut) tau = true;
		}
        }
	jet = central || tau  ;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) muon = true;
        }

        bool ok = muon && jet;
	return ok;

}


// --------------------------------------------------------------------



bool L1Menu2012::Muer_JetCentral(float mucut, float jetcut ) {

        bool jet=false;
        bool muon = false;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) jet = true;
        }

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > 2.1) continue;

          if (pt >= mucut) muon = true;
        }

        bool ok = muon && jet;
	return ok;

}


// --------------------------------------------------------------------


bool L1Menu2012::Muer_JetCentral_LowerTauTh(float mucut, float jetcut, float taucut) {

        bool jet=false;
        bool central = false;
        bool tau = false;
        bool muon = false;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                bool isTauJet = gt_ -> Taujet[ue];
                if (! isTauJet) {       // look at CentralJet
                        if (pt >= jetcut) central = true;
                }
                else   {                // look at TauJets
                        if (pt >= taucut) tau = true;
                }
        }
        jet = central || tau  ;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > 2.1) continue;
          if (pt >= mucut) muon = true;
        }

        bool ok = muon && jet;
	return ok;

}

// --------------------------------------------------------------------

bool L1Menu2012::Mia(float mucut, float jet1, float jet2) {

        bool jet=false;
        bool muon = false;
	int n1 = 0;
	int n2 = 0;
        
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
		if (pt >= jet1) n1 ++;
		if (pt >= jet2) n2 ++;
        }       
        jet = (n1 >= 1 && n2 >= 2 );
        
        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > 2.1) continue;        
          if (pt >= mucut) muon = true;
        } 
        
        bool ok = muon && jet;
        if (! ok) return false;

	// now the CORREL condition


        bool CORREL = false;

        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt < mucut) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > 2.1) continue;

          float phimu = gmt_ -> Phi[imu];
          int iphi_mu = phiINjetCoord(phimu);
	  float etamu = gmt_ -> Eta[imu];
	  int ieta_mu = etaINjetCoord(etamu);

          for (int ue=0; ue < Nj; ue++) {
                  int bxj = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                  if (bxj != 0) continue;
                  bool isFwdJet = gt_ -> Fwdjet[ue];
                  if (isFwdJet) continue;
                  float rank = gt_ -> Rankjet[ue];
                  float ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                  if (ptj < jet2) continue;
                  float phijet = gt_ -> Phijet[ue];
                  int iphi_jet = (int)phijet;
		  float etajet = gt_ -> Etajet[ue];
		  int ieta_jet = (int)etajet;

                  bool corr_phi = correlateInPhi(iphi_jet, iphi_mu);
		  bool corr_eta = correlateInEta(ieta_jet, ieta_mu);
		  bool corr = corr_phi && corr_eta;
                  if (corr) CORREL = true ;
          }
        }

	return CORREL;

}

// --------------------------------------------------------------------

bool L1Menu2012::Mu_JetCentral_delta(float mucut, float jetcut, int IBIT) {

        bool jet=false;
        bool muon = false;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) jet = true;
        }

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) muon = true;
        }

        bool ok = muon && jet;
        if (! ok) return false;

        //  -- now evaluate the delta condition :

        bool CORREL = false;

        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt < mucut) continue;

          float phimu = gmt_ -> Phi[imu];
          int iphi_mu = phiINjetCoord(phimu);
          float etamu = gmt_ -> Eta[imu];
          int ieta_mu = etaINjetCoord(etamu);

          for (int ue=0; ue < Nj; ue++) {
                  int bxj = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                  if (bxj != 0) continue;
                  bool isFwdJet = gt_ -> Fwdjet[ue];
                  if (isFwdJet) continue;
                  float rank = gt_ -> Rankjet[ue];
                  float ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                  if (ptj < jetcut) continue;
                  float phijet = gt_ -> Phijet[ue];
                  int iphi_jet = (int)phijet;
                  float etajet = gt_ -> Etajet[ue];
                  int ieta_jet = (int)etajet;

                  bool corr = correlateInPhi(iphi_jet, iphi_mu, 2) && correlateInEta(ieta_jet, ieta_mu, 2);
                  if (corr) CORREL = true ;
          }
        }

	return CORREL;

}

// --------------------------------------------------------------------

bool L1Menu2012::Mu_JetCentral_deltaOut(float mucut, float jetcut, int IBIT) {

        bool jet=false;
        bool muon = false;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
		float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) jet = true;
        }

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) muon = true;
        }

        bool ok = muon && jet;
        if (! ok) return false;

        //  -- now evaluate the delta condition :

        bool CORREL = false;

        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt < mucut) continue;

          float phimu = gmt_ -> Phi[imu];
          int iphi_mu = phiINjetCoord(phimu);
          float etamu = gmt_ -> Eta[imu];
          int ieta_mu = etaINjetCoord(etamu);

	  int PhiOut[3];
                        PhiOut[0] = iphi_mu;
                        if (iphi_mu< 17) PhiOut[1] = iphi_mu+1;
                        if (iphi_mu == 17) PhiOut[1] = 0;
                        if (iphi_mu > 0) PhiOut[2] = iphi_mu - 1;
                        if (iphi_mu == 0) PhiOut[2] = 17;

          for (int ue=0; ue < Nj; ue++) {
                  int bxj = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                  if (bxj != 0) continue;
                  bool isFwdJet = gt_ -> Fwdjet[ue];
                  if (isFwdJet) continue;
                  float rank = gt_ -> Rankjet[ue];
		  float ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                  if (ptj < jetcut) continue;
                  float phijet = gt_ -> Phijet[ue];
                  int iphi_jet = (int)phijet;
                  float etajet = gt_ -> Etajet[ue];
                  int ieta_jet = (int)etajet;

		   if (! correlateInPhi(iphi_jet, iphi_mu, 8)) CORREL = true;


          }
        }

	return CORREL;

}


// --------------------------------------------------------------------

bool L1Menu2012::Muer_TripleJetCentral(float mucut, float jet1, float jet2, float jet3)  {

        bool jet=false;
        bool muon = false;

        int n1=0;
        int n2=0;
        int n3=0;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jet1) n1 ++;
                if (pt >= jet2) n2 ++;
                if (pt >= jet3) n3 ++;
        }

        jet = ( n1 >= 1 && n2 >= 2 && n3 >= 3 ) ;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_ -> Eta[imu] ;
          if (fabs(eta) > 2.1 ) continue;
          if (pt >= mucut) muon = true;
        }

        bool ok = muon && jet;
	return ok;

}




bool L1Menu2012::Mu_HTT(float mucut, float HTcut ) {

        bool ht=false;
        bool muon = false;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          if (pt >= mucut) muon = true;
        }

        float adc = gt_ -> RankHTT ;
        float TheHTT = adc / 2. ;
	ht = (TheHTT >= HTcut) ;

        bool ok = muon && ht;
	return ok;

}

// --------------------------------------------------------------------

bool L1Menu2012::Muer_ETM(float mucut, float ETMcut ) {

        bool etm = false;
        bool muon = false;

        int Nmu = gmt_ -> N;
        for (int imu=0; imu < Nmu; imu++) {
          int bx = gmt_ -> CandBx[imu];       //    BX = 0, +/- 1 or +/- 2
          if (bx != 0) continue;
          float pt = gmt_ -> Pt[imu];       // the Pt  of the muon in GeV
          int qual = gmt_ -> Qual[imu];        // the quality of the muon
          if ( qual < 4) continue;
          float eta = gmt_  -> Eta[imu];        // physical eta
          if (fabs(eta) > 2.1) continue;

          if (pt >= mucut) muon = true;
        }

        float adc = gt_ -> RankETM ;
        float TheETM = adc / 2. ;
	etm = (TheETM >= ETMcut);

        bool ok = muon && etm;
	return ok;

}


// --------------------------------------------------------------------

bool L1Menu2012::EG_FwdJet(float EGcut, float FWcut ) {

	bool eg = false;
	bool jet = false;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {        
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (!isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= FWcut) jet = true;
        }

	bool ok = ( eg && jet);
	return ok;

}

// --------------------------------------------------------------------

bool L1Menu2012::EG_DoubleJetCentral(float EGcut, float jetcut ) {

        bool eg = false;
        bool jet = false;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        int Nj = gt_ -> Njet ;
	int njets = 0;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) njets ++;
        }
	jet = ( njets >= 2 );

        bool ok = ( eg && jet);
	return ok;

}


// --------------------------------------------------------------------

bool L1Menu2012::EG_HT(float EGcut, float HTcut ) {

        bool eg = false;
        bool ht = false;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        float adc = gt_ -> RankHTT ;
        float TheHTT = adc / 2. ;
        ht = (TheHTT >= HTcut) ;

        bool ok = ( eg && ht);
	return ok;

}

// --------------------------------------------------------------------

bool L1Menu2012::DoubleEG_HT(float EGcut, float HTcut ) {

        bool eg = false;
	int n1 = 0;
        bool ht = false;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) n1 ++;
                }  // end loop over EM objects
	eg = ( n1 >= 2 );
        
        float adc = gt_ -> RankHTT ;
        float TheHTT = adc / 2. ;
        ht = (TheHTT >= HTcut) ;
        
        bool ok = ( eg && ht);
	return ok;
        
}

// --------------------------------------------------------------------

bool L1Menu2012::EGEta2p1_JetCentral(float EGcut, float jetcut) {

        bool eg = false;
        bool jet = false;
                        
                int Nele = gt_ -> Nele; 
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                	float eta = gt_ -> Etael[ue];
                	if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) jet = true;
        }

	bool ok = (eg && jet);
        if (! ok) return false;


	//  -- now evaluate the delta condition :

	bool CORREL = false;

                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt < EGcut) continue;

			float phiel = gt_ -> Phiel[ue];
			int iphiel = (int)phiel;

			int PhiOut[3];
			PhiOut[0] = iphiel;
			if (iphiel< 17) PhiOut[1] = iphiel+1;
			if (iphiel == 17) PhiOut[1] = 0;
			if (iphiel > 0) PhiOut[2] = iphiel - 1;
			if (iphiel == 0) PhiOut[2] = 17;

        			for (int uj=0; uj < Nj; uj++) {
                			int bxj = gt_ -> Bxjet[uj];               //    BX = 0, +/- 1 or +/- 2
                			if (bxj != 0) continue;
                			bool isFwdJet = gt_ -> Fwdjet[uj];
                			if (isFwdJet) continue;
                			float rankj = gt_ -> Rankjet[uj];
                			// float ptj = rankj * 4;
					float ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.);
                			if (ptj < jetcut) continue;
					float phijet = gt_ -> Phijet[uj];
					int iphijet = (int)phijet; 

					if ( iphijet != PhiOut[0] && 
					     iphijet != PhiOut[1] &&
					     iphijet != PhiOut[2] ) CORREL = true;
        			}  // loop over jets

                }  // end loop over EM objects

	return CORREL;

}



// --------------------------------------------------------------------

bool L1Menu2012::EGEta2p1_JetCentral_LowTauTh(float EGcut, float jetcut, float taucut) {

        bool eg = false;
        bool jet = false;
	bool central = false;
	bool tau = false;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
		bool isTauJet = gt_ -> Taujet[ue];
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (! isTauJet) {
			if (pt >= jetcut) central = true;
		}
		else {
			if (pt >= taucut) tau = true;
		}
        }
	jet = tau || central;

        bool ok = (eg && jet);
        if (! ok) return false;


        //  -- now evaluate the delta condition :

        bool CORREL_CENTRAL = false;
	bool CORREL_TAU = false;

                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt < EGcut) continue;

                        float phiel = gt_ -> Phiel[ue];
                        int iphiel = (int)phiel;

                        int PhiOut[3];
                        PhiOut[0] = iphiel;
                        if (iphiel< 17) PhiOut[1] = iphiel+1;
                        if (iphiel == 17) PhiOut[1] = 0;
                        if (iphiel > 0) PhiOut[2] = iphiel - 1;
                        if (iphiel == 0) PhiOut[2] = 17;

                                for (int uj=0; uj < Nj; uj++) {
                                        int bxj = gt_ -> Bxjet[uj];               //    BX = 0, +/- 1 or +/- 2
                                        if (bxj != 0) continue;
                                        bool isFwdJet = gt_ -> Fwdjet[uj];
                                        if (isFwdJet) continue;
					bool isTauJet = gt_ -> Taujet[uj];
                                        float rankj = gt_ -> Rankjet[uj];
                                        float ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.);
                                        float phijet = gt_ -> Phijet[uj];
                                        int iphijet = (int)phijet;

					if (! isTauJet) {

					  if (ptj >= jetcut) { 
                                        	if ( iphijet != PhiOut[0] &&
                                             	iphijet != PhiOut[1] &&
                                             	iphijet != PhiOut[2] ) CORREL_CENTRAL = true;
					  }

					}
					else {
                                          if (ptj >= taucut) {
                                                if ( iphijet != PhiOut[0] &&
                                                iphijet != PhiOut[1] &&
                                                iphijet != PhiOut[2] ) CORREL_TAU = true;
                                          }

					}


                                }  // loop over jets

                }  // end loop over EM objects

	bool CORREL = CORREL_CENTRAL || CORREL_TAU ;
	return CORREL;

}


// --------------------------------------------------------------------

bool L1Menu2012::IsoEGEta2p1_JetCentral_LowTauTh(float EGcut, float jetcut, float taucut) {

        bool eg = false;
        bool jet = false;
        bool central = false;
        bool tau = false;

                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        bool iso = gt_ -> Isoel[ue];
                        if ( ! iso) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;
                }  // end loop over EM objects

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                bool isTauJet = gt_ -> Taujet[ue];
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (! isTauJet) {
                        if (pt >= jetcut) central = true;
                }
                else {
                        if (pt >= taucut) tau = true;
                }
        }
        jet = tau || central;

        bool ok = (eg && jet);
        if (! ok) return false;

        //  -- now evaluate the delta condition :

        bool CORREL_CENTRAL = false;
        bool CORREL_TAU = false;

                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        bool iso = gt_ -> Isoel[ue];
                        if ( ! iso) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt < EGcut) continue;

                        float phiel = gt_ -> Phiel[ue];
                        int iphiel = (int)phiel;
                        
                        int PhiOut[3];
                        PhiOut[0] = iphiel; 
                        if (iphiel< 17) PhiOut[1] = iphiel+1;
                        if (iphiel == 17) PhiOut[1] = 0;
                        if (iphiel > 0) PhiOut[2] = iphiel - 1;
                        if (iphiel == 0) PhiOut[2] = 17;
                
                                for (int uj=0; uj < Nj; uj++) {
                                        int bxj = gt_ -> Bxjet[uj];               //    BX = 0, +/- 1 or +/- 2
                                        if (bxj != 0) continue;
                                        bool isFwdJet = gt_ -> Fwdjet[uj];
                                        if (isFwdJet) continue;
                                        bool isTauJet = gt_ -> Taujet[uj];
                                        float rankj = gt_ -> Rankjet[uj];
                			float ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.);
                                        float phijet = gt_ -> Phijet[uj];
                                        int iphijet = (int)phijet;
                        
                                        if (! isTauJet) {
        
                                          if (ptj >= jetcut) {
                                                if ( iphijet != PhiOut[0] &&
                                                iphijet != PhiOut[1] &&
                                                iphijet != PhiOut[2] ) CORREL_CENTRAL = true;
                                          }
        
                                        }
                                        else {
                                          if (ptj >= taucut) {
                                                if ( iphijet != PhiOut[0] &&
                                                iphijet != PhiOut[1] &&
                                                iphijet != PhiOut[2] ) CORREL_TAU = true;
                                          }
                        
                                        }
                        
                        
                                }  // loop over jets
                        
                }  // end loop over EM objects

        bool CORREL = CORREL_CENTRAL || CORREL_TAU ;

	return CORREL;

}




// --------------------------------------------------------------------

bool L1Menu2012::EGEta2p1_DoubleJetCentral(float EGcut, float jetcut) {

        bool eg = false;
        bool jet = false;
        int n2=0;
                                
                int Nele = gt_ -> Nele; 
                for (int ue=0; ue < Nele; ue++) { 
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true; 
                }  // end loop over EM objects
                                        
        int Nj = gt_ -> Njet ;               
        for (int ue=0; ue < Nj; ue++) {      
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue; 
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) n2 ++;
        }

        jet = (n2 >= 2);

        bool ok = (eg && jet);
        if (! ok) return false;

        //  -- now evaluate the delta condition :

        bool CORREL = false;

                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt < EGcut) continue;
                        
                        float phiel = gt_ -> Phiel[ue];
                        int iphiel = (int)phiel;
                        
                        int PhiOut[3];
                        PhiOut[0] = iphiel;
                        if (iphiel< 17) PhiOut[1] = iphiel+1;
                        if (iphiel == 17) PhiOut[1] = 0;
                        if (iphiel > 0) PhiOut[2] = iphiel - 1;
                        if (iphiel == 0) PhiOut[2] = 17;
                        
                        int npair = 0;
                                
                                for (int uj=0; uj < Nj; uj++) {
                                        int bxj = gt_ -> Bxjet[uj];               //    BX = 0, +/- 1 or +/- 2
                                        if (bxj != 0) continue; 
                                        bool isFwdJet = gt_ -> Fwdjet[uj];
                                        if (isFwdJet) continue;
                                        float rankj = gt_ -> Rankjet[uj];
                                        // float ptj = rankj * 4;
					float ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.);
                                        if (ptj < jetcut) continue;
                                        float phijet = gt_ -> Phijet[uj];
                                        int iphijet = (int)phijet;
                                        
                                        if ( iphijet != PhiOut[0] &&
                                             iphijet != PhiOut[1] &&
                                             iphijet != PhiOut[2] ) npair ++;
                                
                                }  // loop over jets
                        
                        if (npair >= 2 ) CORREL = true ;
                
                }  // end loop over EM objects
        
	return CORREL;

}

// --------------------------------------------------------------------
                        
bool L1Menu2012::EGEta2p1_DoubleJetCentral_TripleJetCentral(float EGcut, float jetcut2, 
			float jetcut3) {

        bool eg = false;
        bool jet = false;
        int n2=0;       
  	int n3=0;
                                
                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue; 
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= EGcut) eg = true;  
                }  // end loop over EM objects
                                        
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut2) n2 ++;
                if (pt >= jetcut3) n3 ++;
        }

        jet = (n2 >= 2 && n3 >= 3 );

        bool ok = (eg && jet);
	return ok;


}


// --------------------------------------------------------------------

bool L1Menu2012::HTT_HTM(float HTTcut, float HTMcut) {

        bool htt = false;
        bool htm = false;
        float adc = gt_ -> RankHTT;   
        float TheHTT = (float)adc / 2.   ;           // HTT in GeV,  2011 data
	htt = ( TheHTT >= HTTcut ) ;

        int adc_HTM  = gt_  -> RankHTM ;    // HTM in adc counts
        float TheHTM = adc_HTM * 2.  ;              //  HTM in GeV
	htm = ( TheHTM >= HTMcut );

        bool ok = (htt && htm);
	return ok;
        
}

// --------------------------------------------------------------------

bool L1Menu2012::JetCentral_ETM(float jetcut, float ETMcut) {

        bool etm = false;
        bool jet = false;

        float adc = gt_ -> RankETM ;
        float TheETM = adc / 2. ;
        etm = (TheETM >= ETMcut);

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut) jet = true;
        }

 	bool ok = ( jet && etm );
	return ok;

}

// --------------------------------------------------------------------
                
bool L1Menu2012::DoubleJetCentral_ETM(float jetcut1, float jetcut2, float ETMcut) {
        
        bool etm = false; 
        bool jet = false;
	int n1=0;
	int n2=0;

        float adc = gt_ -> RankETM ;
        float TheETM = adc / 2. ;
        etm = (TheETM >= ETMcut);

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= jetcut1) n1 ++;
		if (pt >= jetcut2) n2 ++;
        }       
	jet = (n1 >= 1 && n2 >= 2);
        
        bool ok = ( jet && etm );
	return ok;

}



// --------------------------------------------------------------------

bool L1Menu2012::Cross() {

 insert_ibin = 0;

 InsertInMenu("L1_Mu0_HTT50", Mu_HTT(0.,50.) );
 InsertInMenu("L1_Mu0_HTT100", Mu_HTT(0.,100.) );
 InsertInMenu("L1_Mu4_HTT125", Mu_HTT(4.,125.) );

 InsertInMenu("L1_Mu12er_ETM20", Muer_ETM(12.,20.) );
 InsertInMenu("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12", Mia(10.,20.,12.) );
 InsertInMenu("L1_Mu10er_JetC32", Muer_JetCentral(10.,32.) );
 InsertInMenu("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12", Mia(10.,32.,12.) );


 InsertInMenu("L1_Mu3_JetC16_WdEtaPhi2", Mu_JetCentral_delta(3.,16.,0) );
 InsertInMenu("L1_Mu3_JetC52_WdEtaPhi2", Mu_JetCentral_delta(3.,52.,17) );
 InsertInMenu("L1_Mu8_DoubleJetC20", Mu_DoubleJetCentral(8.,20.) );

 InsertInMenu("L1_EG22_ForJet24", EG_FwdJet(22.,24.) );
 InsertInMenu("L1_EG22_ForJet32", EG_FwdJet(22.,32.) );
 InsertInMenu("L1_DoubleEG6_HTT100", DoubleEG_HT(6.,100.) );
 InsertInMenu("L1_DoubleEG6_HTT125", DoubleEG_HT(6.,125.) );

 InsertInMenu("L1_EG18er_JetC_Cen28_Tau20_dPhi1", EGEta2p1_JetCentral_LowTauTh(18.,28.,20.) );
 InsertInMenu("L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1", IsoEGEta2p1_JetCentral_LowTauTh(18.,32.,24.) );
 InsertInMenu("L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1", IsoEGEta2p1_JetCentral_LowTauTh(18.,36.,28.) );
 InsertInMenu("L1_EG18er_JetC_Cen36_Tau28_dPhi1", EGEta2p1_JetCentral_LowTauTh(18.,36.,28.) );

 InsertInMenu("L1_EG8_DoubleJetC20", EG_DoubleJetCentral(8.,20.) );

 InsertInMenu("L1_Mu12_EG7", Mu_EG(12.,7.) );
 InsertInMenu("L1_MuOpen_EG12", MuOpen_EG(0.,12.) );
 InsertInMenu("L1_Mu3p5_EG12", Mu_EG(3.5,12.) );

 InsertInMenu("L1_DoubleMu3p5_EG5", DoubleMu_EG(3.5,5.) );
 InsertInMenu("L1_DoubleMu5_EG5", DoubleMu_EG(5.,5.) );

 InsertInMenu("L1_Mu5_DoubleEG5", Mu_DoubleEG(5., 5.) );
 InsertInMenu("L1_Mu5_DoubleEG6", Mu_DoubleEG(5., 6.) );

 InsertInMenu("L1_DoubleJetC36_ETM30", DoubleJetCentral_ETM(36., 36., 30.) );
 InsertInMenu("L1_DoubleJetC44_ETM30", DoubleJetCentral_ETM(44., 44., 30.) );

 int NN = insert_ibin;
 int kOFFSET_old = kOFFSET;
 for (int k=0; k < NN; k++) {
	TheTriggerBits[k + kOFFSET_old] = insert_val[k];
 }
 kOFFSET += NN;

 if (first) {

	NBITS_CROSS = NN;

        for (int ibin=0; ibin < insert_ibin; ibin++) {
          TString l1name = (TString)insert_names[ibin];
          h_Cross -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
        }
     	h_Cross-> GetXaxis() -> SetBinLabel(NN+1,"CROSS");

        for (int k=1; k <= kOFFSET - kOFFSET_old; k++) {
           h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Cross -> GetXaxis() -> GetBinLabel(k) );
        }

 }

 bool res = false;
 for (int i=0; i < NN; i++) {
  res = res || insert_val[i] ;
  if (insert_val[i]) h_Cross -> Fill(i);
 }
 if (res) h_Cross -> Fill(NN);

 return res;
}



// --------------------------------------------------------------------

bool L1Menu2012::SingleJetCentral(float cut ) {

        bool ok=false;
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue; 
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= cut) ok = true;
        } 
 
	return ok;

}
// --------------------------------------------------------------------
 
bool L1Menu2012::SingleJet(float cut ) {
        
        bool ok=false;
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= cut) ok = true;
        }

        return ok;

}

// --------------------------------------------------------------------

bool L1Menu2012::DoubleJetCentral(float cut1, float cut2 ) {

        int n1=0;
        int n2=0;
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= cut1) n1++;
                if (pt >= cut2) n2++;
        }
        bool ok = ( n1 >=1 && n2 >= 2);
	return ok;
        
}

// --------------------------------------------------------------------

bool L1Menu2012::DoubleJet_Eta1p7_deltaEta4(float cut1, float cut2 ) {

        int n1=0;
        int n2=0;
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                float eta = gt_ -> Etajet[ue];
                if (eta < 5.5 || eta > 15.5) continue;  // eta = 6 - 15
                if (pt >= cut1) n1++;
                if (pt >= cut2) n2++;
        }
        bool ok = ( n1 >=1 && n2 >= 2);
        if (! ok) return false;

	// -- now the correlation

	bool CORREL = false;

        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                float eta1 = gt_ -> Etajet[ue];
                if (eta1 < 5.5 || eta1 > 15.5) continue;  // eta = 6 - 15
		if (pt < cut1) continue;

       		for (int ve=0; ve < Nj; ve++) {
				if (ve == ue) continue;
                		int bx2 = gt_ -> Bxjet[ve];               //    BX = 0, +/- 1 or +/- 2
                		if (bx2 != 0) continue;
                		bool isFwdJet2 = gt_ -> Fwdjet[ve];
                		if (isFwdJet2) continue;
                		float rank2 = gt_ -> Rankjet[ve];
                		float pt2 = rank2 * 4;
                		float eta2 = gt_ -> Etajet[ve];
                		if (eta2 < 5.5 || eta2 > 15.5) continue;  // eta = 6 - 15
                		if (pt2 < cut2) continue;
			
				bool corr = correlateInEta((int)eta1, (int)eta2, 4);
				if (corr) CORREL = true;
		}


        }

	return CORREL ;

}

// --------------------------------------------------------------------

bool L1Menu2012::DoubleTauJetEta2p17(float cut1, float cut2) {

        int n1=0;
        int n2=0;
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue; 
                bool isTauJet = gt_ -> Taujet[ue];
                if (! isTauJet) continue;
                float rank = gt_ -> Rankjet[ue];    // the rank of the electron
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                float eta = gt_ -> Etajet[ue];
                if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                if (pt >= cut1) n1++;
                if (pt >= cut2) n2++;
        }  // end loop over jets
        
        bool ok = ( n1 >=1 && n2 >= 2);
	return ok;
        
}

// --------------------------------------------------------------------

bool L1Menu2012::TripleJetCentral(float cut1, float cut2, float cut3 ) {

        int n1=0;
        int n2=0;
        int n3=0;
        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= cut1) n1++;
                if (pt >= cut2) n2++;
                if (pt >= cut3) n3++;
        }

        bool ok = ( n1 >=1 && n2 >= 2 && n3 >= 3 );
	return ok;

}

// --------------------------------------------------------------------

bool L1Menu2012::TripleJet_VBF(float jet1, float jet2, float jet3 ) {

// jet1 >= jet2  >= jet3

        bool jet=false;        
        bool jetf1=false;           
        bool jetf2=false;   
        // bool jetf3=false;

        int n1=0;
        int n2=0;
        int n3=0;

        int f1=0;
        int f2=0;
        int f3=0;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

                if (isFwdJet) {
                    if (pt >= jet1) f1 ++;
                    if (pt >= jet2) f2 ++;
                    if (pt >= jet3) f3 ++;              
                } 
                else {
                    if (pt >= jet1) n1 ++;
                    if (pt >= jet2) n2 ++;
                    if (pt >= jet3) n3 ++;
                }    
        }

        jet   = ( n1 >= 1 && n2 >= 2 && n3 >= 3 ) ;        
        jetf1 = ( f1 >= 1 && n2 >= 1 && n3 >= 2 ) ;  // numbers change ofcourse    
        jetf2 = ( n1 >= 1 && f2 >= 1 && n3 >= 2 ) ;  
        // jetf3 = ( n1 >= 1 && n2 >= 2 && f3 >= 1 ) ;

        bool ok = false;
        // if( jet || jetf1 || jetf2 || jetf3 ) ok =true;
        if( jet || jetf1 || jetf2 ) ok =true;


	return ok;

}

bool L1Menu2012::QuadJetCentral(float cut1, float cut2, float cut3, float cut4 ) {

// cut1 >= cut2  >= cut3 >= cut4

        int n1=0;
        int n2=0;
        int n3=0;
        int n4=0;

        int Nj = gt_ -> Njet ;
        for (int ue=0; ue < Nj; ue++) {
                int bx = gt_ -> Bxjet[ue];               //    BX = 0, +/- 1 or +/- 2
                if (bx != 0) continue;
                bool isFwdJet = gt_ -> Fwdjet[ue];
                if (isFwdJet) continue;
                float rank = gt_ -> Rankjet[ue];
                float pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
                if (pt >= cut1) n1++;
                if (pt >= cut2) n2++;
                if (pt >= cut3) n3++;
                if (pt >= cut4) n4++;
        }
        
        bool ok = ( n1 >=1 && n2 >= 2 && n3 >= 3 && n4 >= 4);
	return ok;
                
}               
                
        

// --------------------------------------------------------------------

bool L1Menu2012::Jets() {

 insert_ibin = 0;
 
 InsertInMenu("L1_SingleJet16", SingleJet(16.) );
 InsertInMenu("L1_SingleJet36", SingleJet(36.) );
 InsertInMenu("L1_SingleJet52", SingleJet(52.) );
 InsertInMenu("L1_SingleJet68", SingleJet(68.) );
 InsertInMenu("L1_SingleJet92", SingleJet(92.) );
 InsertInMenu("L1_SingleJet128", SingleJet(128.) );
 
 InsertInMenu("L1_DoubleJetC36", DoubleJetCentral(36.,36.) );
 InsertInMenu("L1_DoubleJetC44_Eta1p74_WdEta4", DoubleJet_Eta1p7_deltaEta4(44.,44.) );
 InsertInMenu("L1_DoubleJetC52", DoubleJetCentral(52.,52.) );
 InsertInMenu("L1_DoubleJetC56_Eta1p74_WdEta4", DoubleJet_Eta1p7_deltaEta4(56.,56.) );
 InsertInMenu("L1_DoubleJetC56", DoubleJetCentral(56.,56.) );
 InsertInMenu("L1_DoubleJetC64", DoubleJetCentral(64.,64.) );

 InsertInMenu("L1_TripleJet_64_44_24_VBF", TripleJet_VBF(64.,44.,24.) );
 InsertInMenu("L1_TripleJet_64_48_28_VBF", TripleJet_VBF(64.,48.,28.) );
 InsertInMenu("L1_TripleJet_68_48_32_VBF", TripleJet_VBF(68.,48.,32.) );
 InsertInMenu("L1_TripleJetC_52_28_28", TripleJetCentral(52.,28.,28.) );
 
 InsertInMenu("L1_QuadJetC32", QuadJetCentral(32.,32.,32.,32.) );
 InsertInMenu("L1_QuadJetC36", QuadJetCentral(36.,36.,36.,36.) );
 InsertInMenu("L1_QuadJetC40", QuadJetCentral(40.,40.,40.,40.) );
 
 InsertInMenu("L1_DoubleTauJet44er", DoubleTauJetEta2p17(44.,44.) );

 int NN = insert_ibin;

 int kOFFSET_old = kOFFSET;
 for (int k=0; k < NN; k++) {
        TheTriggerBits[k + kOFFSET_old] = insert_val[k];
 }
 kOFFSET += insert_ibin;

 
 if (first) {

	NBITS_JETS = NN;

        for (int ibin=0; ibin < insert_ibin; ibin++) {
          TString l1name = (TString)insert_names[ibin];
          h_Jets -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
        }

        h_Jets-> GetXaxis() -> SetBinLabel(NN+1,"JETS");

        for (int k=1; k <= kOFFSET -kOFFSET_old; k++) {
           h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Jets -> GetXaxis() -> GetBinLabel(k) );
        }

 }

 bool res = false;
 for (int i=0; i < NN; i++) {
  res = res || insert_val[i] ;
  if (insert_val[i]) h_Jets -> Fill(i);
 }
 if (res) h_Jets -> Fill(NN);

 return res;
}



// --------------------------------------------------------------------

bool L1Menu2012::ETM(float ETMcut ) {

        float adc = gt_ -> RankETM ;
        float TheETM = adc / 2. ;
        
        if (TheETM < ETMcut) return false;
	return true;
        
}

// --------------------------------------------------------------------


bool L1Menu2012::HTT(float HTTcut) {
        
        float adc = gt_ -> RankHTT ;
        float TheHTT = adc / 2. ;
        
        if (TheHTT < HTTcut) return false;
	return true;
        
}

// --------------------------------------------------------------------


bool L1Menu2012::ETT(float ETTcut) {

        // .. no bit selection... approx. but in 179828 (10 b run)
        // the postDT rate was close to the preDT rate, hence overlap
        // with the enabled triggers should be good enough.
        
        float adc = gt_ -> RankETT ;
        float TheETT = adc / 2. ;
        
        if (TheETT < ETTcut) return false;
        
	return true;

}


// --------------------------------------------------------------------

bool L1Menu2012::Sums() {

 insert_ibin = 0;

 InsertInMenu("L1_ETM30", ETM(30.) );
 InsertInMenu("L1_ETM36", ETM(36.) );
 InsertInMenu("L1_ETM40", ETM(40.) );
 InsertInMenu("L1_ETM50", ETM(50.) );
 InsertInMenu("L1_ETM70", ETM(70.) );
 InsertInMenu("L1_ETM100", ETM(100.) );

 InsertInMenu("L1_HTT100", HTT(100.) );
 InsertInMenu("L1_HTT125", HTT(125.) );
 InsertInMenu("L1_HTT150", HTT(150.) );
 InsertInMenu("L1_HTT175", HTT(175.) );
 InsertInMenu("L1_HTT200", HTT(200.) );

 InsertInMenu("L1_ETT300", ETT(300.) );

 int NN = insert_ibin;

 int kOFFSET_old = kOFFSET;
 for (int k=0; k < NN; k++) {
        TheTriggerBits[k + kOFFSET_old] = insert_val[k];
 }
 kOFFSET += insert_ibin;


 if (first) {
     
	NBITS_SUMS = NN;

        for (int ibin=0; ibin < insert_ibin; ibin++) {
          TString l1name = (TString)insert_names[ibin]; 
          h_Sums -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
        }

     	h_Sums -> GetXaxis() -> SetBinLabel(NN+1,"SUMS");

        for (int k=1; k <= kOFFSET -kOFFSET_old; k++) {
           h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Sums -> GetXaxis() -> GetBinLabel(k) );
        }
 }
 
 bool res = false; 
 for (int i=0; i < NN; i++) {
  res = res || insert_val[i] ; 
  if (insert_val[i]) h_Sums -> Fill(i);
 }
 if (res) h_Sums -> Fill(NN);
 
 return res;
}



// --------------------------------------------------------------------

bool L1Menu2012::SingleEG(float cut ) {


                bool ok=false; 
                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {               
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ; 
                        if (pt >= cut) ok = true;
                }  // end loop over EM objects
        
	return ok; 

}

// --------------------------------------------------------------------

bool L1Menu2012::SingleIsoEG_Eta2p1(float cut ) {

                bool ok=false;
                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        bool iso = gt_ -> Isoel[ue];
                        if (! iso) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= cut) ok = true;
                }  // end loop over EM objects

	return ok;

}


// --------------------------------------------------------------------


// --------------------------------------------------------------------

bool L1Menu2012::SingleEG_Eta2p1(float cut ) {


                bool ok=false;
                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float eta = gt_ -> Etael[ue];
                        if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= cut) ok = true;
                }  // end loop over EM objects

	return ok;

}

// --------------------------------------------------------------------

bool L1Menu2012::DoubleEG(float cut1, float cut2 ) {

// cut1 >= cut2 
// e.g. DoubleEG_12_5  cut1 = 12, cut2 = 5
        
                int n1=0;
                int n2=0;
                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {               
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= cut1) n1++;
                        if (pt >= cut2) n2++;
                }  // end loop over EM objects

        bool ok = ( n1 >= 1 && n2 >= 2) ;
	return ok;

}



// --------------------------------------------------------------------

bool L1Menu2012::TripleEG(float cut1, float cut2, float cut3 ) {

// cut1 >= cut2 >= cut3
// e.g. TripleEG_10_7_5  cut1 = 10, cut2 = 7, cut3 = 5


                int n1=0;
                int n2=0;
                int n3=0;
                int Nele = gt_ -> Nele;
                for (int ue=0; ue < Nele; ue++) {
                        int bx = gt_ -> Bxel[ue];               //    BX = 0, +/- 1 or +/- 2
                        if (bx != 0) continue;
                        float rank = gt_ -> Rankel[ue];    // the rank of the electron
                        float pt = rank ;
                        if (pt >= cut1) n1++;
                        if (pt >= cut2) n2++;
                        if (pt >= cut3) n3++;
                }  // end loop over EM objects

        bool ok = ( n1 >= 1 && n2 >= 2 && n3 >= 3) ;
	return ok;


}

// --------------------------------------------------------------------

bool L1Menu2012::EGamma() {
        
 insert_ibin = 0;

 InsertInMenu("L1_SingleEG5", SingleEG(5.) );
 InsertInMenu("L1_SingleEG7", SingleEG(7.) );
 InsertInMenu("L1_SingleEG12", SingleEG(12.) );
 InsertInMenu("L1_SingleEG18er", SingleEG_Eta2p1(18.) );
 InsertInMenu("L1_SingleIsoEG18er", SingleIsoEG_Eta2p1(18.) );
 InsertInMenu("L1_SingleEG20", SingleEG(20.) );
 InsertInMenu("L1_SingleIsoEG20_Eta2p1", SingleIsoEG_Eta2p1(20.) );
 InsertInMenu("L1_SingleEG22", SingleEG(22.) );
 InsertInMenu("L1_SingleEG24", SingleEG(24.) );
 InsertInMenu("L1_SingleEG30", SingleEG(30.) );
 
 // InsertInMenu("L1_DoubleEG_15_5", DoubleEG(15.,5.) );
 InsertInMenu("L1_DoubleEG_13_7", DoubleEG(13.,7.) );

 InsertInMenu("L1_TripleEG7", TripleEG(7.,7.,7.) );
 InsertInMenu("L1_TripleEG_12_7_5", TripleEG(12.,7.,5.) );

 int NN = insert_ibin;

 int kOFFSET_old = kOFFSET;
 for (int k=0; k < NN; k++) {
        TheTriggerBits[k + kOFFSET_old] = insert_val[k];
 }
 kOFFSET += insert_ibin;

        
 if (first) {

        NBITS_EGAMMA = NN;

        for (int ibin=0; ibin < insert_ibin; ibin++) {
          TString l1name = (TString)insert_names[ibin];
          h_Egamma -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
        }
        h_Egamma-> GetXaxis() -> SetBinLabel(NN+1,"EGAMMA");

        for (int k=1; k <= kOFFSET -kOFFSET_old; k++) {
           h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Egamma -> GetXaxis() -> GetBinLabel(k) );
        }
 }                      
                        
 bool res = false;      
 for (int i=0; i < NN; i++) {
  res = res || insert_val[i] ;
  if (insert_val[i]) h_Egamma -> Fill(i);
 }      
 if (res) h_Egamma -> Fill(NN);
        
 return res;
}       
        



// --------------------------------------------------------------------

void L1Menu2012::Loop() {

  int nevents = GetEntries();
  cout << " -- Total number of entries : " << nevents << endl;

    // nevents = nevents / 10 ;

  int NPASS = 0; 

  int NJETS = 0;
  int NEG = 0;
  int NSUMS =0;
  int NMUONS = 0;
  int NCROSS = 0;

  int nPAG =0;
  first = true;

 int verbose = 10000;

 for (Long64_t i=0; i<nevents; i++)
  {     
        if (i % verbose == 0) {
           cout << "  ... iEvent " << i << endl;
           verbose = verbose * 10;
        }

      //load the i-th event
      Long64_t ientry = LoadTree(i); if (ientry < 0) break;
      GetEntry(i);
        
//      Fill the physics bits:
        FillBits();
             
        if (first) MyInit();

  bool raw = PhysicsBits[0];  // ZeroBias
  if (! raw) continue;


//  --- Reset the emulated "Trigger Bits"
	kOFFSET = 0;
	for (int k=0; k < N128; k++) {
	  TheTriggerBits[k] = false;
	}


	bool jets = false;
	bool eg = false;
	bool sums = false;
	bool muons = false;
	bool cross = false;

        cross = Cross();
        eg = EGamma();
        muons = Muons();
        jets = Jets() ;
        sums = Sums();


        bool pass  = jets || eg || sums || muons || cross  ;

	 if (pass) NPASS ++;

        if (cross) NCROSS ++;
        if (muons) NMUONS ++;
        if (sums) NSUMS ++;
        if (eg) NEG ++;
        if (jets) NJETS ++;

	if (pass) h_Block -> Fill(5.);

        bool dec[5];
        dec[0] = eg;
        dec[1] = jets;
        dec[2] = muons;
        dec[3] = sums;
        dec[4] = cross;
        for (int l=0; l < 5; l++) {
          if (dec[l]) {
		h_Block -> Fill(l);
                for (int k=0; k < 5; k++) {
                  if (dec[k]) cor_Block -> Fill(l,k);
                }
          }
        
        }

	 first = false;

	// -- now the pure rate stuff
	// -- kOFFSET now contains the number of triggers we have calculated
	
                bool ddd[NPAGS];
                for (int idd=0; idd < NPAGS; idd++) {
                  ddd[idd] = false; 
                } 

        float weightEvent = 1.;

	for (int k=0; k < kOFFSET; k++) {
		if ( ! TheTriggerBits[k] ) continue;
		h_All -> Fill(k);

		TString name = h_All -> GetXaxis() -> GetBinLabel(k+1);
		std::string L1namest = (std::string)name;
		bool IsTOP = setTOP.count(L1namest) > 0;
		bool IsHIGGS = setHIGGS.count(L1namest) > 0;
		bool IsBPH = setBPH.count(L1namest) > 0;
		bool IsEXO = setEXO.count(L1namest) > 0;
		bool IsSUSY = setSUSY.count(L1namest) > 0;
                bool IsSMP = setSMP.count(L1namest) > 0;
		if (IsHIGGS) ddd[0] = true;
		if (IsSUSY) ddd[1] = true;
		if (IsEXO) ddd[2] = true;
		if (IsTOP) ddd[3] = true;
                if (IsSMP) ddd[4] = true;
		if (IsBPH) ddd[5] = true;

		float ww = WeightsPAGs[L1namest];
		if (ww < weightEvent) weightEvent = ww;

		// did the event pass another trigger ?
		bool nonpure = false;
		for (int k2=0; k2 < kOFFSET; k2++) {
			if (k2 == k) continue;
			if ( TheTriggerBits[k2] ) nonpure = true;
		}
		bool pure = !nonpure ;
		if (pure) h_Pure -> Fill(k);
	}

	// -- for the PAG rates :
		bool PAG = false;
                for (int idd=0; idd < NPAGS; idd++) {
                 if (ddd[idd]) {
		    bool nonpure = false;
		    PAG = true;
                    for (int jdd=0; jdd < NPAGS; jdd++) {
                        if (ddd[jdd]) {
			   cor_PAGS -> Fill(idd,jdd);
			   if (jdd != idd) nonpure = true;
			}
                    }   
		    bool pure = ! nonpure;
		    if (pure) h_PAGS_pure -> Fill(idd);
		    h_PAGS_shared -> Fill(idd,weightEvent);

                 }  
                }
		if (PAG) nPAG ++;


  }  // end evt loop

  float scal = 1./(23.3) ;       // 1 LS
  scal = scal / NLUMIS ;
  scal = scal * 10.;      // nanoDST is p'ed by 10
  scal = scal / 1000.  ;    // rate in kHz
  
  scal = scal * LUMI / LUMIREF   ;    // scale up from LUMIREF to LUMI

  scal = scal * ZEROBIAS_PRESCALE  ;   // because ZeroBias was prescaled by ZEROBIAS_PRESCALE


 cout << endl;
 cout << " --------------------------------------------------------- " << endl;
 cout << " Rate that pass L1 " << NPASS * scal << " kHz  " << endl;
 cout << "        ( claimed by a PAG " << nPAG * scal << " kHz  i.e. " << 100.*(float)nPAG/(float)NPASS << "%. ) " << endl;
 cout << "    scaled to 8 TeV (1.2) and adding 3 kHz " << NPASS * scal * 1.2 + 3.0 << " kHz " << endl;

 h_Cross -> Scale(scal);
 h_Jets -> Scale(scal);
 h_Egamma -> Scale(scal);
 h_Sums -> Scale(scal);
 h_Muons -> Scale(scal);

 // h_All -> Scale(scal);
 MyScale(h_All, scal);

 h_Pure  -> Scale(scal);

 cout << " --------------------------------------------------------- " << endl;
 cout << " Rate that pass L1 jets " << NJETS * scal << " kHz  " << endl;
 cout << " Rate that pass L1 EG " << NEG * scal << " kHz  " << endl;
 cout << " Rate that pass L1 Sums " << NSUMS * scal << " kHz  " << endl;
 cout << " Rate that pass L1 Muons " << NMUONS * scal << " kHz  " << endl;
 cout << " Rate that pass L1 Cross " << NCROSS * scal << " kHz  " << endl;

for (int i=1; i<= 5; i++) {
        float nev = h_Block -> GetBinContent(i);
        for (int j=1; j<= 5; j++) {
          int ibin = cor_Block -> FindBin(i-1,j-1);
          float val = cor_Block -> GetBinContent(ibin);
          val = val / nev;
          cor_Block -> SetBinContent(ibin,val);
        }
}   

 h_Block -> Scale(scal);

 cor_PAGS -> Scale(scal);
 h_PAGS_pure -> Scale(scal);
 h_PAGS_shared -> Scale(scal);


 cout << endl;
 int NBITS_ALL = NBITS_MUONS + NBITS_EGAMMA + NBITS_JETS + NBITS_SUMS + NBITS_CROSS;

 cout << " --- TOTAL NUMBER OF BITS USED : " << NBITS_ALL << endl;
 cout << "                          MUONS : " << NBITS_MUONS << endl;
 cout << "                          EGAMMA : " << NBITS_EGAMMA << endl;
 cout << "                          JETS : " << NBITS_JETS << endl;
 cout << "                          SUMS : " << NBITS_SUMS << endl;
 cout << "                          CROSS : " << NBITS_CROSS << endl;


}

// --------------------------------------------------------------------

void RunL1() {

/*
int CrossBins = 29;
int MuonBins = 23;
int SumsBins = 13;
int JetsBins = 21;
int EGBins = 15;

 h_Cross = new TH1F("h_Cross","h_Cross",CrossBins,-0.5,(float)CrossBins-0.5);
 h_Sums = new TH1F("h_Sums","h_Sums",SumsBins,-0.5,(float)SumsBins-0.5);
 h_Jets = new TH1F("h_Jets","h_Jets",JetsBins,-0.5,(float)JetsBins-0.5);
 h_Egamma = new TH1F("h_Egamma","h_Egamma",EGBins,-0.5,(float)EGBins-0.5);
 h_Muons = new TH1F("h_Muons","h_Muons",MuonBins,-0.5,(float)MuonBins-0.5); 
*/

 int Nbin_max = 50;
 h_Cross = new TH1F("h_Cross","h_Cross",Nbin_max,-0.5,(float)Nbin_max-0.5);
 h_Sums = new TH1F("h_Sums","h_Sums",Nbin_max,-0.5,(float)Nbin_max-0.5);
 h_Jets = new TH1F("h_Jets","h_Jets",Nbin_max,-0.5,(float)Nbin_max-0.5);
 h_Egamma = new TH1F("h_Egamma","h_Egamma",Nbin_max,-0.5,(float)Nbin_max-0.5);
 h_Muons = new TH1F("h_Muons","h_Muons",Nbin_max,-0.5,(float)Nbin_max-0.5);


 h_Block = new TH1F("h_Block","h_Block",6,-0.5,5.5);
 cor_Block = new TH2F("cor_Block","cor_Block",5,-0.5,4.5,5,-0.5,4.5);

 cor_PAGS = new TH2F("cor_PAGS","cor_PAGS",NPAGS,-0.5,(float)NPAGS-0.5,NPAGS,-0.5,(float)NPAGS-0.5);
 h_PAGS_pure = new TH1F("h_PAGS_pure","h_PAGS_pure",NPAGS,-0.5,(float)NPAGS-0.5);
 h_PAGS_shared = new TH1F("h_PAGS_shared","h_PAGS_shared",NPAGS,-0.5,(float)NPAGS-0.5);


 h_All = new TH1F("h_All","h_All",N128,-0.5,N128-0.5);
 h_Pure = new TH1F("h_Pure","h_Pure",N128,-0.5,N128-0.5);


 L1Menu2012 a;

 string filename = L1NtupleFileName; 

 a.Open(filename);

 a.Loop();

TString YaxisName;
if (LUMI == 50.) YaxisName = "Rate (kHz) at 5e33 (PU = 28)";
if (LUMI == 70.) YaxisName = "Rate (kHz) at 7e33 (PU = 33)";
if (LUMI == 70.001) YaxisName = "Rate (kHz) at 7e33 (PU = 33)";


 TCanvas* c1 = new TCanvas("c1","c1");
 c1 -> cd();
 gStyle -> SetOptStat(0);
 h_Cross -> SetLineColor(4);
 h_Cross -> GetXaxis() -> SetLabelSize(0.035);
 h_Cross -> SetYTitle(YaxisName);
 h_Cross -> Draw();

 TCanvas* c2 = new TCanvas("c2","c2");
 c2 -> cd();
 h_Sums -> SetLineColor(4);
 h_Sums -> GetXaxis() -> SetLabelSize(0.035);
 h_Sums -> SetYTitle(YaxisName);
 h_Sums -> Draw();

 TCanvas* c3 = new TCanvas("c3","c3");
 c3 -> cd();
 h_Egamma -> SetLineColor(4);
 h_Egamma -> GetXaxis() -> SetLabelSize(0.035);
 h_Egamma -> SetYTitle(YaxisName);
 h_Egamma -> Draw();

 TCanvas* c4 = new TCanvas("c4","c4");
 c4 -> cd();
 h_Jets -> SetLineColor(4);
 h_Jets -> GetXaxis() -> SetLabelSize(0.035);
 h_Jets -> SetYTitle(YaxisName);
 h_Jets -> Draw();

 TCanvas* c5 = new TCanvas("c5","c5");
 c5 -> cd();
 h_Muons -> SetLineColor(4);
 h_Muons -> GetXaxis() -> SetLabelSize(0.035);
 h_Muons -> SetYTitle(YaxisName);
 h_Muons -> Draw();


 TCanvas* c6 = new TCanvas("c6","c6");
 c6 -> cd();
 cor_Block -> GetXaxis() -> SetBinLabel(1,"EG");
 cor_Block -> GetXaxis() -> SetBinLabel(2,"Jets");
 cor_Block -> GetXaxis() -> SetBinLabel(3,"Muons");
 cor_Block -> GetXaxis() -> SetBinLabel(4,"Sums");
 cor_Block -> GetXaxis() -> SetBinLabel(5,"Cross");

 cor_Block -> GetYaxis() -> SetBinLabel(1,"EG");
 cor_Block -> GetYaxis() -> SetBinLabel(2,"Jets");
 cor_Block -> GetYaxis() -> SetBinLabel(3,"Muons");
 cor_Block -> GetYaxis() -> SetBinLabel(4,"Sums");
 cor_Block -> GetYaxis() -> SetBinLabel(5,"Cross");

 cor_Block -> Draw("colz");
 cor_Block -> Draw("same,text");

 TCanvas* c7 = new TCanvas("c7","c7");
 c7 -> cd();

 cor_PAGS -> GetXaxis() -> SetBinLabel(1,"HIGGS");
 cor_PAGS -> GetXaxis() -> SetBinLabel(2,"SUSY");
 cor_PAGS -> GetXaxis() -> SetBinLabel(3,"EXO");
 cor_PAGS -> GetXaxis() -> SetBinLabel(4,"TOP");
 cor_PAGS -> GetXaxis() -> SetBinLabel(5,"SMP");
 cor_PAGS -> GetXaxis() -> SetBinLabel(6,"BPH");

 cor_PAGS -> GetYaxis() -> SetBinLabel(1,"HIGGS");
 cor_PAGS -> GetYaxis() -> SetBinLabel(2,"SUSY");
 cor_PAGS -> GetYaxis() -> SetBinLabel(3,"EXO");
 cor_PAGS -> GetYaxis() -> SetBinLabel(4,"TOP");
 cor_PAGS -> GetYaxis() -> SetBinLabel(5,"SMP");
 cor_PAGS -> GetYaxis() -> SetBinLabel(6,"BPH");

 cor_PAGS -> Draw("colz");
 cor_PAGS -> Draw("same,text"); 

 TCanvas* c8 = new TCanvas("c8","c8");
 c8 -> cd();
 h_PAGS_pure -> GetXaxis() -> SetBinLabel(1,"HIGGS");
 h_PAGS_pure -> GetXaxis() -> SetBinLabel(2,"SUSY");
 h_PAGS_pure -> GetXaxis() -> SetBinLabel(3,"EXO");
 h_PAGS_pure -> GetXaxis() -> SetBinLabel(4,"TOP");
 h_PAGS_pure -> GetXaxis() -> SetBinLabel(5,"SMP");
 h_PAGS_pure -> GetXaxis() -> SetBinLabel(6,"BPH");
 h_PAGS_pure -> SetYTitle("Pure rate (kHz)");
 h_PAGS_pure -> Draw();

 TCanvas* c9 = new TCanvas("c9","c9");
 c9 -> cd();
 h_PAGS_shared -> GetXaxis() -> SetBinLabel(1,"HIGGS");
 h_PAGS_shared -> GetXaxis() -> SetBinLabel(2,"SUSY");
 h_PAGS_shared -> GetXaxis() -> SetBinLabel(3,"EXO");
 h_PAGS_shared -> GetXaxis() -> SetBinLabel(4,"TOP");
 h_PAGS_shared -> GetXaxis() -> SetBinLabel(5,"SMP");
 h_PAGS_shared -> GetXaxis() -> SetBinLabel(6,"BPH");
 h_PAGS_shared -> SetYTitle("Shared rate (kHz)");
 h_PAGS_shared -> Draw();




        // -- kOFFSET now contains the number of triggers we have calculated

 for (int k=1; k < kOFFSET+1; k++) {
	TString name = h_All -> GetXaxis() -> GetBinLabel(k);
	float rate = h_All -> GetBinContent(k);
	float err_rate  = h_All -> GetBinError(k);
	float pure = h_Pure -> GetBinContent(k);
	std::string L1namest = (std::string)name;
	map<string, int>::const_iterator it = a.Prescales.find(L1namest);
	float pre;
       if (it == a.Prescales.end() ) {
        cout << " --- SET P = 1 FOR SEED :  " << L1namest << endl;
        pre = 1;
        }
        else {
          pre = it -> second;
        }
       bool bias = a.Biased[L1namest];
        if (bias) cout << name << " \t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" << pure << "\t" << " ***  BIAS  *** " << endl;
         else
      cout << name << " \t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" <<  pure  << endl;

 }




}



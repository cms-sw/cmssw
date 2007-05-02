
#ifndef TtJetCombLRsetup_h
#define TtJetCombLRsetup_h


///////////////////////
//Overall Constants  //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const  int       nrFiles  	  = 5;
const  TString   path     	  = "dcap://maite.iihe.ac.be/pnfs/iihe/becms/heyninck/CMSSW_1_2_0_TtEvents/TtSemiLepEvents_";
const  bool  	useSpaceAngle     = true;
const  double 	SumAlphaCut  	  = 0.7;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////








///////////////////////////////////////////////////////////
// B-selection LR constants, observable & fit Definition //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//
// B-selection LR
//
const  bool  	doBSelLRObsLoop   	= true;
const  bool  	doBSelPurEffLoop  	= true;
const  int   	nrBSelHistBins    	= 50;
const  int   	nrBSelLRtotBins   	= 30;
const  double 	BSelLRtotMin   	  	= -10;
const  double 	BSelLRtotMax      	= 0;
const  TString   BSeloutfileName   	= "../data/TtSemiBJetSelectionLR.root";
const  int       nrBSelObs  		= 3;
const  int       bSelObs[nrBSelObs] 	= {0,1,3};
const  double    bSelObsMin[nrBSelObs]	= {0,-20,0};
const  double    bSelObsMax[nrBSelObs]	= {1,15,1};

inline static vector<double> getBSelObsValues(TtSemiEvtSolution *sol, TF1 *bprobfit){
    vector<double> vals;
 
    //obs0: solutions' Chi2-value of the kinfit
    vals.push_back(sol->getChi2());
    
    //obs1: combined b-tag information
    vals.push_back(log(bprobfit->Eval(sol->getLepb().getBdiscriminant())    *     bprobfit->Eval(sol->getHadb().getBdiscriminant()) 
                  *  (1-bprobfit->Eval(sol->getHadp().getBdiscriminant())) * (1 - bprobfit->Eval(sol->getHadq().getBdiscriminant()))));
    /*if(sol->getHadb().getBdiscriminant()>-9 && sol->getLepb().getBdiscriminant()>-9 ) {
      vals.push_back(sol->getLepb().getBdiscriminant() + sol->getHadb().getBdiscriminant()); 
    }
    else
    {
      vals.push_back(-20); 
    }*/
    //obs2: dummy
    vals.push_back(2.);
    
    //obs3: (pt_b1 + pt_b2)/(sum jetpt)
     vals.push_back((sol->getHadb().getBCalJet().et()+sol->getLepb().getBCalJet().et())
                   /(sol->getHadp().getLCalJet().et()+sol->getHadq().getLCalJet().et()+sol->getHadb().getBCalJet().et()+sol->getLepb().getBCalJet().et()));

    return vals;
}

inline static vector<TF1> getBSelObsFitFunctions(){
    vector<TF1> fits;
    TFormula gauss("gauss", "gaus");
    TFormula symgauss("symgauss", "[0]*(exp(-0.5*(x/[1])**2))");
    TFormula dblgauss("dblgauss", "[0]*(exp(-0.5*((x-[1])/[2])**2)+exp(-0.5*((x+[3])/[4])**2))");
    TFormula symdblgauss("symdblgauss", "[0]*(exp(-0.5*((x-[1])/[2])**2)+exp(-0.5*((x+[1])/[2])**2))");
    TFormula sigm("sigm", "[0]/(1 + 1/exp([1]*([2] - x)))");
    TFormula sigmc("sigmc", "[0]/(1 + 1/exp([1]*([2] - x)))+[3]");
    TFormula dblsigm("dblsigm", "[0]/(1 + 1/exp([1]**2*([2] - x)))/(1 + 1/exp([3]**2*(x - [4])))");
    TFormula symdblsigm("symdblsigm", "[0]/(1 + 1/exp([1]**2*([2] - x)))/(1 + 1/exp([1]**2*([2] + x)))");
    
    //fit function for obs0
    fits.push_back(TF1("fSoverB_Obs0", "sigm+sigm+sigm", 0, 1));
    fits[0].SetParameters(0.14, -27, .24, 0.46, -19, .58, 0.23, -53, .80);
    
    //fit function for obs1
    fits.push_back(TF1("fSoverB_Obs1", "sigm+sigm+sigm", 0, 1));
    fits[1].SetParameters(0.14, -27, .24, 0.46, -19, .58, 0.23, -53, .80);
    
    //fit function for obs2
    fits.push_back(TF1("fSoverB_Obs2", "pol4+sigm", 0, 1));
    fits[2].SetParameters(0.27, .42, -3.9, 8.3, -5.1, .3, -2000, .35);
    
    //fit function for obs3
    fits.push_back(TF1("fSoverB_Obs3", "[0]-[1]*exp(-[2]*pow(1000.*x,[3]))", 0, 1));
    fits[3].SetParameters(0.8,0.5,0.04,1.);
    return fits;
}   
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////










//////////////////////////////////////////////////////
// B-hadr LR constants, observable & fit Definition //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//select which observables you want to use in the LR-method
const  bool  	doBhadrLRObsLoop   	= true;
const  bool  	doBhadrPurEffLoop  	= true;
const  int   	nrBhadrHistBins    	= 50;
const  int   	nrBhadrLRtotBins   	= 30;
const  double 	BhadrLRtotMin      	= -10;
const  double 	BhadrLRtotMax      	= 0;
const  TString   BhadroutfileName   	= "../data/TtSemiBhadrSelectionLR.root";
const  int       nrBhadrObs   	  	= 3;
const  int       bHadrObs[nrBhadrObs] 	= {0,1,2};
const  float     bHadrObsMin[nrBhadrObs]	= {0,0,0};
const  float     bHadrObsMax[nrBhadrObs]	= {5,5,1};


inline static vector<double> getBhadrObsValues(TtSemiEvtSolution *sol, TF1 *bprobfit){
    vector<double> vals;
 
    //obs0: DeltaR fitted blept-lep
    double phidiff = 0, etadiff = 0;
    if(sol->getDecay() == "muon") {
      phidiff = fabs(sol->getLepb().getBCalJet().phi() - sol->getMuon().getRecMuon().phi()); 
      etadiff = fabs(sol->getLepb().getBCalJet().eta() - sol->getMuon().getRecMuon().eta());
    }
    else if(sol->getDecay() == "electron") {
      phidiff = fabs(sol->getLepb().getBCalJet().phi() - sol->getElectron().getRecElectron().phi()); 
      etadiff = fabs(sol->getLepb().getBCalJet().eta() - sol->getElectron().getRecElectron().eta());
    }
    if(phidiff>3.141596) phidiff = 2.*3.14159 - phidiff;
    vals.push_back(sqrt(pow(phidiff,2)+pow(etadiff,2)));
    
    //obs1: DeltaR fitted Whadr-bhadr
    phidiff = fabs(sol->getHadb().getBCalJet().phi() - sol->getCalHadW().phi());
    if(phidiff>3.141596) phidiff = 2.*3.14159 - phidiff;
    etadiff = fabs(sol->getHadb().getBCalJet().eta() - sol->getCalHadW().eta());
    vals.push_back(sqrt(pow(phidiff,2)+pow(etadiff,2)));
    
    //obs2: vector pt_top sum/scalar sum pt_top
     double pxsum = sol->getHadp().getLCalJet().px()+sol->getHadq().getLCalJet().px()+sol->getHadb().getBCalJet().px();
     double pysum = sol->getHadp().getLCalJet().py()+sol->getHadq().getLCalJet().py()+sol->getHadb().getBCalJet().py();
     double ptsum = sol->getHadp().getLCalJet().pt()+sol->getHadq().getLCalJet().pt()+sol->getHadb().getBCalJet().pt();
     
     vals.push_back(sqrt(pow(pxsum,2)+pow(pysum,2))/ptsum);

    return vals;
}



inline static vector<TF1> getBhadrObsFitFunctions(){
    vector<TF1> fits;
    TFormula gauss("gauss", "gaus");
    TFormula symgauss("symgauss", "[0]*(exp(-0.5*(x/[1])**2))");
    TFormula dblgauss("dblgauss", "[0]*(exp(-0.5*((x-[1])/[2])**2)+exp(-0.5*((x+[3])/[4])**2))");
    TFormula symdblgauss("symdblgauss", "[0]*(exp(-0.5*((x-[1])/[2])**2)+exp(-0.5*((x+[1])/[2])**2))");
    TFormula sigm("sigm", "[0]/(1 + 1/exp([1]*([2] - x)))");
    TFormula sigmc("sigmc", "[0]/(1 + 1/exp([1]*([2] - x)))+[3]");
    TFormula dblsigm("dblsigm", "[0]/(1 + 1/exp([1]**2*([2] - x)))/(1 + 1/exp([3]**2*(x - [4])))");
    TFormula symdblsigm("symdblsigm", "[0]/(1 + 1/exp([1]**2*([2] - x)))/(1 + 1/exp([1]**2*([2] + x)))");
    
    //fit function for obs0
    fits.push_back(TF1("fSoverB_Obs0", "sigm+sigm+sigm", 0, 1));
    fits[0].SetParameters(0.14, -27, .24, 0.46, -19, .58, 0.23, -53, .80);
    
    //fit function for obs1
    fits.push_back(TF1("fSoverB_Obs1", "sigm+sigm+sigm", 0, 1));
    fits[1].SetParameters(0.14, -27, .24, 0.46, -19, .58, 0.23, -53, .80);
    
    //fit function for obs2
    fits.push_back(TF1("fSoverB_Obs2", "pol4+sigm", 0, 1));
    fits[2].SetParameters(0.27, .42, -3.9, 8.3, -5.1, .3, -2000, .35);
    
    return fits;
}   
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////








///////////////////////////////////////////////////////////////////////////////
//
//  Function to get the Probability that a True Jet Combination exists
///////////////////////////////////////////////////////////////////////////////

inline static double getPExistingTrueComb(TString BSeloutfileName){ 
    //re-open output file & read observable fits
    TFile bSelOutfile(BSeloutfileName, "UPDATE");
    bSelOutfile.cd();
    TH1F *hAngleDiff = (TH1F*) bSelOutfile.GetKey("hAngleDiff") -> ReadObj();
    bool found = false;
    int bestbin = 0;
    while(bestbin<hAngleDiff->GetNbinsX() && !found){
      ++bestbin;
      if(SumAlphaCut > hAngleDiff -> GetXaxis()-> GetBinLowEdge(bestbin) && SumAlphaCut < hAngleDiff -> GetXaxis()-> GetBinUpEdge(bestbin))	found = true;
    }
    double frac = hAngleDiff->Integral(0,bestbin)/hAngleDiff->Integral(0,hAngleDiff->GetNbinsX());
    bSelOutfile.Close();
    return frac;
}









///////////////////////////////////////////////////////////////////////////////
//
//  Function to read the Fit Functions
///////////////////////////////////////////////////////////////////////////////

//b-prob
inline static TF1 readBTagLR(TString rootFilePath) {
  TFile * bProbLRFile = new TFile(rootFilePath);
  TH1 * bProbLRHist = (TH1 *) bProbLRFile->GetKey("obsHist3")->ReadObj();
  TF1 combBTagFit = *bProbLRHist->GetFunction("obsFit");
  delete bProbLRFile;
  return combBTagFit;
}

//b-jet selection LR
inline static vector<TF1> getBSelObsFits(TString BSeloutfileName){  
    vector<TF1> fits;
    //re-open output file & read observable fits
    TFile bSelOutfile(BSeloutfileName, "UPDATE");
    bSelOutfile.cd();
    for (int j = 0; j < nrBSelObs; j++) {
      TString th =  "hSoverB_Obs"; th += bSelObs[j];
      TString tf =  "fSoverB_Obs"; tf += bSelObs[j];
      fits.push_back((*((TF1 *) ((TH1F*) bSelOutfile.GetKey((TString)(th)) -> ReadObj()) -> GetFunction((TString)(tf)))));
    }
    bSelOutfile.Close();
    return fits;
}

inline static TF1 getBSelPurVsLRtotFit(TString BSeloutfileName){ 
    //re-open output file & read observable fits
    TFile bSelOutfile(BSeloutfileName, "UPDATE");
    bSelOutfile.cd();
    TF1 fit = *((TF1 *) ((TH1F*) bSelOutfile.GetKey("hPurity") -> ReadObj()) -> GetFunction("fBSelPurity"));
    bSelOutfile.Close();
    return fit;
}

//b-hadr selection LR
inline static vector<TF1> getBhadrObsFits(TString BhadroutfileName){  
    vector<TF1> fits;
    //re-open output file & read observable fits
    TFile bHadrOutfile(BhadroutfileName, "UPDATE");
    bHadrOutfile.cd();
    for (int j = 0; j < nrBhadrObs; j++) {
      TString th =  "hSoverB_Obs"; th += bHadrObs[j];
      TString tf =  "fSoverB_Obs"; tf += bHadrObs[j];
      fits.push_back((*((TF1 *) ((TH1F*) bHadrOutfile.GetKey((TString)(th)) -> ReadObj()) -> GetFunction((TString)(tf)))));
    }
    bHadrOutfile.Close();
    return fits;
}

inline static TF1 getBhadrPurVsLRtotFit(TString BhadroutfileName){ 
    //re-open output file & read observable fits
    TFile bHadrOutfile(BhadroutfileName, "UPDATE");
    bHadrOutfile.cd();
    TF1 fit = *((TF1 *) ((TH1F*) bHadrOutfile.GetKey("hPurity") -> ReadObj()) -> GetFunction("fBhadrPurity"));
    bHadrOutfile.Close();
    return fit;
}

#endif

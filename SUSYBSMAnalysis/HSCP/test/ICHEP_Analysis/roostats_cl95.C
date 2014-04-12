static const char* desc =
"=====================================================================\n"
"|                                                                    \n"
"|\033[1m        roostats_cl95.C  version 1.15                 \033[0m\n"
"|                                                                    \n"
"| Standard c++ routine for 95% C.L. limit calculation                \n"
"| for cross section in a 'counting experiment'                       \n"
"| Fully backwards-compatible with the CL95 macro                     \n"
"|                                                                    \n"
"| also known as 'CL95 with RooStats'                                 \n"
"|                                                                    \n"
"|\033[1m Gena Kukartsev, Stefan Schmitz, Gregory Schott       \033[0m\n"
"|\033[1m Lorenzo Moneta (CLs core)                            \033[0m\n"
"|\033[1m Michael Segala (Feldman-Cousins)                     \033[0m\n"
"|                                                                    \n"
"| July  2010: first version                                          \n"
"| March 2011: restructuring, interface change, expected limits       \n"
"| May 2011:   added expected limit median,                           \n"
"|             68%, 95% quantile bands and actual coverage            \n"
"| July 2011:  added CLs observed and expected limits                 \n"
"|             added option to run using Feldman Cousins              \n"
"|                                                                    \n"
"=====================================================================\n"
"                                                                     \n"
"Prerequisites:                                                       \n"
"                ROOT version 5.30.00 or higher                       \n"
"                                                                     \n"
"                                                                     \n"
"                                                                     \n"
"The code should be compiled in ROOT:                                 \n"
"                                                                     \n"
"root -l                                                              \n"
"                                                                     \n"
".L roostats_cl95.C+                                                  \n"
"                                                                     \n"
"Usage:                                                               \n"
" Double_t             limit = roostats_cl95(ilum, slum, eff, seff, bck, sbck, n, gauss = false, nuisanceModel, method, plotFileName, seed); \n"
" LimitResult expected_limit = roostats_clm(ilum, slum, eff, seff, bck, sbck, ntoys, nuisanceModel, method, seed); \n"
" Double_t     average_limit = roostats_cla(ilum, slum, eff, seff, bck, sbck, nuisanceModel, method, seed); \n"
"                                                                     \n"
" LimitResult limit = roostats_limit(ilum, slum, eff, seff, bck, sbck, n, gauss = false, nuisanceModel, method, plotFileName, seed); \n"
" Double_t obs_limit = limit.GetObservedLimit();                      \n"
" Double_t exp_limit = limit.GetExpectedLimit();                      \n"
" Double_t exp_up    = limit.GetOneSigmaHighRange();                  \n"
" Double_t exp_down  = limit.GetOneSigmaLowRange();                   \n"
" Double_t exp_2up   = limit.GetTwoSigmaHighRange();                  \n"
" Double_t exp_2down = limit.GetTwoSigmaLowRange();                   \n"
"                                                                     \n"
"Inputs:                                                              \n"
"       ilum          - Nominal integrated luminosity (pb^-1)         \n"
"       slum          - Absolute error on the integrated luminosity   \n"
"       eff           - Nominal value of the efficiency times         \n"
"                       acceptance (in range 0 to 1)                  \n"
"       seff          - Absolute error on the efficiency times        \n"
"                       acceptance                                    \n"
"       bck           - Nominal value of the background estimate      \n"
"       sbck          - Absolute error on the background              \n"
"       n             - Number of observed events (not used for the   \n"
"                       expected limit)                               \n"
"       ntoys         - Number of pseudoexperiments to perform for    \n"
"                       expected limit calculation)                   \n"
"       gauss         - if true, use Gaussian statistics for signal   \n"
"                       instead of Poisson; automatically false       \n"
"                       for n = 0.                                    \n"
"                       Always false for expected limit calculations  \n"
"       nuisanceModel - distribution function used in integration over\n"
"                       nuisance parameters:                          \n"
"                       0 - Gaussian (default), 1 - lognormal,        \n"
"                       2 - gamma;                                    \n"
"                       (automatically 0 when gauss == true)          \n"
"       method        - method of statistical inference:              \n"
"                       \"bayesian\"  - Bayesian with numeric         \n"
"                                       integration (default),        \n"
"                       \"mcmc\"      - another implementation of     \n"
"                                       Bayesian, not optimized,      \n"
"                                       to be used for cross checks   \n"
"                                       only!                         \n"
"                       \"cls\"       - CLs observed limit. We suggest\n"
"                                       using the dedicated interface \n"
"                                       roostats_cls() instead        \n"
"                       \"fc\"        - Feldman Cousins with numeric  \n"
"                                     integration,                    \n"
"                       \"workspace\" - only create workspace and save\n"
"                                     to file, no interval calculation\n"
"       plotFileName  - file name for the control plot to be created  \n"
"                       file name extension will define the format,   \n"
"                       <plot_cl95.pdf> is the default value,         \n"
"                       specify empty string if you do not want       \n"
"                       the plot to be created (saves time)           \n"
"       seed          - seed for random number generation,            \n"
"                       specify 0 for unique irreproducible seed      \n"
"                                                                     \n"
"                                                                     \n"
"The statistics model in this routine: the routine addresses the task \n"
"of a Bayesian evaluation of limits for a one-bin counting experiment \n"
"with systematic uncertainties on luminosity and efficiency for the   \n"
"signal and a global uncertainty on the expected background (implying \n"
"no correlated error on the luminosity for signal and  background,    \n"
"which will not be suitable for all use cases!). The observable is the\n"
"measured number of events.                                           \n"
"                                                                     \n"
"For more details see                                                 \n"
"        https://twiki.cern.ch/twiki/bin/view/CMS/RooStatsCl95        \n"
"                                                                     \n"
"\033[1m       Note!                                           \033[0m\n"
"If you are running nonstandard ROOT environment, e.g. in CMSSW,      \n"
"you need to make sure that the RooFit and RooStats header files      \n"
"can be found since they might be in a nonstandard location.          \n"
"                                                                     \n"
"For CMSSW_4_2_0_pre8 and later, add the following line to your       \n"
"rootlogon.C:                                                         \n"
"      gSystem -> SetIncludePath( \"-I$ROOFITSYS/include\" );         \n";


#include <algorithm>

#include "TCanvas.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TUnixSystem.h"
#include "TStopwatch.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TLine.h"

#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooRandom.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/SimpleInterval.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/MCMCInterval.h"
#include "RooStats/MCMCIntervalPlot.h"
#include "RooStats/FeldmanCousins.h"
#include "RooStats/PointSetInterval.h"
#include "RooStats/ConfidenceBelt.h"
#include "RooStats/ProposalHelper.h"
#include "RooStats/HybridCalculator.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/HypoTestPlot.h"
#include "RooStats/NumEventsTestStat.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/SimpleLikelihoodRatioTestStat.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"
#include "RooStats/MaxLikelihoodEstimateTestStat.h"
#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/HypoTestInverterPlot.h"

// FIXME: remove namespaces
using namespace RooFit;
using namespace RooStats;
using namespace std;

class LimitResult;

Double_t roostats_cl95(Double_t ilum, Double_t slum,
		       Double_t eff, Double_t seff,
		       Double_t bck, Double_t sbck,
		       Int_t n,
		       Bool_t gauss = kFALSE,
		       Int_t nuisanceModel = 0,
		       std::string method = "bayesian",
		       std::string plotFileName = "plot_cl95.pdf",
		       UInt_t seed = 12345,
		       LimitResult * pLimitResult = 0);

LimitResult roostats_clm(Double_t ilum, Double_t slum,
			 Double_t eff, Double_t seff,
			 Double_t bck, Double_t sbck,
			 Int_t nit = 200, Int_t nuisanceModel = 0,
			 std::string method = "bayesian",
			 UInt_t seed = 12345);

// legacy support: use roostats_clm() instead
Double_t roostats_cla(Double_t ilum, Double_t slum,
		      Double_t eff, Double_t seff,
		      Double_t bck, Double_t sbck,
		      Int_t nuisanceModel = 0,
		      std::string method = "bayesian",
		      UInt_t seed = 12345);




// ---> implementation below --------------------------------------------


class LimitResult{

  friend class CL95Calc;
  
public:
  LimitResult():
    _observed_limit(0),
    _observed_limit_error(0),
    _expected_limit(0),
    _low68(0),
    _high68(0),
    _low95(0),
    _high95(0),
    _cover68(0),
    _cover95(0){};

  // copy constructor
  LimitResult(const LimitResult & other):
    _observed_limit(other._observed_limit),
    _observed_limit_error(other._observed_limit_error),
    _expected_limit(other._expected_limit),
    _low68(other._low68),
    _high68(other._high68),
    _low95(other._low95),
    _high95(other._high95),
    _cover68(other._cover68),
    _cover95(other._cover95){}

  ~LimitResult(){};

  Double_t GetObservedLimit(){return _observed_limit;};
  Double_t GetObservedLimitError(){return _observed_limit_error;};
  Double_t GetExpectedLimit(){return _expected_limit;};

  Double_t GetOneSigmaLowRange(){return _low68;};
  Double_t GetOneSigmaHighRange(){return _high68;};
  Double_t GetOneSigmaCoverage(){return _cover68;};

  Double_t GetTwoSigmaLowRange(){return _low95;};
  Double_t GetTwoSigmaHighRange(){return _high95;};
  Double_t GetTwoSigmaCoverage(){return _cover95;};

private:
  Double_t _observed_limit;
  Double_t _observed_limit_error;
  Double_t _expected_limit;
  Double_t _low68;
  Double_t _high68;
  Double_t _low95;
  Double_t _high95;
  Double_t _cover68;
  Double_t _cover95;
};


class CL95Calc{

public:
  CL95Calc();
  CL95Calc( UInt_t seed );
  ~CL95Calc();

  RooWorkspace * makeWorkspace(Double_t ilum, Double_t slum,
			       Double_t eff, Double_t seff,
			       Double_t bck, Double_t sbck,
			       Bool_t gauss,
			       Int_t nuisanceModel);
  RooWorkspace * getWorkspace(){ return ws;}

  RooAbsData * makeData(Int_t n);

  Double_t cl95(std::string method = "bayesian", LimitResult * result = 0);

  Double_t cla( Double_t ilum, Double_t slum,
		Double_t eff, Double_t seff,
		Double_t bck, Double_t sbck,
		Int_t nuisanceModel,
		std::string method );
  
  LimitResult clm(Double_t ilum, Double_t slum,
		  Double_t eff, Double_t seff,
		  Double_t bck, Double_t sbck,
		  Int_t nit = 200, Int_t nuisanceModel = 0,
		  std::string method = "bayesian");
  
  int makePlot( std::string method,
		std::string plotFileName = "plot_cl95.pdf" );

  Double_t FC_calc(int Nbins, float conf_int, float ULprecision, bool UseAdaptiveSampling = true, bool CreateConfidenceBelt = true);

private:

  void init( UInt_t seed ); //  to be called by constructor

  // methods
  Double_t GetRandom( std::string pdf, std::string var );
  Long64_t LowBoundarySearch(std::vector<Double_t> * cdf, Double_t value);
  Long64_t HighBoundarySearch(std::vector<Double_t> * cdf, Double_t value);
  MCMCInterval * GetMcmcInterval(double conf_level,
				 int n_iter,
				 int n_burn,
				 double left_side_tail_fraction,
				 int n_bins);
  void makeMcmcPosteriorPlot( std::string filename );
  double printMcmcUpperLimit( std::string filename = "" );

  Double_t RoundUpperBound(Double_t bound);

  // data members
  RooWorkspace * ws;
  RooStats::ModelConfig SbModel;
  RooStats::ModelConfig BModel;
  RooAbsData * data;
  BayesianCalculator * bcalc;
  RooStats::SimpleInterval * sInt;
  double nsig_rel_err;
  double nbkg_rel_err;
  Int_t _nuisance_model;

  // attributes
  bool hasSigErr;
  bool hasBgErr;

  // for Bayesian MCMC calculation
  MCMCInterval * mcInt;
  
  // for Feldman-Cousins Calculator
  FeldmanCousins * fcCalc;

  // random numbers
  TRandom3 r;

  // expected limits
  Double_t _expected_limit;
  Double_t _low68;
  Double_t _high68;
  Double_t _low95;
  Double_t _high95;
  
};



// CLs limit calculator
std::vector<Double_t>
GetClsLimits(RooWorkspace * pWs,
	     const char * modelSBName = "SbModel",
	     const char * modelBName = "BModel",
	     const char * dataName = "observed_data",                  
	     int calculatorType = 0, // calculator type 
	     int testStatType = 3, // test stat type
	     bool useCls = true,
	     int npoints = 10,
	     double poimin = 1,  // use default is poimin >= poimax
	     double poimax = 0,
	     int ntoys=10000,
	     std::string suffix = "test");



// default constructor
CL95Calc::CL95Calc(){
  init(0);
}


CL95Calc::CL95Calc(UInt_t seed){
  init(seed);
}


void CL95Calc::init(UInt_t seed){
  ws = new RooWorkspace("ws");
  data = 0;

  sInt = 0;
  bcalc = 0;
  mcInt = 0;
  fcCalc = 0;
  SbModel.SetName("SbModel");
  SbModel.SetTitle("ModelConfig for roostats_cl95");

  nsig_rel_err = -1.0; // default non-initialized value
  nbkg_rel_err = -1.0; // default non-initialized value

  // set random seed
  if (seed == 0){
    r.SetSeed();
    UInt_t _seed = r.GetSeed();
    UInt_t _pid = gSystem->GetPid();
    std::cout << "[CL95Calc]: random seed: " << _seed << std::endl;
    std::cout << "[CL95Calc]: process ID: " << _pid << std::endl;
    _seed = 31*_seed+_pid;
    std::cout << "[CL95Calc]: new random seed (31*seed+pid): " << _seed << std::endl;
    r.SetSeed(_seed);
    
    // set RooFit random seed (it has a private copy)
    RooRandom::randomGenerator()->SetSeed(_seed);
  }
  else{
    std::cout << "[CL95Calc]: random seed: " << seed << std::endl;
    r.SetSeed(seed);
    
    // set RooFit random seed (it has a private copy)
    RooRandom::randomGenerator()->SetSeed(seed);
  }

  // default Gaussian nuisance model
  _nuisance_model = 0;

  // set default attributes
  hasSigErr = false;
  hasBgErr = false;
}


CL95Calc::~CL95Calc(){
  delete ws;
  delete data;
  delete sInt;
  delete bcalc;
  delete mcInt;
  delete fcCalc;
}


RooWorkspace * CL95Calc::makeWorkspace(Double_t ilum, Double_t slum,
				       Double_t eff, Double_t seff,
				       Double_t bck, Double_t sbck,
				       Bool_t gauss,
				       Int_t nuisanceModel){

  if ( bck>0.0 && (sbck/bck)<5.0 ){
    // check that bck is not too close to zero,
    // so lognormal and gamma modls still make sense
    std::cout << "[CL95Calc]: checking background expectation and its uncertainty - ok" << std::endl;
    _nuisance_model = nuisanceModel;
  }
  else{
    _nuisance_model = 0;
    std::cout << "[CL95Calc]: background expectation is too close to zero compared to its uncertainty" << std::endl;
    std::cout << "[CL95Calc]: switching to the Gaussian nuisance model" << std::endl;

    // FIXME: is this appropriate fix for 0 bg expectation?
    if (bck<0.001){
      bck = std::max(bck,sbck/1000.0);
    }
  }

  // Workspace
  // RooWorkspace * ws = new RooWorkspace("ws",true);
  
  // observable: number of events
  ws->factory( "n[0]" );

  // integrated luminosity
  ws->factory( "lumi[0]" );

  // cross section - parameter of interest
  ws->factory( "xsec[0]" );

  // selection efficiency * acceptance
  ws->factory( "efficiency[0]" );

  // nuisance parameter: factor 1 with combined relative uncertainty
  ws->factory( "nsig_nuis[1.0]" ); // will adjust range below

  // signal yield
  ws->factory( "prod::nsig(lumi,xsec,efficiency, nsig_nuis)" );

  // estimated background yield
  ws->factory( "bkg_est[1.0]" );
  ws->factory( "lbkg_est[0]" ); // for special case of lognormal prior

  // nuisance parameter: factor 1 with background relative uncertainty
  //ws->factory( "nbkg_nuis[1.0]" ); // will adjust range below

  // background yield
  ws->factory( "nbkg[1.0]" ); // will adjust value and range below

  // core model:
  ws->factory("sum::yield(nsig,nbkg)");
  if (gauss){
    // Poisson probability with mean signal+bkg
    std::cout << "[CL95Calc]: creating Gaussian probability as core model..." << std::endl;
    ws->factory( "Gaussian::model_core(n,yield,expr('sqrt(yield)',yield))" );
  }
  else{
    // Poisson probability with mean signal+bkg
    std::cout << "[CL95Calc]: creating Poisson probability as core model..." << std::endl;
    ws->factory( "Poisson::model_core(n,yield)" );
  }


  // systematic uncertainties
  nsig_rel_err = sqrt(slum*slum/ilum/ilum+seff*seff/eff/eff);
  nbkg_rel_err = sbck/bck;
  if (nsig_rel_err > 1.0e-10) hasSigErr = true;
  if (nbkg_rel_err > 1.0e-10) hasBgErr = true;

  if (_nuisance_model == 0){ // gaussian model for nuisance parameters

    std::cout << "[roostats_cl95]: Gaussian PDFs for nuisance parameters" << endl;

    // cumulative signal uncertainty
    ws->factory( "nsig_sigma[0.1]" );
    ws->factory( "nsig_global[1.0,0.1,10.0]" ); // mean of the nsig nuisance par
    if (hasSigErr){
      // non-zero overall signal sensitivity systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero systematics on overall signal sensitivity, creating constraint term" << endl;
      ws->factory( "Gaussian::syst_nsig(nsig_nuis, nsig_global, nsig_sigma)" );
    }
    // background uncertainty
    ws->factory( "nbkg_sigma[0.1]" );
    if (hasBgErr){
      // non-zero background systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero background systematics, creating constraint term" << endl;
      ws->factory( "Gaussian::syst_nbkg(nbkg, bkg_est, nbkg_sigma)" );
    }

    ws->var("nsig_sigma")->setVal(nsig_rel_err);
    ws->var("nbkg_sigma")->setVal(sbck);
    ws->var("nsig_global")->setConstant(kTRUE);
    ws->var("nsig_sigma")->setConstant(kTRUE);
    ws->var("nbkg_sigma")->setConstant(kTRUE);
  }
  else if (_nuisance_model == 1){// Lognormal model for nuisance parameters
    // this is the "old" implementation of the lognormal model, better use
    // the new one, nuisance_model=3

    std::cout << "[roostats_cl95]: Lognormal PDFs for nuisance parameters" << endl;

    // cumulative signal uncertainty
    ws->factory( "nsig_kappa[1.1]" );
    ws->factory( "nsig_global[1.0,0.1,10.0]" ); // mean of the nsig nuisance par

    if (hasSigErr){
      // non-zero overall signal sensitivity systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero systematics on overall signal sensitivity, creating constraint term" << endl;
      ws->factory( "Lognormal::syst_nsig(nsig_nuis, nsig_global, nsig_kappa)" );
    }

    // background uncertainty
    ws->factory( "nbkg_kappa[1.1]" );
    if (hasBgErr){
      // non-zero background systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero background systematics, creating constraint term" << endl;
      ws->factory( "Lognormal::syst_nbkg(nbkg, bkg_est, nbkg_kappa)" );
    }

    ws->var("nsig_kappa")->setVal(1.0 + nsig_rel_err);
    ws->var("nbkg_kappa")->setVal(1.0 + nbkg_rel_err);
    ws->var("nsig_global")->setConstant(kTRUE);
    ws->var("nsig_kappa")->setConstant(kTRUE);
    ws->var("nbkg_kappa")->setConstant(kTRUE);
  }
  else if (_nuisance_model == 3){
    //
    // Lognormal nuisance model implemented as Gaussian of
    // a log of the parameter. The corresponding global observable
    // is the log of the estimate for the parameter.
    //

    std::cout << "[roostats_cl95]: Lognormal PDFs for nuisance parameters" << endl;

    // cumulative signal uncertainty
    ws->factory( "lnsig_sigma[0.1]" );
    ws->factory( "nsig_global[0.0,-0.5,0.5]" ); // log of mean of the nsig nuisance par
    //ws->factory( "Gaussian::syst_nsig(cexpr::lnsig('log(nsig_nuis)', nsig_nuis), nsig_global, lnsig_sigma)" );
    if (hasSigErr){
      // non-zero overall signal sensitivity systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero systematics on overall signal sensitivity, creating constraint term" << endl;
      ws->factory( "Gaussian::syst_nsig(cexpr::lnsig('log(nsig_nuis)', nsig_nuis), nsig_global, lnsig_sigma)" );
    }

    // background uncertainty
    ws->factory( "lnbkg_sigma[0.1]" );
    if (hasBgErr){
      // non-zero background systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero background systematics, creating constraint term" << endl;
      ws->factory( "Gaussian::syst_nbkg(cexpr::lnbkg('log(nbkg)',nbkg), lbkg_est, lnbkg_sigma)" );
    }

    ws->var("lnsig_sigma")->setVal(nsig_rel_err);
    ws->var("lnbkg_sigma")->setVal(nbkg_rel_err);
    ws->var("nsig_global")->setConstant(kTRUE);
    ws->var("lnsig_sigma")->setConstant(kTRUE);
    ws->var("lnbkg_sigma")->setConstant(kTRUE);
  }
  else if (_nuisance_model == 2){ // Gamma model for nuisance parameters

    std::cout << "[roostats_cl95]: Gamma PDFs for nuisance parameters" << endl;

    // cumulative signal uncertainty
    ws->factory( "nsig_global[1.0,0.1,10.0]" ); // mean of the nsig nuisance par
    ws->factory( "nsig_rel_err[0.1, 0.0, 1.0]" );
    ws->factory( "expr::nsig_beta('nsig_rel_err*nsig_rel_err/nsig_global',nsig_rel_err,nsig_global)" );
    ws->factory( "expr::nsig_gamma('nsig_global*nsig_global/nsig_rel_err/nsig_rel_err+1.0',nsig_global,nsig_rel_err)" );
    ws->var("nsig_rel_err") ->setVal(nsig_rel_err);
    if (hasSigErr){
      // non-zero overall signal sensitivity systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero systematics on overall signal sensitivity, creating constraint term" << endl;
      ws->factory( "Gamma::syst_nsig(nsig_nuis, nsig_gamma, nsig_beta, 0.0)" );
    }

    // background uncertainty
    //ws->factory( "nbkg_global[1.0]" ); // mean of the nbkg nuisance par
    ws->factory( "nbkg_rel_err[0.1, 0.0, 1.0]" );
    ws->factory( "expr::nbkg_beta('nbkg_rel_err*nbkg_rel_err/bkg_est',nbkg_rel_err,bkg_est)" );
    ws->factory( "expr::nbkg_gamma('bkg_est*bkg_est/nbkg_rel_err/nbkg_rel_err+1.0',bkg_est,nbkg_rel_err)" );
    //ws->var("nbkg_global") ->setVal( bck );
    ws->var("nbkg_rel_err")->setVal(nbkg_rel_err);
    if (hasBgErr){
      // non-zero background systematics: need to create
      // the corresponding constraint term for the likelihood
      std::cout << "[roostats_cl95]: non-zero background systematics, creating constraint term" << endl;
      ws->factory( "Gamma::syst_nbkg(nbkg, nbkg_gamma, nbkg_beta, 0.0)" );
    }

    ws->var("nsig_rel_err")->setConstant(kTRUE);
    ws->var("nsig_global")->setConstant(kTRUE);
    ws->var("nbkg_rel_err")->setConstant(kTRUE);
    //ws->var("nbkg_global")->setConstant(kTRUE);

  }
  else{
    std::cout <<"[roostats_cl95]: undefined nuisance parameter model specified, exiting" << std::endl;
  }

  // model with systematics
  if (hasSigErr && hasBgErr){
    std::cout << "[roostats_cl95]: factoring in signal sensitivity and background rate systematics constraint terms" << endl;
    ws->factory( "PROD::model(model_core, syst_nsig, syst_nbkg)" );
    ws->var("nsig_nuis") ->setConstant(kFALSE); // nuisance
    ws->var("nbkg")      ->setConstant(kFALSE); // nuisance
    ws->factory( "PROD::nuis_prior(syst_nsig,syst_nbkg)" );  
  }
  else if (hasSigErr && !hasBgErr){
    std::cout << "[roostats_cl95]: factoring in signal sensitivity systematics constraint term" << endl;
    ws->factory( "PROD::model(model_core, syst_nsig)" );
    ws->var("nsig_nuis") ->setConstant(kFALSE); // nuisance
    ws->var("nbkg")      ->setConstant(kTRUE); // nuisance
    ws->factory( "PROD::nuis_prior(syst_nsig)" );  
  }
  else if (!hasSigErr && hasBgErr){
    std::cout << "[roostats_cl95]: factoring in background rate systematics constraint term" << endl;
    ws->factory( "PROD::model(model_core, syst_nbkg)" );
    ws->var("nsig_nuis") ->setConstant(kTRUE); // nuisance
    ws->var("nbkg")      ->setConstant(kFALSE); // nuisance
    ws->factory( "PROD::nuis_prior(syst_nbkg)" );  
  }
  else{
    ws->factory( "PROD::model(model_core)" );
    ws->var("nsig_nuis") ->setConstant(kTRUE); // nuisance
    ws->var("nbkg")      ->setConstant(kTRUE); // nuisance
  }

  // flat prior for the parameter of interest
  ws->factory( "Uniform::prior(xsec)" );  

  // parameter values
  ws->var("lumi")      ->setVal(ilum);
  ws->var("efficiency")->setVal(eff);
  ws->var("bkg_est")   ->setVal(bck);
  ws->var("lbkg_est")   ->setVal(TMath::Log(bck));
  ws->var("xsec")      ->setVal(0.0);
  ws->var("nsig_nuis") ->setVal(1.0);
  ws->var("nbkg")      ->setVal(bck);

  // set some parameters as constants
  ws->var("lumi")      ->setConstant(kTRUE);
  ws->var("efficiency")->setConstant(kTRUE);
  ws->var("bkg_est")   ->setConstant(kTRUE);
  ws->var("lbkg_est")   ->setConstant(kTRUE);
  ws->var("n")         ->setConstant(kFALSE); // observable
  ws->var("xsec")      ->setConstant(kFALSE); // parameter of interest
  //ws->var("nsig_nuis") ->setConstant(kFALSE); // nuisance
  //ws->var("nbkg")      ->setConstant(kFALSE); // nuisance

  // floating parameters ranges
  // crude estimates! Need to know data to do better
  ws->var("n")        ->setRange( 0.0, bck+(5.0*sbck)+10.0); // ad-hoc range for obs
  ws->var("xsec")     ->setRange( 0.0, 15.0*(1.0+nsig_rel_err)/ilum/eff ); // ad-hoc range for POI
  ws->var("nsig_nuis")->setRange( std::max(0.0, 1.0 - 5.0*nsig_rel_err), 1.0 + 5.0*nsig_rel_err);
  ws->var("nbkg")     ->setRange( std::max(0.0, bck - 5.0*sbck), bck + 5.0*sbck);
  ws->var("bkg_est")  ->setRange( std::max(0.0, bck - 5.0*sbck), bck + 5.0*sbck);
  // FIXME: check for zeros in the log
  ws->var("lbkg_est")  ->setRange( TMath::Log(ws->var("bkg_est")->getMin()), TMath::Log(ws->var("bkg_est")->getMin()));
  
  // Definition of observables and parameters of interest

  // observables
  RooArgSet obs(*ws->var("n"), "obs");

  // global observables
  //RooArgSet globalObs(*ws->var("nsig_global"), *ws->var("bkg_est"), "global_obs");
  //RooArgSet globalObs(*ws->var("nsig_global"), "global_obs");
  RooArgSet globalObs("global_obs");
  if (hasSigErr) globalObs.add( *ws->var("nsig_global") );
  if (hasBgErr){
    if (_nuisance_model == 3){
      globalObs.add( *ws->var("lbkg_est") );
    }
    else{
      globalObs.add( *ws->var("bkg_est") );
    }
  }

  // parameters of interest
  RooArgSet poi(*ws->var("xsec"), "poi");

  // nuisance parameters
  //RooArgSet nuis(*ws->var("nsig_nuis"), *ws->var("nbkg"), "nuis");
  RooArgSet nuis("nuis");
  if (hasSigErr) nuis.add( *ws->var("nsig_nuis") );
  if (hasBgErr) nuis.add( *ws->var("nbkg") );

  // setup the S+B model
  SbModel.SetWorkspace(*ws);
  SbModel.SetPdf(*(ws->pdf("model")));
  SbModel.SetParametersOfInterest(poi);
  SbModel.SetPriorPdf(*(ws->pdf("prior")));
  SbModel.SetNuisanceParameters(nuis);
  SbModel.SetObservables(obs);
  SbModel.SetGlobalObservables(globalObs);

  // will import the model config once the snapshot is saved

  // background-only model
  // use the same PDF as s+b, with xsec=0
  // (poi zero value will be set in the snapshot)
  //BModel = *(RooStats::ModelConfig *)ws->obj("SbModel");
  BModel = SbModel;
  BModel.SetName("BModel");
  BModel.SetWorkspace(*ws);

  // We also need to set up parameter snapshots for the models
  // but we need data for that, so it is done in makeData()

  return ws;
}


RooAbsData * CL95Calc::makeData( Int_t n ){
  //
  // make the dataset owned by the class
  // the current one is deleted
  //
  // set ranges as well
  //
  
  // make RooFit quiet
  // cash the current message level first
  RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

  // floating parameters ranges
  if (nsig_rel_err < 0.0 || nbkg_rel_err < 0.0){
    std::cout << "[roostats_cl95]: Workspace not initialized, cannot create a dataset" << std::endl;
    return 0;
  }
  
  double ilum = ws->var("lumi")->getVal();
  double eff  = ws->var("efficiency")->getVal();
  double bck  = ws->var("bkg_est")->getVal();
  double sbck = nbkg_rel_err*bck;

  ws->var("n")        ->setRange( 0.0, bck+(5.0*sbck)+10.0*(n+1.0)); // ad-hoc range for obs
  Double_t xsec_upper_bound = 4.0*(std::max(3.0,n-bck)+sqrt(n)+sbck)/ilum/eff;  // ad-hoc range for POI
  xsec_upper_bound = RoundUpperBound(xsec_upper_bound);
  ws->var("xsec")     ->setRange( 0.0, xsec_upper_bound );
  ws->var("nsig_nuis")->setRange( std::max(0.0, 1.0 - 5.0*nsig_rel_err), 1.0 + 5.0*nsig_rel_err);
  ws->var("nbkg")     ->setRange( std::max(0.0, bck - 5.0*sbck), bck + 5.0*sbck);

  // create data
  ws->var("n")         ->setVal(n);
  delete data;
  data = new RooDataSet("data","",*(SbModel.GetObservables()));
  data->add( *(SbModel.GetObservables()));

  
  // Now set up parameter snapshots for the S+B and B models

  // find global maximum with the signal+background model
  // with conditional MLEs for nuisance parameters
  // and save the parameter point snapshot in the Workspace
  //  - safer to keep a default name because some RooStats calculators
  //    will anticipate it
  RooAbsReal * pNll = SbModel.GetPdf()->createNLL(*data);
  RooAbsReal * pProfile = pNll->createProfile(RooArgSet());
  pProfile->getVal(); // this will do fit and set POI and nuisance parameters to fitted values
  RooArgSet * pPoiAndNuisance = new RooArgSet("poiAndNuisance");
  if(SbModel.GetNuisanceParameters())
    pPoiAndNuisance->add(*SbModel.GetNuisanceParameters());
  pPoiAndNuisance->add(*SbModel.GetParametersOfInterest());
  std::cout << "\nWill save these parameter points that correspond to the fit to data" << std::endl;
  pPoiAndNuisance->Print("v");
  SbModel.SetSnapshot(*pPoiAndNuisance);
  delete pProfile;
  delete pNll;
  delete pPoiAndNuisance;

  // Find a parameter point for generating pseudo-data
  // with the background-only data.
  // Save the parameter point snapshot in the Workspace
  //
  // POI value under the background hypothesis
  Double_t poiValueForBModel = 0.0;
  pNll = BModel.GetPdf()->createNLL(*data);
  const RooArgSet * poi = BModel.GetParametersOfInterest();
  pProfile = pNll->createProfile(*poi);
  ((RooRealVar *)poi->first())->setVal(poiValueForBModel);
  pProfile->getVal(); // this will do fit and set nuisance parameters to profiled values
  pPoiAndNuisance = new RooArgSet("poiAndNuisance");
  if(BModel.GetNuisanceParameters())
    pPoiAndNuisance->add(*BModel.GetNuisanceParameters());
  pPoiAndNuisance->add(*BModel.GetParametersOfInterest());
  std::cout << "\nShould use these parameter points to generate pseudo data for bkg only" << std::endl;
  pPoiAndNuisance->Print("v");
  BModel.SetSnapshot(*pPoiAndNuisance);
  delete pProfile;
  delete pNll;
  delete pPoiAndNuisance;

  // import the model configs, has to be after all snapshots are saved
  ws->import(SbModel);
  ws->import(BModel);

  // restore RooFit messaging level
  RooMsgService::instance().setGlobalKillBelow(msglevel);

  return data;
}


MCMCInterval * CL95Calc::GetMcmcInterval(double conf_level,
					int n_iter,
					int n_burn,
					double left_side_tail_fraction,
					int n_bins){
  // use MCMCCalculator  (takes about 1 min)
  // Want an efficient proposal function, so derive it from covariance
  // matrix of fit
  
  RooFitResult * fit = ws->pdf("model")->fitTo(*data,Save(),
					       Verbose(kFALSE),
					       PrintLevel(-1),
					       Warnings(0),
					       PrintEvalErrors(-1));
  ProposalHelper ph;
  ph.SetVariables((RooArgSet&)fit->floatParsFinal());
  ph.SetCovMatrix(fit->covarianceMatrix());
  ph.SetUpdateProposalParameters(kTRUE); // auto-create mean vars and add mappings
  ph.SetCacheSize(100);
  ProposalFunction* pf = ph.GetProposalFunction();
  
  MCMCCalculator mcmc( *data, SbModel );
  mcmc.SetConfidenceLevel(conf_level);
  mcmc.SetNumIters(n_iter);          // Metropolis-Hastings algorithm iterations
  mcmc.SetProposalFunction(*pf);
  mcmc.SetNumBurnInSteps(n_burn); // first N steps to be ignored as burn-in
  mcmc.SetLeftSideTailFraction(left_side_tail_fraction);
  mcmc.SetNumBins(n_bins);
  
  delete mcInt;
  mcInt = mcmc.GetInterval();

  return mcInt;
}


void CL95Calc::makeMcmcPosteriorPlot( std::string filename ){
  
  TCanvas c1("c1");
  MCMCIntervalPlot plot(*mcInt);
  plot.Draw();
  c1.SaveAs(filename.c_str());
  
  return;
}


double CL95Calc::printMcmcUpperLimit( std::string filename ){
  //
  // print out the upper limit on the first Parameter of Interest
  //

  RooRealVar * firstPOI = (RooRealVar*) SbModel.GetParametersOfInterest()->first();
  double _limit = mcInt->UpperLimit(*firstPOI);
  cout << "\n95% upper limit on " <<firstPOI->GetName()<<" is : "<<
    _limit <<endl;

  if (filename.size()!=0){
    
    std::ofstream aFile;

    // append to file if exists
    aFile.open(filename.c_str(), std::ios_base::app);

    char buf[1024];
    sprintf(buf, "%7.6f", _limit);

    aFile << buf << std::endl;

    // close outfile here so it is safe even if subsequent iterations crash
    aFile.close();

  }

  return _limit;
}



Double_t CL95Calc::FC_calc(int Nbins, float conf_int, float ULprecision, bool UseAdaptiveSampling, bool CreateConfidenceBelt){


  Double_t upper_limit = 0;
  int cnt = 0;
  bool verbose = true; //Set to true to see the output of each FC step

  std::cout << "[roostats_cl95]: FC calculation is still experimental in this context!!!" << std::endl;
      
  std::cout << "[roostats_cl95]: Range of allowed cross section values: [" 
	    << ws->var("xsec")->getMin() << ", " 
	    << ws->var("xsec")->getMax() << "]" << std::endl;


  //prepare Feldman-Cousins Calulator

  delete fcCalc;
  fcCalc = new FeldmanCousins(*data,SbModel);
      
  fcCalc->SetConfidenceLevel(conf_int); // confidence interval
  //fcCalc->AdditionalNToysFactor(0.1); // to speed up the result 
  fcCalc->UseAdaptiveSampling(UseAdaptiveSampling); // speed it up a bit
  fcCalc->SetNBins(Nbins); // set how many points per parameter of interest to scan
  fcCalc->CreateConfBelt(CreateConfidenceBelt); // save the information in the belt for plotting

      
  if(!SbModel.GetPdf()->canBeExtended()){
    if(data->numEntries()==1)     
      fcCalc->FluctuateNumDataEntries(false);
    else
      cout <<"Not sure what to do about this model" <<endl;
  }

  RooRealVar* firstPOI = (RooRealVar*) SbModel.GetParametersOfInterest()->first();
  
  double max = firstPOI->getMax();
  double min = firstPOI->getMin();
  double med = (max + min)/2.0;
      
  double maxPerm = firstPOI->getMax();
  double minPerm = firstPOI->getMin();
    
  double UpperLimit = 0;
  
  PointSetInterval* interval = 0;

  while ( 1 ){
    
    ++cnt;
    firstPOI->setMax( max );
    firstPOI->setMin( min );
    
    if ( verbose ) std::cout << "[FeldmanCousins]: Setting max/min/med to = " << max << " / " << min << " / " << med <<  std::endl;
	
    interval = fcCalc->GetInterval();
    interval -> Delete();
	
    UpperLimit = interval -> UpperLimit(*firstPOI);
    if ( verbose ) std::cout <<"[FeldmanCousins]: Updating Upper Limt to = "<< UpperLimit << std::endl;

    if ( UpperLimit > 0.000001 ){

      min = med;
      med = (max + min)/2.0;
      
    }
    else{
      
      max = med;
      med = (max + min)/2.0;
      
    }
    
    if (  ( UpperLimit > 0.000001 ) && ( (max - min) < ULprecision)  ) {
      upper_limit = UpperLimit;
      std::cout <<"[FeldmanCousins]: In "<< cnt << " steps Upper Limt converged to " << upper_limit << std::endl;
      break; 
    }
    
    if ( cnt > 50 ) {
      upper_limit = -1;
      std::cout << std::endl;
      std::cout <<"[FeldmanCousins     WARNING!!!!!!!!!!!!       ]: Calculator could not converge in under 50 steps. Returning Upper Limit of -1." << std::endl;
      std::cout << std::endl;
      break;
    }

  }
      
  ws->var("xsec")->setMax( maxPerm );
  ws->var("xsec")->setMin( minPerm );

  return upper_limit;

}





Double_t CL95Calc::cl95( std::string method, LimitResult * result ){
  //
  // Compute the observed limit
  // For some methods - CLs - compute the expected limts too.
  // Extended results are returned via reference as LimitResul object
  //
  // this method assumes that the workspace,
  // data and model config are ready
  //

  Double_t upper_limit = -1.0;

  // make RooFit quiet
  // cash the current message level first
  RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
  // get ugly RooFit print out of the way
  // FIXME: uncomment
  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

  Int_t _attempt = 0; // allow several attempts for limit calculation, stop after that
  while(1){

    ++_attempt;
    
    // too many attempts
    if (_attempt > 5){
      std::cout << "[roostats_cl95]: limit calculation did not converge, exiting..." << std::endl;
      return -1.0;
    }

    if (method.find("bayesian") != std::string::npos){
      
      std::cout << "[roostats_cl95]: Range of allowed cross section values: [" 
		<< ws->var("xsec")->getMin() << ", " 
		<< ws->var("xsec")->getMax() << "]" << std::endl;

      //prepare Bayesian Calulator
      delete bcalc;
      bcalc = new BayesianCalculator(*data, SbModel);
      TString namestring = "mybc";
      bcalc->SetName(namestring);
      bcalc->SetConfidenceLevel(0.95);
      bcalc->SetLeftSideTailFraction(0.0);
      //bcalc->SetIntegrationType("ROOFIT");
      
      delete sInt;
      sInt = bcalc->GetInterval();
      upper_limit = sInt->UpperLimit();
      delete sInt;
      sInt = 0;
      
    }
    else if (method.find("mcmc") != std::string::npos){
      
      std::cout << "[roostats_cl95]: Bayesian MCMC calculation is still experimental in this context!!!" << std::endl;
      
      std::cout << "[roostats_cl95]: Range of allowed cross section values: [" 
		<< ws->var("xsec")->getMin() << ", " 
		<< ws->var("xsec")->getMax() << "]" << std::endl;

      //prepare Bayesian Markov Chain MC Calulator
      mcInt = GetMcmcInterval(0.95, 50000, 100, 0.0, 40);
      upper_limit = printMcmcUpperLimit();
    }
    else if (method.find("cls") != std::string::npos){
      //
      // testing CLs
      //
      
      std::cout << "[roostats_cl95]: CLs calculation is still experimental in this context!!!" << std::endl;
      
      std::cout << "[roostats_cl95]: Range of allowed cross section values: [" 
		<< ws->var("xsec")->getMin() << ", " 
		<< ws->var("xsec")->getMax() << "]" << std::endl;

      // timer
      TStopwatch t;
      t.Start();

      // load parameter point with the best fit to data
      SbModel.LoadSnapshot();
      RooRealVar * pPoi = (RooRealVar *)(SbModel.GetParametersOfInterest()->first());
      // get POI upper error from the fit
      Double_t poi_err = pPoi->getErrorHi();
      // get POI upper range boundary
      Double_t poi_upper_range = pPoi->getMax();
      // get the upper range boundary for CLs as min of poi range and 5*error
      Double_t upper_range = std::min(10.0*poi_err,poi_upper_range);
      // debug output
      //std::cout << "range, error, new range " << poi_upper_range << ", "<< poi_err << ", " << upper_range << std::endl;

      RooMsgService::instance().setGlobalKillBelow(RooFit::PROGRESS);

      std::vector<Double_t> lim = 
	GetClsLimits( ws,
		      "SbModel",
		      "BModel",
		      "observed_data",
		      0, // calculator type, 0-freq, 1-hybrid
		      3, // test statistic, 0-lep, 1-tevatron, 2-PL, 3-PL 1-sided
		      true, // useCls
		      10, // npoints in the scan
		      0, // poimin: use default is poimin >= poimax
		      upper_range,
		      10000,// ntoys
		      "test" );
      
      t.Stop();
      t.Print();

      if (result){
	result->_observed_limit = lim[0];
	result->_observed_limit_error = lim[1];
	result->_expected_limit = lim[2];
	result->_low68  = lim[3];
	result->_high68 = lim[4];
	result->_low95  = lim[5];
	result->_high95 = lim[6];
	result->_cover68 = -1.0;
	result->_cover95 = -1.0;
      }

      upper_limit = lim[0];

    } // end of the CLs block
    else if (method.find("fc") != std::string::npos){

      int Nbins = 1;
      float conf_int = 0.95;
      float ULprecision = 0.1;
      bool UseAdaptiveSampling = true;
      bool CreateConfidenceBelt = true;
      
      
      upper_limit = FC_calc(Nbins, conf_int, ULprecision, UseAdaptiveSampling, CreateConfidenceBelt);
	
    } // end of the FC block
    else{

      std::cout << "[roostats_cl95]: method " << method 
		<< " is not implemented, exiting" <<std::endl;
      return -1.0;

    } // end of choice of method block

    
    // adaptive range in case the POI range was not guessed properly
    Double_t _poi_max_range = ws->var("xsec")->getMax();

    if (method.find("cls")!=std::string::npos) break;
    if (method.find("fc") != std::string::npos ) break;
    // range too wide
    else if (upper_limit < _poi_max_range/10.0){
      std::cout << "[roostats_cl95]: POI range is too wide, will narrow the range and rerun" << std::endl;
      ws->var("xsec")->setMax(RoundUpperBound(_poi_max_range/2.0));
    }
    // range too narrow
    else if (upper_limit > _poi_max_range/2.0){
      std::cout << "[roostats_cl95]: upper limit is too narrow, will widen the range and rerun" << std::endl;
      ws->var("xsec")->setMax(RoundUpperBound(2.0*_poi_max_range));
    }
    // all good, limit is ready
    else{
      break;
    }
    
  } // end of while(1) loop
  
  // restore RooFit messaging level
  RooMsgService::instance().setGlobalKillBelow(msglevel);

  return upper_limit;
  
}


Double_t CL95Calc::cla( Double_t ilum, Double_t slum,
			Double_t eff, Double_t seff,
			Double_t bck, Double_t sbck,
			Int_t nuisanceModel,
			std::string method ){

  makeWorkspace( ilum, slum,
		 eff, seff,
		 bck, sbck,
		 kFALSE,
		 nuisanceModel );
  
  Double_t CL95A = 0, precision = 1.e-4;

  Int_t i;
  for (i = bck; i >= 0; i--)
    {
      makeData( i );

      Double_t s95 = cl95( method );
      Double_t s95w =s95*TMath::Poisson( (Double_t)i, bck );
      CL95A += s95w;
      cout << "[roostats_cla]: n = " << i << "; 95% C.L. = " << s95 << " pb; weighted 95% C.L. = " << s95w << " pb; running <s95> = " << CL95A << " pb" << endl;

      if (s95w < CL95A*precision) break;
    }
  cout << "[roostats_cla]: Lower bound on n has been found at " << i+1 << endl;

  for (i = bck+1; ; i++)
    {
      makeData( i );
      Double_t s95 = cl95( method );
      Double_t s95w =s95*TMath::Poisson( (Double_t)i, bck );
      CL95A += s95w;
      cout << "[roostats_cla]: n = " << i << "; 95% C.L. = " << s95 << " pb; weighted 95% C.L. = " << s95w << " pb; running <s95> = " << CL95A << " pb" << endl;

      if (s95w < CL95A*precision) break;
    }
  cout << "[roostats_cla]: Upper bound on n has been found at " << i << endl;
  cout << "[roostats_cla]: Average upper 95% C.L. limit = " << CL95A << " pb" << endl;

  return CL95A;
}



LimitResult CL95Calc::clm( Double_t ilum, Double_t slum,
			   Double_t eff, Double_t seff,
			   Double_t bck, Double_t sbck,
			   Int_t nit, Int_t nuisanceModel,
			   std::string method ){
  
  makeWorkspace( ilum, slum,
		 eff, seff,
		 bck, sbck,
		 kFALSE,
		 nuisanceModel );
  
  Double_t CLM = 0.0;
  LimitResult _result;

  Double_t b68[2] = {0.0, 0.0}; // 1-sigma expected band
  Double_t b95[2] = {0.0, 0.0}; // 2-sigma expected band

  std::vector<Double_t> pe;

  // timer
  TStopwatch t;
  t.Start(); // start timer
  Double_t _realtime = 0.0;
  Double_t _cputime = 0.0;
  Double_t _realtime_last = 0.0;
  Double_t _cputime_last = 0.0;
  Double_t _realtime_average = 0.0;
  Double_t _cputime_average = 0.0;

  // throw pseudoexperiments
  if (nit <= 0)return _result;
  std::map<Int_t,Double_t> cached_limit;
  for (Int_t i = 0; i < nit; i++)
    {
      // throw random nuisance parameter (bkg yield)
      Double_t bmean = GetRandom("syst_nbkg", "nbkg");

      std::cout << "[roostats_clm]: generatin pseudo-data with bmean = " << bmean << std::endl;
      Int_t n = r.Poisson(bmean);

      // check if the limit for this n is already cached
      Double_t _pe = -1.0;
      if (cached_limit.find(n)==cached_limit.end()){
	
	makeData( n );
	std::cout << "[roostats_clm]: invoking CL95 with n = " << n << std::endl;
	
	_pe = cl95( method );
	cached_limit[n] = _pe;
      }
      else{
	std::cout << "[roostats_clm]: returning previously cached limit for n = " << n << std::endl;
	_pe = cached_limit[n];
      }

      pe.push_back(_pe);
      CLM += pe[i];

      _realtime_last = t.RealTime() - _realtime;
      _cputime_last  = t.CpuTime() - _cputime;
      _realtime = t.RealTime();
      _cputime = t.CpuTime();
      t.Continue();
      _realtime_average = _realtime/((Double_t)(i+1));
      _cputime_average  = _cputime/((Double_t)(i+1));

      std::cout << "n = " << n << "; 95% C.L. = " << _pe << " pb; running <s95> = " << CLM/(i+1.) << std::endl;
      std::cout << "Real time (s), this iteration: " << _realtime_last << ", average per iteration: " << _realtime_average << ", total: " << _realtime << std::endl;
      std::cout << "CPU time (s),  this iteration: " << _cputime_last << ", average per iteration: " << _cputime_average << ", total: " << _cputime << std::endl << std::endl;
    }

  CLM /= nit;

  // sort the vector with limits
  std::sort(pe.begin(), pe.end());

  // median for the expected limit
  Double_t _median = TMath::Median(nit, &pe[0]);

  // quantiles for the expected limit bands
  Double_t _prob[4]; // array with quantile boundaries
  _prob[0] = 0.021;
  _prob[1] = 0.159;
  _prob[2] = 0.841;
  _prob[3] = 0.979;

  Double_t _quantiles[4]; // array for the results

  TMath::Quantiles(nit, 4, &pe[0], _quantiles, _prob); // evaluate quantiles

  b68[0] = _quantiles[1];
  b68[1] = _quantiles[2];
  b95[0] = _quantiles[0];
  b95[1] = _quantiles[3]; 

  // let's get actual coverages now

  Long64_t lc68 = LowBoundarySearch(&pe, _quantiles[1]);
  Long64_t uc68 = HighBoundarySearch(&pe, _quantiles[2]);
  Long64_t lc95 = LowBoundarySearch(&pe, _quantiles[0]);
  Long64_t uc95 = HighBoundarySearch(&pe, _quantiles[3]);

  Double_t _cover68 = (nit - lc68 - uc68)*100./nit;
  Double_t _cover95 = (nit - lc95 - uc95)*100./nit;

  std::cout << "[CL95Calc::clm()]: median limit: " << _median << std::endl;
  std::cout << "[CL95Calc::clm()]: 1 sigma band: [" << b68[0] << "," << b68[1] << 
    "]; actual coverage: " << _cover68 << 
    "%; lower/upper percentile: " << lc68*100./nit <<"/" << uc68*100./nit << std::endl;
  std::cout << "[CL95Calc::clm()]: 2 sigma band: [" << b95[0] << "," << b95[1] << 
    "]; actual coverage: " << _cover95 << 
    "%; lower/upper percentile: " << lc95*100./nit <<"/" << uc95*100./nit << std::endl;

  t.Print();

  _result._expected_limit = _median;
  _result._low68  = b68[0];
  _result._high68 = b68[1];
  _result._low95  = b95[0];
  _result._high95 = b95[1];
  _result._cover68 = _cover68;
  _result._cover95 = _cover95;

  return _result;
}



int CL95Calc::makePlot( std::string method,
			std::string plotFileName ){

  if (method.find("bayesian") != std::string::npos){

    std::cout << "[roostats_cl95]: making Bayesian posterior plot" << endl;
  
    TCanvas c1("posterior");
    bcalc->SetScanOfPosterior(100);
    RooPlot * plot = bcalc->GetPosteriorPlot();
    plot->Draw();
    c1.SaveAs(plotFileName.c_str());
  }
  else if (method.find("mcmc") != std::string::npos){

    std::cout << "[roostats_cl95]: making Bayesian MCMC posterior plot" << endl;

    makeMcmcPosteriorPlot(plotFileName);
  
  }
  else{
    std::cout << "[roostats_cl95]: plot for method " << method 
	      << " is not implemented" <<std::endl;
    return -1;
  }

  return 0;
}



Double_t CL95Calc::GetRandom( std::string pdf, std::string var ){
  //
  // generates a random number using a pdf in the workspace
  //
  
  // generate a dataset with one entry
  RooDataSet * _ds = ws->pdf(pdf.c_str())->generate(*ws->var(var.c_str()), 1);

  Double_t _result = ((RooRealVar *)(_ds->get(0)->first()))->getVal();
  delete _ds;

  return _result;
}


Long64_t CL95Calc::LowBoundarySearch(std::vector<Double_t> * cdf, Double_t value){
  //
  // return number of elements which are < value with precision 1e-10
  //

  Long64_t result = 0;
  std::vector<Double_t>::const_iterator i = cdf->begin();
  while( (*i<value) && fabs(*i-value)>1.0e-10 && (i!=cdf->end()) ){
    ++i;
    ++result;
  }
  return result;
}


Long64_t CL95Calc::HighBoundarySearch(std::vector<Double_t> * cdf, Double_t value){
  //
  // return number of elements which are > value with precision 1e-10
  //

  Long64_t result = 0;
  std::vector<Double_t>::const_iterator i = cdf->end();
  while(1){ // (*i<value) && (i!=cdf->begin()) ){
    --i;
    if (*i>value && fabs(*i-value)>1.0e-10 ){
      ++result;
    }
    else break;
    if (i==cdf->begin()) break;
  }
  return result;
}



Double_t CL95Calc::RoundUpperBound(Double_t bound){
  //
  // find a round upper bound for a floating point
  //
  Double_t power = log10(bound);
  Int_t int_power = power>0.0 ? (Int_t)power : (Int_t)(power-1.0);
  Int_t int_bound = (Int_t)(bound/pow(10,(Double_t)int_power) * 10.0 + 1.0);
  bound = (Double_t)(int_bound/10.0*pow(10,(Double_t)int_power));
  return bound;
}



Int_t banner(){
  //#define __ROOFIT_NOBANNER // banner temporary off
#ifndef __EXOST_NOBANNER
//  std::cout << desc << std::endl;
#endif
  return 0 ;
}
static Int_t dummy_ = banner() ;



Double_t roostats_cl95(Double_t ilum, Double_t slum,
		       Double_t eff, Double_t seff,
		       Double_t bck, Double_t sbck,
		       Int_t n,
		       Bool_t gauss,
		       Int_t nuisanceModel,
		       std::string method,
		       std::string plotFileName,
		       UInt_t seed,
		       LimitResult * result){
  //
  // Global function to run the CL95 routine
  // 
  // If a non-null pointer to a LimitResult object is provided,
  // it will be filled, and the caller keeps the ownership of
  // the object. This is mainly an internal interface design solution,
  // users are not expected to use that (but they can of course)
  //

  std::cout << "[roostats_cl95]: estimating 95% C.L. upper limit" << endl;
  if (method.find("bayesian") != std::string::npos){
    std::cout << "[roostats_cl95]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("mcmc") != std::string::npos){
    std::cout << "[roostats_cl95]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("cls") != std::string::npos){
    std::cout << "[roostats_cl95]: using CLs calculation" << endl;
  }
  else if (method.find("fc") != std::string::npos){
    std::cout << "[roostats_cl95]: using Feldman-Cousins approach" << endl;
  }
  else if (method.find("workspace") != std::string::npos){
    std::cout << "[roostats_cl95]: no interval calculation, only create and save workspace" << endl;
  }
  else{
    std::cout << "[roostats_cl95]: method " << method 
	      << " is not implemented, exiting" <<std::endl;
    return -1.0;
  }

  // some input validation
  if (n < 0){
    std::cout << "Negative observed number of events specified, exiting" << std::endl;
    return -1.0;
  }

  if (n == 0) gauss = kFALSE;

  if (gauss){
    nuisanceModel = 0;
    std::cout << "[roostats_cl95]: Gaussian statistics used" << endl;
  }
  else{
    std::cout << "[roostats_cl95]: Poisson statistics used" << endl;
  }
    
  // limit calculation
  CL95Calc theCalc(seed);

  // container for computed limits
  LimitResult limitResult;

  RooWorkspace * ws = theCalc.makeWorkspace( ilum, slum,
					     eff, seff,
					     bck, sbck,
					     gauss,
					     nuisanceModel );

  RooDataSet * data = (RooDataSet *)( theCalc.makeData( n )->Clone() );
  data->SetName("observed_data");
  ws->import(*data);

  //ws->Print();

  ws->SaveAs("ws.root");

  // if only workspace requested, exit here
  if ( method.find("workspace") != std::string::npos ) return 0.0;

  Double_t limit = theCalc.cl95( method, &limitResult );
  std::cout << "[roostats_cl95]: 95% C.L. upper limit: " << limit << std::endl;

  // check if the plot is requested
  if (plotFileName.size() != 0){
    theCalc.makePlot(method, plotFileName);
  }

  if (result) *result = limitResult;

  return limit;
}



LimitResult roostats_limit(Double_t ilum, Double_t slum,
			   Double_t eff, Double_t seff,
			   Double_t bck, Double_t sbck,
			   Int_t n,
			   Bool_t gauss,
			   Int_t nuisanceModel,
			   std::string method,
			   std::string plotFileName,
			   UInt_t seed){
  //
  // Global function to run the CL95 routine
  // 

  LimitResult limitResult;

  roostats_cl95(ilum, slum,
		eff, seff,
		bck, sbck,
		n,
		gauss,
		nuisanceModel,
		method,
		plotFileName,
		seed,
		&limitResult);

  std::cout << " expected limit (median) " << limitResult.GetExpectedLimit() << std::endl;
  std::cout << " expected limit (-1 sig) " << limitResult.GetOneSigmaLowRange() << std::endl;
  std::cout << " expected limit (+1 sig) " << limitResult.GetOneSigmaHighRange() << std::endl;
  std::cout << " expected limit (-2 sig) " << limitResult.GetTwoSigmaLowRange() << std::endl;
  std::cout << " expected limit (+2 sig) " << limitResult.GetTwoSigmaHighRange() << std::endl;
  
  return limitResult;
}



Double_t roostats_cla(Double_t ilum, Double_t slum,
		      Double_t eff, Double_t seff,
		      Double_t bck, Double_t sbck,
		      Int_t nuisanceModel,
		      std::string method,
		      UInt_t seed){
  //
  // Global function to run old-style average limit routine.
  // Please use roostats_clm() instead.
  //

  Double_t limit = -1.0;

  std::cout << "[roostats_cla]: estimating average 95% C.L. upper limit" << endl;
  if (method.find("bayesian") != std::string::npos){
    std::cout << "[roostats_cla]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("mcmc") != std::string::npos){
    std::cout << "[roostats_cl95]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("cls") != std::string::npos){
    std::cout << "[roostats_cl95]: using CLs calculation" << endl;
  }
  else if (method.find("fc") != std::string::npos){
    std::cout << "[roostats_cl95]: using Feldman-Cousins approach" << endl;
  }
  else{
    std::cout << "[roostats_cla]: method " << method 
	      << " is not implemented, exiting" <<std::endl;
    return -1.0;
  }

  std::cout << "[roostats_cla]: Poisson statistics used" << endl;
    
  CL95Calc theCalc(seed);
  limit = theCalc.cla( ilum, slum,
		       eff, seff,
		       bck, sbck,
		       nuisanceModel,
		       method );

  //std::cout << "[roostats_cla]: average 95% C.L. upper limit: " << limit << std::endl;

  return limit;
}



LimitResult roostats_clm(Double_t ilum, Double_t slum,
			 Double_t eff, Double_t seff,
			 Double_t bck, Double_t sbck,
			 Int_t nit, Int_t nuisanceModel,
			 std::string method,
			 UInt_t seed){
  //
  // Global function to evaluate median expected limit and 1/2 sigma bands.
  //
  
  LimitResult limit;

  std::cout << "[roostats_clm]: estimating expected 95% C.L. upper limit" << endl;
  if (method.find("bayesian") != std::string::npos){
    std::cout << "[roostats_clm]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("mcmc") != std::string::npos){
    std::cout << "[roostats_cl95]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("cls") != std::string::npos){
    std::cout << "[roostats_cl95]: using CLs calculation" << endl;
  }
  else if (method.find("fc") != std::string::npos){
    std::cout << "[roostats_cl95]: using Feldman-Cousins approach" << endl;
  }
  else{
    std::cout << "[roostats_clm]: method " << method 
	      << "is not implemented, exiting" <<std::endl;
    return limit;
  }

  std::cout << "[roostats_clm]: Poisson statistics used" << endl;
    
  CL95Calc theCalc(seed);
  limit = theCalc.clm( ilum, slum,
		       eff, seff,
		       bck, sbck,
		       nit, nuisanceModel,
		       method );

  return limit;
}


/////////////////////////////////////////////////////////////////////////
//
// CLs helper methods from Lorenzo Moneta
// This is the core of the CLs calculation
//

bool plotHypoTestResult = false; 
bool useProof = false;
bool optimize = false;
bool writeResult = false;
int nworkers = 1;


// internal routine to run the inverter
HypoTestInverterResult * RunInverter(RooWorkspace * w, const char * modelSBName, const char * modelBName, const char * dataName,
                                     int type,  int testStatType, int npoints, double poimin, double poimax, int ntoys, bool useCls );




std::vector<Double_t>
GetClsLimits(RooWorkspace * pWs,
	     const char * modelSBName,
	     const char * modelBName,
	     const char * dataName,
	     int calculatorType,  // calculator type
	     int testStatType, // test stat type
	     bool useCls,
	     int npoints,
	     double poimin,  // use default is poimin >= poimax
	     double poimax,
	     int ntoys,
	     std::string suffix)
{

  //
  // Return a vector of numbers (terrible design, I know) ordered as
  //  - observed limit
  //  - observed limit error
  //  - expected limit median
  //  - expected limit -1 sigma
  //  - expected limit +1 sigma
  //  - expected limit -2 sigma
  //  - expected limit +2 sigma
  //

/*

   Other Parameter to pass in tutorial
   apart from standard for filename, ws, modelconfig and data

    type = 0 Freq calculator 
    type = 1 Hybrid 

    testStatType = 0 LEP
                 = 1 Tevatron 
                 = 2 Profile Likelihood
                 = 3 Profile Likelihood one sided (i.e. = 0 if mu < mu_hat)

    useCLs          scan for CLs (otherwise for CLs+b)    

    npoints:        number of points to scan , for autoscan set npoints = -1 

    poimin,poimax:  min/max value to scan in case of fixed scans 
                    (if min >= max, try to find automatically)                           

    ntoys:         number of toys to use 

    extra options are available as global paramters of the macro. They are: 

    plotHypoTestResult   plot result of tests at each point (TS distributions) 
    useProof = true;
    writeResult = true;
    nworkers = 4;


   */


  // result
  std::vector<Double_t> result;

  // check that workspace is present
  if (!pWs){
    std::cout << "No workspace found, null pointer" << std::endl;
    return result;
  }
  
  HypoTestInverterResult * r = 0;
  HypoTestInverterResult * r2 = 0;
  
  // terrible hack to check appending results
  if (suffix.find("merge")!=std::string::npos){
    std::string resFile = "Freq_CLs_grid_ts2_test_1.root";
    std::string resFile2 = "Freq_CLs_grid_ts2_test_2.root";
    std::string resName = "result_xsec";
    //std::cout << "Reading an HypoTestInverterResult with name " << resName << " from file " << resFile << std::endl;
    TFile * file = new TFile(resFile.c_str(), "read");
    TFile * file2 = new TFile(resFile2.c_str(), "read");
    r = dynamic_cast<HypoTestInverterResult*>( file->Get(resName.c_str()) ); 
    r2 = dynamic_cast<HypoTestInverterResult*>( file2->Get(resName.c_str()) ); 
    r->Add(*r2);
  }
  else{
    r = RunInverter(pWs, modelSBName, modelBName, dataName, calculatorType, testStatType, npoints, poimin, poimax,  ntoys, useCls );    
    if (!r) { 
      std::cerr << "Error running the HypoTestInverter - Exit " << std::endl;
      return result;
    }
  }
      		

   double upperLimit = r->UpperLimit();
   double ulError = r->UpperLimitEstimatedError();
   result.push_back(upperLimit);
   result.push_back(ulError);


   //std::cout << "The computed upper limit is: " << upperLimit << " +/- " << ulError << std::endl;
 
   //   const int nEntries = r->ArraySize();

   const char *  limitType = (useCls) ? "CLs" : "Cls+b";
   const char * scanType = (npoints < 0) ? "auto" : "grid";

   const char *  typeName = (calculatorType == 0) ? "Frequentist" : "Hybrid";
   const char * resultName = (pWs) ? pWs->GetName() : r->GetName();
   TString plotTitle = TString::Format("%s CL Scan for workspace %s",typeName,resultName);

   /*
   HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot",plotTitle,r);
   TCanvas c1;
   //plot->Draw("CLb 2CL");  // plot all and Clb
   plot->Draw("2CL");  // plot all and Clb
   TString resultFileName = TString::Format("%s_%s_ts%d_scan_",limitType,scanType,testStatType);
   resultFileName += suffix;
   resultFileName += ".pdf";
   c1.SaveAs(resultFileName);

   if (plotHypoTestResult) { 
      TCanvas * c2 = new TCanvas();
      c2->Divide( 2, TMath::Ceil(nEntries/2));
      for (int i=0; i<nEntries; i++) {
         c2->cd(i+1);
         SamplingDistPlot * pl = plot->MakeTestStatPlot(i);
         pl->SetLogYaxis(true);
         pl->Draw();
      }
   }
   */

   Double_t q[5];
   q[0] = r->GetExpectedUpperLimit(0);
   q[1] = r->GetExpectedUpperLimit(-1);
   q[2] = r->GetExpectedUpperLimit(1);
   q[3] = r->GetExpectedUpperLimit(-2);
   q[4] = r->GetExpectedUpperLimit(2);
   //std::cout << " expected limit (median) " << q[0] << std::endl;
   //std::cout << " expected limit (-1 sig) " << q[1] << std::endl;
   //std::cout << " expected limit (+1 sig) " << q[2] << std::endl;
   //std::cout << " expected limit (-2 sig) " << q[3] << std::endl;
   //std::cout << " expected limit (+2 sig) " << q[4] << std::endl;
   result.push_back(q[0]);
   result.push_back(q[1]);
   result.push_back(q[2]);
   result.push_back(q[3]);
   result.push_back(q[4]);


   if (pWs != NULL && writeResult) {

      // write to a file the results
      const char *  calcType = (calculatorType == 0) ? "Freq" : "Hybr";
      //const char *  limitType = (useCls) ? "CLs" : "Cls+b";
      //const char * scanType = (npoints < 0) ? "auto" : "grid";
      TString resultFileName = TString::Format("%s_%s_%s_ts%d_",calcType,limitType,scanType,testStatType);      
      //resultFileName += fileName;
      resultFileName += suffix;
      resultFileName += ".root";

      TFile * fileOut = new TFile(resultFileName,"RECREATE");
      r->Write();
      fileOut->Close();                                                                     
   }   

   return result;
}


// internal routine to run the inverter
HypoTestInverterResult *  RunInverter(RooWorkspace * w, const char * modelSBName, const char * modelBName, 
                                      const char * dataName, int type,  int testStatType, 
                                      int npoints, double poimin, double poimax, 
                                      int ntoys, bool useCls ) 
{

  //std::cout << "Running HypoTestInverter on the workspace " << w->GetName() << std::endl;

   //w->Print();


   RooAbsData * data = w->data(dataName); 
   if (!data) { 
      Error("StandardHypoTestDemo","Not existing data %s",dataName);
      return 0;
   }
   //else 
   //  std::cout << "Using data set " << dataName << std::endl;

   
   // get models from WS
   // get the modelConfig out of the file
   ModelConfig* bModel = (ModelConfig*) w->obj(modelBName);
   ModelConfig* sbModel = (ModelConfig*) w->obj(modelSBName);

   if (!sbModel) {
      Error("StandardHypoTestDemo","Not existing ModelConfig %s",modelSBName);
      return 0;
   }
   // check the model 
   if (!sbModel->GetPdf()) { 
      Error("StandardHypoTestDemo","Model %s has no pdf ",modelSBName);
      return 0;
   }
   if (!sbModel->GetParametersOfInterest()) {
      Error("StandardHypoTestDemo","Model %s has no poi ",modelSBName);
      return 0;
   }
   if (!sbModel->GetParametersOfInterest()) {
      Error("GetClsLimits","Model %s has no poi ",modelSBName);
      return 0;
   }
   if (!sbModel->GetSnapshot() ) { 
      Info("GetClsLimits","Model %s has no snapshot  - make one using model poi",modelSBName);
      sbModel->SetSnapshot( *sbModel->GetParametersOfInterest() );
   }


   if (!bModel || bModel == sbModel) {
      Info("GetClsLimits","The background model %s does not exist",modelBName);
      Info("GetClsLimits","Copy it from ModelConfig %s and set POI to zero",modelSBName);
      bModel = (ModelConfig*) sbModel->Clone();
      bModel->SetName(TString(modelSBName)+TString("_with_poi_0"));      
      RooRealVar * var = dynamic_cast<RooRealVar*>(bModel->GetParametersOfInterest()->first());
      if (!var) return 0;
      double oldval = var->getVal();
      var->setVal(0);
      bModel->SetSnapshot( RooArgSet(*var)  );
      var->setVal(oldval);
   }
   else { 
      if (!bModel->GetSnapshot() ) { 
         Info("GetClsLimits","Model %s has no snapshot  - make one using model poi and 0 values ",modelBName);
         RooRealVar * var = dynamic_cast<RooRealVar*>(bModel->GetParametersOfInterest()->first());
         if (var) { 
            double oldval = var->getVal();
            var->setVal(0);
            bModel->SetSnapshot( RooArgSet(*var)  );
            var->setVal(oldval);
         }
         else { 
            Error("GetClsLimits","Model %s has no valid poi",modelBName);
            return 0;
         }         
      }
   }


   SimpleLikelihoodRatioTestStat slrts(*sbModel->GetPdf(),*bModel->GetPdf());
   if (sbModel->GetSnapshot()) slrts.SetNullParameters(*sbModel->GetSnapshot());
   if (bModel->GetSnapshot()) slrts.SetAltParameters(*bModel->GetSnapshot());

   // ratio of profile likelihood - need to pass snapshot for the alt
   RatioOfProfiledLikelihoodsTestStat 
      ropl(*sbModel->GetPdf(), *bModel->GetPdf(), bModel->GetSnapshot());
   ropl.SetSubtractMLE(false);
   
   ProfileLikelihoodTestStat profll(*sbModel->GetPdf());
   if (testStatType == 3) profll.SetOneSided(1);
   if (optimize) profll.SetReuseNLL(true);

   TestStatistic * testStat = &slrts;
   if (testStatType == 1) testStat = &ropl;
   if (testStatType == 2 || testStatType == 3) testStat = &profll;
  
   
   HypoTestCalculatorGeneric *  hc = 0;
   if (type == 0) hc = new FrequentistCalculator(*data, *bModel, *sbModel);
   else hc = new HybridCalculator(*data, *bModel, *sbModel);

   ToyMCSampler *toymcs = (ToyMCSampler*)hc->GetTestStatSampler();
   // FIXME:
   toymcs->SetNEventsPerToy(1);
   toymcs->SetTestStatistic(testStat);
   if (optimize) toymcs->SetUseMultiGen(true);


   if (type == 1) { 
      HybridCalculator *hhc = (HybridCalculator*) hc;
      hhc->SetToys(ntoys,ntoys); 

      // check for nuisance prior pdf 
      //if (bModel->GetPriorPdf() && sbModel->GetPriorPdf() ) {
      //   hhc->ForcePriorNuisanceAlt(*bModel->GetPriorPdf());
      //   hhc->ForcePriorNuisanceNull(*sbModel->GetPriorPdf());
      //}
      RooAbsPdf * nuis_prior =  w->pdf("nuis_prior");
      if (nuis_prior ) {
         hhc->ForcePriorNuisanceAlt(*nuis_prior);
         hhc->ForcePriorNuisanceNull(*nuis_prior);
      }
      else {
         if (bModel->GetNuisanceParameters() || sbModel->GetNuisanceParameters() ) {
            Error("GetClsLimits","Cannnot run Hybrid calculator because no prior on the nuisance parameter is specified");
            return 0;
         }
      }
   } 
   else 
      ((FrequentistCalculator*) hc)->SetToys(ntoys,ntoys); 

   // Get the result
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);


   TStopwatch tw; tw.Start(); 
   const RooArgSet * poiSet = sbModel->GetParametersOfInterest();
   RooRealVar *poi = (RooRealVar*)poiSet->first();

   // fit the data first

   sbModel->GetPdf()->fitTo(*data, 
			    Verbose(0), PrintLevel(-1), Warnings(0), PrintEvalErrors(-1));

   double poihat  = poi->getVal();


   HypoTestInverter calc(*hc);
   calc.SetConfidenceLevel(0.95);

   calc.UseCLs(useCls);
   calc.SetVerbose(true);

   // can speed up using proof-lite
   if (useProof && nworkers > 1) { 
      ProofConfig pc(*w, nworkers, "", kFALSE);
      toymcs->SetProofConfig(&pc);    // enable proof
   }

   
   if (npoints > 0) {
      if (poimin >= poimax) { 
         // if no min/max given scan between MLE and +4 sigma 
         poimin = int(poihat);
         poimax = int(poihat +  4 * poi->getError());
      }
      //std::cout << "Doing a fixed scan in interval : " << poimin << " , " << poimax << std::endl;
      calc.SetFixedScan(npoints,poimin,poimax);
   }
   else { 
      //poi->setMax(10*int( (poihat+ 10 *poi->getError() )/10 ) );
     //std::cout << "Doing an  automatic scan in interval : " << poi->getMin() << " , " << poi->getMax() << std::endl;
   }

   HypoTestInverterResult * r = calc.GetInterval();

   return r; 
}

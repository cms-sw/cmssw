#ifndef TauAnalysis_SVfitStandalone_SVfitStandaloneLikelihood_h
#define TauAnalysis_SVfitStandalone_SVfitStandaloneLikelihood_h

#include "TauAnalysis/SVfitStandalone/interface/svFitStandaloneAuxFunctions.h"

#include "TMath.h"
#include "TMatrixD.h"
#include "Math/Minimizer.h"

#include <vector>

namespace svFitStandalone
{
  /**
     \enum    SVfitStandalone::kDecayType
     \brief   enumeration of all tau decay types
  */
  enum kDecayType {
    kUndefinedDecayType,
    kTauToHadDecay,  /* < hadronic tau lepton decay                                                        */ 
    kTauToElecDecay, /* < tau lepton decay to electron                                                     */
    kTauToMuDecay,   /* < tau lepton decay to muon                                                         */
    kPrompt          /* < prompt electron or muon not originating from tau decay (for LFV analysis)        */
  };
  /**
     \enum    SVfitStandalone::kFitParams
     \brief   enumeration of all parameters used by the SVfitAlgorithm to reconstruct the di-tau system
  */
  enum kFitParams {
    kXFrac,               /* < relative fraction of the visible energy on the total energy of the tau lepton    */
    kMNuNu,               /* < invariant mass of the neutrino system                                            */
    kPhi,                 /* < phi angle of the tau lepton (this is parameter is not constraint by measurement) */
    kVisMassShifted,      /* < mass resolution of the visible parts of the first tau decay branch               */    
    kRecTauPtDivGenTauPt, /* < Pt resolution of the visible parts of the first tau decay branch                 */       
    kMaxFitParams         /* < maximum number of fit parameters per resonance decay branch                      */
  };
  /**
     \enum    SVfitStandalone::kNLLParams
     \brief   enumeration of all parameters used to construct the combined likelihood including the logM term
  */
  enum kNLLParams {
    kNuNuMass1,            /* < mass of the neutrino system for the first decay branch                           */
    kVisMass1,             /* < mass of the visible parts of the first tau decay branch                          */
    kDecayAngle1,          /* < decay angle for the first decay branch (in restframe of the tau lepton decay)    */
    kDeltaVisMass1,        /* < mass resolution of the visible parts of the first tau decay branch               */    
    kRecTauPtDivGenTauPt1, /* < Pt resolution of the visible parts of the first tau decay branch                 */       
    kNuNuMass2,            /* < mass of the neutrino system for the second decay branch                          */
    kVisMass2,             /* < mass of the visible parts of the second tau decay branch                         */
    kDecayAngle2,          /* < decay angle for the second decay branch (in restframe of the tau lepton decay)   */  
    kDeltaVisMass2,        /* < mass resolution of the visible parts of the first tau decay branch               */    
    kRecTauPtDivGenTauPt2, /* < Pt resolution of the visible parts of the first tau decay branch                 */ 
    kDMETx,                /* < difference between the sum of the fitted neutrino px and px of the MET           */ 
    kDMETy,                /* < difference between the sum of the fitted neutrino py and py of the MET           */
    kMTauTau,              /* < invariant mass of the fitted di-tau system (used for the logM penalty term)      */
    kMaxNLLParams          /* < max number of parameters used for for the combined likelihood                    */
  };
  /**
     \class   MeasuredTauLepton SVfitStandaloneLikelihood.h "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneLikelihood.h"
     
     \brief   Helper class to simplify the configuration of the SVfitStandaloneLikelihood class. 
     
     This is a helper class to facilitate the configuration of the SVfitStandaloneLikelihood class. It keeps the spacial momentum
     energy and information about the type of tau lepton decay. All information is stored in the lab-frame. A few getter functions 
     facilitate access to the information.
  */
  class MeasuredTauLepton
  {
   public:
    /// default constructor 
    MeasuredTauLepton()
      : type_(kUndefinedDecayType),
        pt_(0.),
        eta_(0.),
        phi_(0.),
        mass_(0.),
        decayMode_(-1)
    {
      initialize();
    }
    /// constructor from the measured quantities per decay branch
    MeasuredTauLepton(kDecayType type, double pt, double eta, double phi, double mass, int decayMode = -1) 
      : type_(type), 
        pt_(roundToNdigits(pt)),
        eta_(roundToNdigits(eta)),
        phi_(roundToNdigits(phi)),
        mass_(roundToNdigits(mass)),
 	decayMode_(decayMode)
    {
      //std::cout << "<MeasuredTauLepton>: Pt = " << pt_ << ", eta = " << eta_ << ", phi = " << phi_ << ", mass = " << mass_ << std::endl;
      double minVisMass = electronMass;
      double maxVisMass = tauLeptonMass;
      std::string type_string;
      if ( type_ == kTauToElecDecay ) {
	minVisMass = electronMass;
	maxVisMass = minVisMass;
      } else if ( type_ == kTauToMuDecay ) {
	minVisMass = muonMass;
	maxVisMass = minVisMass;
      } else if ( type_ == kTauToHadDecay ) {
	if ( decayMode_ == -1 ) {
	  minVisMass = chargedPionMass;
	  maxVisMass = 1.5;
	} else if ( decayMode_ == 0 ) {
	  minVisMass = chargedPionMass;
	  maxVisMass = minVisMass;
	} else {
	  minVisMass = 0.3;
	  maxVisMass = 1.5;
	}
      } 
      preciseVisMass_ = mass_;
      if ( preciseVisMass_ < (0.9*minVisMass) || preciseVisMass_ > (1.1*maxVisMass) ) {
	std::string type_string;
	if      ( type_ == kTauToElecDecay ) type_string = "tau -> electron decay";
	else if ( type_ == kTauToMuDecay   ) type_string = "tau -> muon decay";
	else if ( type_ == kTauToHadDecay  ) type_string = "tau -> had decay";
	else if ( type_ == kPrompt         ) type_string = "prompt lepton"; 
	else {
	  std::cerr << "Error: Invalid type " << type_ << " declared for leg: Pt = " << pt_ << ", eta = " << eta_ << ", phi = " << phi_ << ", mass = " << mass_ << " !!" << std::endl;
	  assert(0);
	}
	std::cerr << "Warning: " << type_string << " declared for leg: Pt = " << pt_ << ", eta = " << eta_ << ", phi = " << phi_ << ", mass = " << mass_ << " !!" << std::endl;
	std::cerr << " (mass expected in the range = " << minVisMass << ".." << maxVisMass << ")" << std::endl;
      }
      if ( preciseVisMass_ < minVisMass ) preciseVisMass_ = minVisMass;
      if ( preciseVisMass_ > maxVisMass ) preciseVisMass_ = maxVisMass;
      initialize();
    }
    /// copy constructor
    MeasuredTauLepton(const MeasuredTauLepton& measuredTauLepton)
      : type_(measuredTauLepton.type()), 
        pt_(measuredTauLepton.pt()),
        eta_(measuredTauLepton.eta()),
        phi_(measuredTauLepton.phi()),
        mass_(measuredTauLepton.mass()), 
        decayMode_(measuredTauLepton.decayMode())     
    {
      preciseVisMass_ = measuredTauLepton.mass();
      initialize();
    }
    /// default destructor
    ~MeasuredTauLepton() {}

    void initialize()
    {
      momentum_ = pt_*TMath::CosH(eta_);
      energy_ = TMath::Sqrt(momentum_*momentum_ + mass_*mass_);
      px_ = pt_*TMath::Cos(phi_);
      py_ = pt_*TMath::Sin(phi_);
      pz_ = pt_*TMath::SinH(eta_);
      p4_ = LorentzVector(px_, py_, pz_, energy_);
      p_ = p4_.Vect();
      direction_ = p_.unit();
    }

    /// return pt of the measured tau lepton in labframe
    double pt() const { return pt_; }
    /// return px of the measured tau lepton in labframe
    double px() const { return px_; }
    /// return py of the measured tau lepton in labframe
    double py() const { return py_; }
    /// return visible mass in labframe
    double mass() const { return preciseVisMass_; }    
    /// return visible energy in labframe
    double energy() const { return energy_; }
    /// return visible momenumt in labframe
    double momentum() const { return momentum_; }
    /// return pseudo-rapidity of the measured tau lepton in labframe
    double eta() const { return eta_; }
    /// return azimuthal angle of the measured tau lepton in labframe
    double phi() const { return phi_; }
    /// return decay type of the tau lepton
    int type() const { return type_; }
    /// return decay mode of the reconstructed hadronic tau decay
    int decayMode() const { return decayMode_; }    
    /// return the spacial momentum vector in the labframe
    Vector p() const { return p_; }
    /// return the lorentz vector in the labframe
    LorentzVector p4() const { return p4_; }
    /// return the direction of the visible 
    Vector direction() const { return direction_; }
    
   private:
    /// decay type
    int type_;
    /// visible momentum in labframe 
    double pt_;
    double eta_;
    double phi_;
    double mass_;
    double energy_;
    double px_;
    double py_;
    double pz_;
    double momentum_;
    LorentzVector p4_;
    Vector p_;
    Vector direction_;
    /// mass of visible tau decay products (recomputed to reduce rounding errors)
    double preciseVisMass_;
    /// decay mode (hadronic tau decays only)
    int decayMode_;
  };

  /**
   \class   SVfitStandaloneLikelihood SVfitStandaloneLikelihood.h "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneLikelihood.h"
       
     \brief   Negative log likelihood for a resonance decay into two tau leptons that may themselves decay hadronically or leptonically
     
     Negative log likelihood for a resonance decay into two tau leptons that may themselves decay hadronically or leptonically 
     Depending on the configuration during object creation it will be a combination of MET, TauToHad, TauToLep and additional
     penalty terms, e.g. to suppress tails in m(tau,tau) (logM). Configurables during creation time are:
     
     \var measuredTauLeptons : the vector of the two reconstructed tau leptons
     \var measuredMET        : the spacial vector of the measured MET
     \var covMET             : the covariance matrix of the MET (as determined from the MEt significance for instance)
     \verbose                : indicating the verbosity level 

     In fit mode additional optional values may be set before the fit is performed. During construction the class is initialized with 
     default values as indicated in braces (below):

     \var metPower : indicating an additional power to enhance the MET likelihood (default is 1.)
     \var addLogM  : specifying whether to use the LogM penalty term or not (default is true)     

     A typical way to obtain the covariance matrix of the MET is to follow the MET significance algorithm as provided by RecoMET.
     The SVfitStandaloneLikelihood class is for internal use only. The general use calse is to access it from the class 
     SVfitStandaloneAlgorithm as defined in interface/SVfitStandaloneAlgorithm.h in the same package. The SVfitLikelihood class 
     keeps all necessary information to calculate the combined likelihood but does not perform any fit nor integration. It is 
     interfaced to the ROOT minuit minimization package or to the VEGAS integration packages via the global function pointer 
     gSVfitStandaloneLikelihood as defined in src/SVfitStandaloneLikelihood.cc in the same package. 
  */

  class SVfitStandaloneLikelihood 
  {
   public:
    /// error codes that can be read out by SVfitAlgorithm
    enum ErrorCodes {
      None            = 0x00000000,
      MatrixInversion = 0x00000001,
      LeptonNumber    = 0x00000010
    };
    /// constructor with a minimla set of configurables 
    SVfitStandaloneLikelihood(const std::vector<svFitStandalone::MeasuredTauLepton>& measuredTauLeptons, const svFitStandalone::Vector& measuredMET, const TMatrixD& covMET, bool verbose);
    /// default destructor
    ~SVfitStandaloneLikelihood() {}
    /// static pointer to this (needed for the minuit function calls)
    static const SVfitStandaloneLikelihood* gSVfitStandaloneLikelihood;

    /// add an additional logM(tau,tau) term to the nll to suppress tails on M(tau,tau) (default is false)
    void addLogM(bool value, double power = 1.) { addLogM_ = value; powerLogM_ = power; }
    /// add derrivative of delta-function 
    /// WARNING: to be used when SVfit is run in "integration" mode only
    void addDelta(bool value) { addDelta_ = value; }
    /// add a penalty term in case phi runs outside of interval 
    /// WARNING: to be used when SVfit is run in "fit" mode only
    void addPhiPenalty(bool value) { addPhiPenalty_ = value; }        
    /// add sin(theta) term to likelihood for tau lepton decays
    /// WARNING: to be used when SVfit is run in "fit" mode only
    void addSinTheta(bool value) { addSinTheta_ = value; }  
    /// marginalize unknown mass of hadronic tau decay products (ATLAS case)
    void marginalizeVisMass(bool value, const TH1* l1lutVisMass, const TH1* l2lutVisMass);  
    /// take resolution on energy and mass of hadronic tau decays into account
    void shiftVisMass(bool value, const TH1* l1lutVisMassRes, const TH1* l2lutVisMassRes);
    void shiftVisPt(bool value, const TH1* l1lutVisPtRes, const TH1* l2lutVisPtRes);
    /// add a penalty term in case phi runs outside of interval 
    /// modify the MET term in the nll by an additional power (default is 1.)
    void metPower(double value) { metPower_=value; };    

    /// flag to force prob to be zero in case of unphysical solutions
    /// (to be used in integration, but not in fit mode, as MINUIT will get confused otherwise)
    void requirePhysicalSolution(bool value) { requirePhysicalSolution_ = value; }

    /// fit function to be called from outside. Has to be const to be usable by minuit. This function will call the actual 
    /// functions transform and prob internally 
    double prob(const double* x, bool fixToMtest = false, double mtest = -1.) const;
    /// read out potential likelihood errors
    unsigned error() const { return errorCode_; }

    /// return vector of measured MET
    const svFitStandalone::Vector& measuredMET() const { return measuredMET_; }
    /// return vector of measured tau leptons
    const std::vector<svFitStandalone::MeasuredTauLepton>& measuredTauLeptons() const { return measuredTauLeptons_; }
    /// return vector of fitted tau leptons, which will be the actual fit result. This function is a subset of transform.
    /// It needs to be factored out though as transform has to be const to be usable by minuit and therefore is not allowed 
    /// change the class members.  
    void results(std::vector<svFitStandalone::LorentzVector>& fittedTauLeptons, const double* x) const;

   protected:
    /// transformation from x to xPrime, x are the actual fit parameters, xPrime are the transformed parameters that go into 
    /// the prob function. Has to be const to be usable by minuit.
    const double* transform(double* xPrime, const double* x, bool fixToMtest, double mtest) const;
    /// combined likelihood function. The same function os called for fit and integratino mode. Has to be const to be usable 
    /// by minuit/VEGAS/MarkovChain. The additional boolean phiPenalty is added to prevent singularities at the +/-pi boundaries 
    /// of kPhi within the fit parameters (kFitParams). It is only used in fit mode. In integration mode the passed on value 
    /// is always 0. 
    double prob(const double* xPrime, double phiPenalty) const;
    
   protected:
    /// additional power to enhance MET term in the nll (default is 1.)
    double metPower_;
    /// add a logM penalty term in the nll
    bool addLogM_;
    double powerLogM_;
    /// delta-function derrivative 
    bool addDelta_;
    /// sin(theta) term in the nll
    bool addSinTheta_;
    /// add a penalty term in case phi runs outside of interval [-pi,+pi]
    bool addPhiPenalty_;
    /// verbosity level
    bool verbose_;
    /// monitor the number of function calls
    mutable unsigned int idxObjFunctionCall_;

    /// measured tau leptons
    std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons_;
    /// measured MET
    svFitStandalone::Vector measuredMET_;
    /// transfer matrix for MET in (inverse covariance matrix) 
    TMatrixD invCovMET_;
    /// determinant of the covariance matrix of MET
    double covDet_;
    /// error code that can be passed on
    unsigned int errorCode_;

    /// flag to force prob to be zero in case of unphysical solutions
    /// (to be used in integration, but not in fit mode, as MINUIT will get confused otherwise)
    bool requirePhysicalSolution_;

    /// resolution on energy and mass of hadronic taus
    bool marginalizeVisMass_;
    const TH1* l1lutVisMass_;
    const TH1* l2lutVisMass_;
    bool shiftVisMass_;
    const TH1* l1lutVisMassRes_;
    const TH1* l2lutVisMassRes_;
    bool shiftVisPt_;
    const TH1* l1lutVisPtRes_;
    const TH1* l2lutVisPtRes_;
  };
}

#endif

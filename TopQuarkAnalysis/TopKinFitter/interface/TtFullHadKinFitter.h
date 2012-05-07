#ifndef TtFullHadKinFitter_h
#define TtFullHadKinFitter_h

#include <vector>

#include "TLorentzVector.h"

#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/CovarianceMatrix.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TopKinFitter.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TAbsFitParticle;
class TFitConstraintM;

/*
  \class   TtFullHadKinFitter TtFullHadKinFitter.h "TopQuarkAnalysis/TopKinFitter/interface/TtFullHadKinFitter.h"
  
  \brief   one line description to be added here...

  text to be added here...
  
**/

class TtFullHadKinFitter : public TopKinFitter {

 public:
  /// supported constraints
  enum Constraint{ kWPlusMass=1, kWMinusMass, kTopMass, kTopBarMass, kEqualTopMasses };
  
 public:
  /// default constructor
  TtFullHadKinFitter();
  /// used to convert vector of int's to vector of constraints (just used in TtFullHadKinFitter(int, int, double, double, std::vector<unsigned int>))
  std::vector<TtFullHadKinFitter::Constraint> intToConstraint(std::vector<unsigned int> constraints);
  /// constructor initialized with build-in types as custom parameters (only included to keep TtHadEvtSolutionMaker.cc running)
  TtFullHadKinFitter(int jetParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<unsigned int> constraints,
		     double mW=80.4, double mTop=173.);
  /// constructor initialized with built-in types and class enum's custom parameters
  TtFullHadKinFitter(Param jetParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<Constraint> constraints,
		     double mW=80.4, double mTop=173.);
  /// default destructor
  ~TtFullHadKinFitter();

  /// kinematic fit interface
  int fit(const std::vector<pat::Jet>& jets);
  int fit(const std::vector<pat::Jet>& jets, const std::vector<edm::ParameterSet> udscResolutions, const std::vector<edm::ParameterSet> bResolutions, const double energyResolutionSmearFactor);
  /// return fitted b quark candidate
  const pat::Particle fittedB() const { return (fitter_->getStatus()==0 ? fittedB_ : pat::Particle()); };
  /// return fitted b quark candidate
  const pat::Particle fittedBBar() const { return (fitter_->getStatus()==0 ? fittedBBar_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightQ() const { return (fitter_->getStatus()==0 ? fittedLightQ_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightQBar() const { return (fitter_->getStatus()==0 ? fittedLightQBar_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightP() const { return (fitter_->getStatus()==0 ? fittedLightP_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightPBar() const { return (fitter_->getStatus()==0 ? fittedLightPBar_ : pat::Particle()); };
  /// add kin fit information to the old event solution (in for legacy reasons)
  TtHadEvtSolution addKinFitInfo(TtHadEvtSolution * asol);
  
 private:
  /// print fitter setup
  void printSetup() const;
  /// setup fitter  
  void setupFitter();
  /// initialize jet inputs
  void setupJets();
  /// initialize constraints
  void setupConstraints();

 private:
  /// input particles
  TAbsFitParticle* b_;
  TAbsFitParticle* bBar_;
  TAbsFitParticle* lightQ_;
  TAbsFitParticle* lightQBar_;
  TAbsFitParticle* lightP_;
  TAbsFitParticle* lightPBar_;
  /// supported constraints
  std::map<Constraint, TFitConstraintM*> massConstr_;
  /// output particles
  pat::Particle fittedB_;
  pat::Particle fittedBBar_;
  pat::Particle fittedLightQ_;
  pat::Particle fittedLightQBar_;
  pat::Particle fittedLightP_;
  pat::Particle fittedLightPBar_;
  /// jet parametrization
  Param jetParam_;
  /// vector of constraints to be used
  std::vector<Constraint> constraints_;

  /// get object resolutions and put them into a matrix
  CovarianceMatrix * covM;

 public:

  /// struct for fit results
  struct KinFitResult {
    int Status;
    double Chi2;
    double Prob;
    pat::Particle B;
    pat::Particle BBar;
    pat::Particle LightQ;
    pat::Particle LightQBar;
    pat::Particle LightP;
    pat::Particle LightPBar;
    std::vector<int> JetCombi;
    bool operator< (const KinFitResult& rhs) { return Chi2 < rhs.Chi2; };
  };

  /// class that does the fitting
  class KinFit {

  public:
    
    /// default constructor  
    KinFit();
    /// special constructor  
    KinFit(bool useBTagging, unsigned int bTags, std::string bTagAlgo, double minBTagValueBJet, double maxBTagValueNonBJet,
	   std::vector<edm::ParameterSet> udscResolutions, std::vector<edm::ParameterSet> bResolutions, double energyResolutionSmearFactor,
	   std::string jetCorrectionLevel, int maxNJets, int maxNComb,
	   unsigned int maxNrIter, double maxDeltaS, double maxF, unsigned int jetParam, std::vector<unsigned> constraints, double mW, double mTop);
    /// default destructor  
    ~KinFit();
    
    /// set all parameters for b-tagging
    void setBTagging(bool useBTagging, unsigned int bTags, std::string bTagAlgo, double minBTagValueBJet, double maxBTagValueNonBJet){
      useBTagging_         = useBTagging;
      bTags_               = bTags;
      bTagAlgo_            = bTagAlgo;
      minBTagValueBJet_    = minBTagValueBJet;
      maxBTagValueNonBJet_ = maxBTagValueNonBJet;
    }
    /// set resolutions
    void setResolutions(std::vector<edm::ParameterSet> udscResolutions, std::vector<edm::ParameterSet> bResolutions, double energyResolutionSmearFactor = 1.){
      udscResolutions_       = udscResolutions;
      bResolutions_          = bResolutions;
      energyResolutionSmearFactor_ = energyResolutionSmearFactor;
    }
    /// set parameters for fitter
    void setFitter(int maxNJets, unsigned int maxNrIter, double maxDeltaS, double maxF,
		   unsigned int jetParam, std::vector<unsigned> constraints, double mW, double mTop){
      maxNJets_    = maxNJets;
      maxNrIter_   = maxNrIter;
      maxDeltaS_   = maxDeltaS;
      maxF_        = maxF;
      jetParam_    = jetParam;
      constraints_ = constraints;
      mW_          = mW;
      mTop_        = mTop;
    }
    /// set jec level
    void setJEC(std::string jetCorrectionLevel){
      jetCorrectionLevel_ = jetCorrectionLevel;
    }
    /// set useOnlyMatch
    void setUseOnlyMatch(bool useOnlyMatch){
      useOnlyMatch_ = useOnlyMatch;
    }
    /// set match to be used
    void setMatch(std::vector<int> match){
      match_ = match;
    }
    /// set the validity of a match
    void setMatchInvalidity(bool invalidMatch){
      invalidMatch_ = invalidMatch;
    }
    /// set number of combinations of output
    void setOutput(int maxNComb){
      maxNComb_ = maxNComb;
    }

    /// do the fitting and return fit result
    std::list<TtFullHadKinFitter::KinFitResult> fit(const std::vector<pat::Jet>& jets);
    
  private:

    // helper function for b-tagging
    bool doBTagging(const std::vector<pat::Jet>& jets, const unsigned int& bJetCounter, std::vector<int>& combi);
    /// helper function to construct the proper corrected jet for its corresponding quarkType
    pat::Jet corJet(const pat::Jet& jet, const std::string& quarkType);
    
    // convert unsigned to Param
    TtFullHadKinFitter::Param param(unsigned int configParameter);
    // convert unsigned int to Constraint
    TtFullHadKinFitter::Constraint constraint(unsigned int configParameter);
    // convert vector of unsigned int's to vector of Contraint's
    std::vector<TtFullHadKinFitter::Constraint> constraints(std::vector<unsigned int>& configParameters);
    
    /// switch to tell whether all possible 
    /// combinations should be used for the
    /// switch to tell whether to use b-tagging or not
    bool useBTagging_;
    /// minimal number of b-jets
    unsigned int bTags_;
    /// input tag for b-tagging algorithm
    std::string bTagAlgo_;
    /// min value of bTag for a b-jet
    double minBTagValueBJet_;
    /// max value of bTag for a non-b-jet
    double maxBTagValueNonBJet_;
    /// store the resolutions for the jets
    std::vector<edm::ParameterSet> udscResolutions_, bResolutions_;
    /// smearing factor for jet energy resolutions
    double energyResolutionSmearFactor_;
    /// correction level for jets
    std::string jetCorrectionLevel_;
    /// maximal number of jets (-1 possible to indicate 'all')
    int maxNJets_;
    /// maximal number of combinations to be written to the event
    int maxNComb_;
    /// maximal number of iterations to be performed for the fit
    unsigned int maxNrIter_;
    /// maximal chi2 equivalent
    double maxDeltaS_;
    /// maximal deviation for contstraints
    double maxF_;
    /// numbering of different possible jet parametrizations
    unsigned int jetParam_;
    /// numbering of different possible kinematic constraints
    std::vector<unsigned> constraints_;
    /// W mass value used for constraints
    double mW_;
    /// top mass value used for constraints
    double mTop_;
    /// fit or only a certain combination
    bool useOnlyMatch_;
    /// the combination that should be used
    std::vector<int> match_;
    /// match is invalid
    bool invalidMatch_;

    /// kinematic fit interface
    TtFullHadKinFitter* fitter;
 
  };
};

#endif

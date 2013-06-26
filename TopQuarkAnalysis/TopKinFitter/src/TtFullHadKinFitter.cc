#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullHadKinFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

static const unsigned int nPartons=6;

/// default constructor
TtFullHadKinFitter::TtFullHadKinFitter():
  TopKinFitter(),
  b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  udscResolutions_(0), bResolutions_(0),
  jetEnergyResolutionScaleFactors_(0), jetEnergyResolutionEtaBinning_(0),
  jetParam_(kEMom)
{
  setupFitter();
}

/// used to convert vector of int's to vector of constraints (just used in TtFullHadKinFitter(int, int, double, double, std::vector<unsigned int>))
std::vector<TtFullHadKinFitter::Constraint>
TtFullHadKinFitter::intToConstraint(const std::vector<unsigned int>& constraints)
{
  std::vector<TtFullHadKinFitter::Constraint> cConstraints;
  cConstraints.resize(constraints.size());
  for(unsigned int i=0;i<constraints.size();++i)
    {
      cConstraints[i]=(Constraint)constraints[i];
    }
  return cConstraints;
}

/// constructor initialized with build-in types as custom parameters (only included to keep TtHadEvtSolutionMaker.cc running)
TtFullHadKinFitter::TtFullHadKinFitter(int jetParam, int maxNrIter, double maxDeltaS, double maxF,
				       const std::vector<unsigned int>& constraints, double mW, double mTop,
				       const std::vector<edm::ParameterSet>* udscResolutions, 
				       const std::vector<edm::ParameterSet>* bResolutions,
				       const std::vector<double>* jetEnergyResolutionScaleFactors,
				       const std::vector<double>* jetEnergyResolutionEtaBinning):
  TopKinFitter(maxNrIter, maxDeltaS, maxF, mW, mTop),
  b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  udscResolutions_(udscResolutions), bResolutions_(bResolutions),
  jetEnergyResolutionScaleFactors_(jetEnergyResolutionScaleFactors),
  jetEnergyResolutionEtaBinning_(jetEnergyResolutionEtaBinning),
  jetParam_((Param)jetParam), constraints_(intToConstraint(constraints))
{
  setupFitter();
}

/// constructor initialized with build-in types and class enum's custom parameters
TtFullHadKinFitter::TtFullHadKinFitter(Param jetParam, int maxNrIter, double maxDeltaS, double maxF,
				       const std::vector<Constraint>& constraints, double mW, double mTop,
				       const std::vector<edm::ParameterSet>* udscResolutions, 
				       const std::vector<edm::ParameterSet>* bResolutions,
				       const std::vector<double>* jetEnergyResolutionScaleFactors,
				       const std::vector<double>* jetEnergyResolutionEtaBinning):
  TopKinFitter(maxNrIter, maxDeltaS, maxF, mW, mTop),
  b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  udscResolutions_(udscResolutions), bResolutions_(bResolutions),
  jetEnergyResolutionScaleFactors_(jetEnergyResolutionScaleFactors),
  jetEnergyResolutionEtaBinning_(jetEnergyResolutionEtaBinning),
  jetParam_(jetParam), constraints_(constraints)
{
  setupFitter();
}

/// default destructor
TtFullHadKinFitter::~TtFullHadKinFitter() 
{
  delete b_; 
  delete bBar_; 
  delete lightQ_;
  delete lightQBar_; 
  delete lightP_; 
  delete lightPBar_;
  delete covM_;
  for(std::map<Constraint, TFitConstraintM*>::iterator it = massConstr_.begin(); it != massConstr_.end(); ++it)
    delete it->second;
}

/// print fitter setup
void 
TtFullHadKinFitter::printSetup() const
{
  std::stringstream constr;
  for(unsigned int i=0; i<constraints_.size(); ++i){
    switch(constraints_[i]){
    case kWPlusMass      : constr << "    * W+-mass   (" << mW_   << " GeV) \n"; break;
    case kWMinusMass     : constr << "    * W--mass   (" << mW_   << " GeV) \n"; break;
    case kTopMass        : constr << "    * t-mass    (" << mTop_ << " GeV) \n"; break;
    case kTopBarMass     : constr << "    * tBar-mass (" << mTop_ << " GeV) \n"; break;
    case kEqualTopMasses : constr << "    * equal t-masses \n"; break;
    }
  }
  edm::LogVerbatim( "TtFullHadKinFitter" ) 
    << "\n"
    << "+++++++++++ TtFullHadKinFitter Setup ++++++++++++ \n"
    << "  Parametrization:                                \n" 
    << "   * jet : " << param(jetParam_) << "\n"
    << "  Constraints:                                    \n"
    <<    constr.str()
    << "  Max(No iterations): " << maxNrIter_ << "\n"
    << "  Max(deltaS)       : " << maxDeltaS_ << "\n"
    << "  Max(F)            : " << maxF_      << "\n"
    << "+++++++++++++++++++++++++++++++++++++++++++++++++ \n";
}

/// initialize jet inputs
void 
TtFullHadKinFitter::setupJets()
{
  TMatrixD empty3x3(3,3); 
  TMatrixD empty4x4(4,4);
  switch(jetParam_){ // setup jets according to parameterization
  case kEMom :
    b_        = new TFitParticleEMomDev   ("Jet1", "Jet1", 0, &empty4x4);
    bBar_     = new TFitParticleEMomDev   ("Jet2", "Jet2", 0, &empty4x4);
    lightQ_   = new TFitParticleEMomDev   ("Jet3", "Jet3", 0, &empty4x4);
    lightQBar_= new TFitParticleEMomDev   ("Jet4", "Jet4", 0, &empty4x4);
    lightP_   = new TFitParticleEMomDev   ("Jet5", "Jet5", 0, &empty4x4);
    lightPBar_= new TFitParticleEMomDev   ("Jet6", "Jet6", 0, &empty4x4);
    break;
  case kEtEtaPhi :
    b_        = new TFitParticleEtEtaPhi  ("Jet1", "Jet1", 0, &empty3x3);
    bBar_     = new TFitParticleEtEtaPhi  ("Jet2", "Jet2", 0, &empty3x3);
    lightQ_   = new TFitParticleEtEtaPhi  ("Jet3", "Jet3", 0, &empty3x3);
    lightQBar_= new TFitParticleEtEtaPhi  ("Jet4", "Jet4", 0, &empty3x3);
    lightP_   = new TFitParticleEtEtaPhi  ("Jet5", "Jet5", 0, &empty3x3);
    lightPBar_= new TFitParticleEtEtaPhi  ("Jet6", "Jet6", 0, &empty3x3);
    break;
  case kEtThetaPhi :
    b_        = new TFitParticleEtThetaPhi("Jet1", "Jet1", 0, &empty3x3);
    bBar_     = new TFitParticleEtThetaPhi("Jet2", "Jet2", 0, &empty3x3);
    lightQ_   = new TFitParticleEtThetaPhi("Jet3", "Jet3", 0, &empty3x3);
    lightQBar_= new TFitParticleEtThetaPhi("Jet4", "Jet4", 0, &empty3x3);
    lightP_   = new TFitParticleEtThetaPhi("Jet5", "Jet5", 0, &empty3x3);
    lightPBar_= new TFitParticleEtThetaPhi("Jet6", "Jet6", 0, &empty3x3);
    break;
  }
}

/// initialize constraints
void 
TtFullHadKinFitter::setupConstraints() 
{
  massConstr_[kWPlusMass     ] = new TFitConstraintM("WPlusMass"     , "WPlusMass"      ,  0,  0, mW_   );
  massConstr_[kWMinusMass    ] = new TFitConstraintM("WMinusMass"    , "WMinusMass"     ,  0,  0, mW_   );
  massConstr_[kTopMass       ] = new TFitConstraintM("TopMass"       , "TopMass"        ,  0,  0, mTop_ );
  massConstr_[kTopBarMass    ] = new TFitConstraintM("TopBarMass"    , "TopBarMass"     ,  0,  0, mTop_ );
  massConstr_[kEqualTopMasses] = new TFitConstraintM("EqualTopMasses", "EqualTopMasses" ,  0,  0, 0     );
  
  massConstr_[kWPlusMass     ]->addParticles1(lightQ_, lightQBar_);
  massConstr_[kWMinusMass    ]->addParticles1(lightP_, lightPBar_);
  massConstr_[kTopMass       ]->addParticles1(b_, lightQ_, lightQBar_);
  massConstr_[kTopBarMass    ]->addParticles1(bBar_, lightP_, lightPBar_);
  massConstr_[kEqualTopMasses]->addParticles1(b_, lightQ_, lightQBar_);
  massConstr_[kEqualTopMasses]->addParticles2(bBar_, lightP_, lightPBar_);

}

/// setup fitter 
void 
TtFullHadKinFitter::setupFitter() 
{
  printSetup();
  setupJets();
  setupConstraints();

  // add measured particles
  fitter_->addMeasParticle(b_);
  fitter_->addMeasParticle(bBar_);
  fitter_->addMeasParticle(lightQ_);
  fitter_->addMeasParticle(lightQBar_);
  fitter_->addMeasParticle(lightP_);
  fitter_->addMeasParticle(lightPBar_);

  // add constraints
  for(unsigned int i=0; i<constraints_.size(); i++){
    fitter_->addConstraint(massConstr_[constraints_[i]]);
  }

  // initialize helper class used to bring the resolutions into covariance matrices
  if(udscResolutions_->size() &&  bResolutions_->size())
    covM_ = new CovarianceMatrix(*udscResolutions_, *bResolutions_,
				 *jetEnergyResolutionScaleFactors_,
				 *jetEnergyResolutionEtaBinning_);
  else
    covM_ = new CovarianceMatrix();
}

/// kinematic fit interface
int 
TtFullHadKinFitter::fit(const std::vector<pat::Jet>& jets)
{
  if( jets.size()<6 ){
    throw edm::Exception( edm::errors::Configuration, "Cannot run the TtFullHadKinFitter with less than 6 jets" );
  }

  // get jets in right order
  const pat::Jet& b         = jets[TtFullHadEvtPartons::B        ];
  const pat::Jet& bBar      = jets[TtFullHadEvtPartons::BBar     ];
  const pat::Jet& lightQ    = jets[TtFullHadEvtPartons::LightQ   ];
  const pat::Jet& lightQBar = jets[TtFullHadEvtPartons::LightQBar];
  const pat::Jet& lightP    = jets[TtFullHadEvtPartons::LightP   ];
  const pat::Jet& lightPBar = jets[TtFullHadEvtPartons::LightPBar];
 
  // initialize particles
  const TLorentzVector p4B( b.px(), b.py(), b.pz(), b.energy() );
  const TLorentzVector p4BBar( bBar.px(), bBar.py(), bBar.pz(), bBar.energy() );
  const TLorentzVector p4LightQ( lightQ.px(), lightQ.py(), lightQ.pz(), lightQ.energy() );
  const TLorentzVector p4LightQBar( lightQBar.px(), lightQBar.py(), lightQBar.pz(), lightQBar.energy() );
  const TLorentzVector p4LightP( lightP.px(), lightP.py(), lightP.pz(), lightP.energy() );
  const TLorentzVector p4LightPBar( lightPBar.px(), lightPBar.py(), lightPBar.pz(), lightPBar.energy() );

  // initialize covariance matrices
  TMatrixD m1 = covM_->setupMatrix(lightQ,    jetParam_);
  TMatrixD m2 = covM_->setupMatrix(lightQBar, jetParam_);
  TMatrixD m3 = covM_->setupMatrix(b,         jetParam_, "bjets");
  TMatrixD m4 = covM_->setupMatrix(lightP,    jetParam_);
  TMatrixD m5 = covM_->setupMatrix(lightPBar, jetParam_);
  TMatrixD m6 = covM_->setupMatrix(bBar     , jetParam_, "bjets");

  // set the kinematics of the objects to be fitted
  b_        ->setIni4Vec(&p4B        );
  bBar_     ->setIni4Vec(&p4BBar     );
  lightQ_   ->setIni4Vec(&p4LightQ   );
  lightQBar_->setIni4Vec(&p4LightQBar);
  lightP_   ->setIni4Vec(&p4LightP   );
  lightPBar_->setIni4Vec(&p4LightPBar);
  
  // initialize covariance matrices
  lightQ_   ->setCovMatrix( &m1);
  lightQBar_->setCovMatrix( &m2);
  b_        ->setCovMatrix( &m3);
  lightP_   ->setCovMatrix( &m4);
  lightPBar_->setCovMatrix( &m5);
  bBar_     ->setCovMatrix( &m6);
  
  // perform the fit!
  fitter_->fit();
  
  // add fitted information to the solution
  if( fitter_->getStatus()==0 ){
    // read back jet kinematics
    fittedB_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(b_->getCurr4Vec()->X(), b_->getCurr4Vec()->Y(), b_->getCurr4Vec()->Z(), b_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightQ_   = pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightQ_->getCurr4Vec()->X(), lightQ_->getCurr4Vec()->Y(), lightQ_->getCurr4Vec()->Z(), lightQ_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightQBar_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightQBar_->getCurr4Vec()->X(), lightQBar_->getCurr4Vec()->Y(), lightQBar_->getCurr4Vec()->Z(), lightQBar_->getCurr4Vec()->E()), math::XYZPoint()));


    fittedBBar_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(bBar_->getCurr4Vec()->X(), bBar_->getCurr4Vec()->Y(), bBar_->getCurr4Vec()->Z(), bBar_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightP_   = pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightP_->getCurr4Vec()->X(), lightP_->getCurr4Vec()->Y(), lightP_->getCurr4Vec()->Z(), lightP_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightPBar_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightPBar_->getCurr4Vec()->X(), lightPBar_->getCurr4Vec()->Y(), lightPBar_->getCurr4Vec()->Z(), lightPBar_->getCurr4Vec()->E()), math::XYZPoint()));
  }
  return fitter_->getStatus();
}

/// add kin fit information to the old event solution (in for legacy reasons)
TtHadEvtSolution 
TtFullHadKinFitter::addKinFitInfo(TtHadEvtSolution * asol) 
{
  TtHadEvtSolution fitsol(*asol);

  std::vector<pat::Jet> jets;
  jets.resize(6);
  jets[TtFullHadEvtPartons::LightQ   ] = fitsol.getCalHadp();
  jets[TtFullHadEvtPartons::LightQBar] = fitsol.getCalHadq();
  jets[TtFullHadEvtPartons::B        ] = fitsol.getCalHadb();
  jets[TtFullHadEvtPartons::LightP   ] = fitsol.getCalHadj();
  jets[TtFullHadEvtPartons::LightPBar] = fitsol.getCalHadk();
  jets[TtFullHadEvtPartons::BBar     ] = fitsol.getCalHadbbar();

  fit( jets);

  // add fitted information to the solution
  if (fitter_->getStatus() == 0) {
    // finally fill the fitted particles
    fitsol.setFitHadb(fittedB_);
    fitsol.setFitHadp(fittedLightQ_);
    fitsol.setFitHadq(fittedLightQBar_);
    fitsol.setFitHadk(fittedLightP_);
    fitsol.setFitHadj(fittedLightPBar_);
    fitsol.setFitHadbbar(fittedBBar_);

    // store the fit's chi2 probability
    fitsol.setProbChi2( fitProb() );
  }
  return fitsol;
}


/// default constructor  
TtFullHadKinFitter::KinFit::KinFit() :
  useBTagging_(true),
  bTags_(2),
  bTagAlgo_("trackCountingHighPurBJetTags"),
  minBTagValueBJet_(3.41),
  maxBTagValueNonBJet_(3.41),
  udscResolutions_(std::vector<edm::ParameterSet>(0)),
  bResolutions_(std::vector<edm::ParameterSet>(0)),
  jetEnergyResolutionScaleFactors_(0),
  jetEnergyResolutionEtaBinning_(0),
  jetCorrectionLevel_("L3Absolute"),
  maxNJets_(-1),
  maxNComb_(1),
  maxNrIter_(500),
  maxDeltaS_(5e-5),
  maxF_(0.0001),
  jetParam_(1),
  mW_(80.4),
  mTop_(173.),
  useOnlyMatch_(false),
  match_(std::vector<int>(0)),
  invalidMatch_(false)
{
  constraints_.push_back(1);
  constraints_.push_back(2);
  constraints_.push_back(5);
}

/// special constructor  
TtFullHadKinFitter::KinFit::KinFit(bool useBTagging, unsigned int bTags, std::string bTagAlgo, double minBTagValueBJet, double maxBTagValueNonBJet,
				   const std::vector<edm::ParameterSet>& udscResolutions, const std::vector<edm::ParameterSet>& bResolutions,
				   const std::vector<double>& jetEnergyResolutionScaleFactors, const std::vector<double>& jetEnergyResolutionEtaBinning,
				   std::string jetCorrectionLevel, int maxNJets, int maxNComb,
				   unsigned int maxNrIter, double maxDeltaS, double maxF, unsigned int jetParam, const std::vector<unsigned>& constraints, double mW, double mTop) :
  useBTagging_(useBTagging),
  bTags_(bTags),
  bTagAlgo_(bTagAlgo),
  minBTagValueBJet_(minBTagValueBJet),
  maxBTagValueNonBJet_(maxBTagValueNonBJet),
  udscResolutions_(udscResolutions),
  bResolutions_(bResolutions),
  jetEnergyResolutionScaleFactors_(jetEnergyResolutionScaleFactors),
  jetEnergyResolutionEtaBinning_(jetEnergyResolutionEtaBinning),
  jetCorrectionLevel_(jetCorrectionLevel),
  maxNJets_(maxNJets),
  maxNComb_(maxNComb),
  maxNrIter_(maxNrIter),
  maxDeltaS_(maxDeltaS),
  maxF_(maxF),
  jetParam_(jetParam),
  constraints_(constraints),
  mW_(mW),
  mTop_(mTop),
  useOnlyMatch_(false),
  invalidMatch_(false)
{
  // define kinematic fit interface
  fitter = new TtFullHadKinFitter(param(jetParam_), maxNrIter_, maxDeltaS_, maxF_, TtFullHadKinFitter::KinFit::constraints(constraints_), mW_, mTop_,
				  &udscResolutions_, &bResolutions_, &jetEnergyResolutionScaleFactors_, &jetEnergyResolutionEtaBinning_);
}

/// default destructor  
TtFullHadKinFitter::KinFit::~KinFit()
{
  delete fitter;
}    

bool
TtFullHadKinFitter::KinFit::doBTagging(const std::vector<pat::Jet>& jets, const unsigned int& bJetCounter, std::vector<int>& combi){
  
  if( !useBTagging_ ) {
    return true;
  }
  if( bTags_ == 2 &&
      jets[combi[TtFullHadEvtPartons::B        ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
      jets[combi[TtFullHadEvtPartons::BBar     ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
      jets[combi[TtFullHadEvtPartons::LightQ   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
      jets[combi[TtFullHadEvtPartons::LightQBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
      jets[combi[TtFullHadEvtPartons::LightP   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
      jets[combi[TtFullHadEvtPartons::LightPBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ ) {
    return true;
  }
  else if( bTags_ == 1 ){  
    if( bJetCounter == 1 &&
        (jets[combi[TtFullHadEvtPartons::B        ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ ||
         jets[combi[TtFullHadEvtPartons::BBar     ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_) &&
	 jets[combi[TtFullHadEvtPartons::LightQ   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	 jets[combi[TtFullHadEvtPartons::LightQBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	 jets[combi[TtFullHadEvtPartons::LightP   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	 jets[combi[TtFullHadEvtPartons::LightPBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ ) {
      return true;
    }
    else if( bJetCounter > 1 &&
	     jets[combi[TtFullHadEvtPartons::B        ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
	     jets[combi[TtFullHadEvtPartons::BBar     ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightQ   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightQBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightP   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightPBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ ) {
      return true;
    }
  }
  else if( bTags_ == 0 ){  
    if( bJetCounter == 0){
      return true;
    }
    else if( bJetCounter == 1 &&
	     (jets[combi[TtFullHadEvtPartons::B        ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ ||
	      jets[combi[TtFullHadEvtPartons::BBar     ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_) &&
	      jets[combi[TtFullHadEvtPartons::LightQ   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	      jets[combi[TtFullHadEvtPartons::LightQBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	      jets[combi[TtFullHadEvtPartons::LightP   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	      jets[combi[TtFullHadEvtPartons::LightPBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ ) {
      return true;
    }
    else if( bJetCounter > 1 &&
	     jets[combi[TtFullHadEvtPartons::B        ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
	     jets[combi[TtFullHadEvtPartons::BBar     ]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightQ   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightQBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightP   ]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ &&
	     jets[combi[TtFullHadEvtPartons::LightPBar]].bDiscriminator(bTagAlgo_) <  maxBTagValueNonBJet_ ) {
      return true;
    }
  }
  else if( bTags_ > 2 ){
    throw cms::Exception("Configuration")
      << "Wrong number of bTags (" << bTags_ << " bTags not supported)!\n";
    return true;
  }
  return false;
}

/// helper function to construct the proper corrected jet for its corresponding quarkType
pat::Jet
TtFullHadKinFitter::KinFit::corJet(const pat::Jet& jet, const std::string& quarkType)
{
  // jetCorrectionLevel was not configured
  if(jetCorrectionLevel_.empty())
    throw cms::Exception("Configuration")
      << "Unconfigured jetCorrectionLevel. Please use an appropriate, non-empty string.\n";

  // quarkType is unknown
  if( !(quarkType=="wMix" ||
	quarkType=="uds" ||
	quarkType=="charm" ||
	quarkType=="bottom") )
    throw cms::Exception("Configuration")
      << quarkType << " is unknown as a quarkType for the jetCorrectionLevel.\n";

  float jecFactor = 1.;
  if(quarkType=="wMix") jecFactor = 0.75 * jet.jecFactor(jetCorrectionLevel_, "uds") + 0.25 * jet.jecFactor(jetCorrectionLevel_, "charm");
  else jecFactor = jet.jecFactor(jetCorrectionLevel_, quarkType);

  pat::Jet ret = jet;
  ret.setP4(ret.p4()*jecFactor);
  return ret;
}

std::list<TtFullHadKinFitter::KinFitResult> 
TtFullHadKinFitter::KinFit::fit(const std::vector<pat::Jet>& jets){

  std::list<TtFullHadKinFitter::KinFitResult>  fitResults;

  /**
   // --------------------------------------------------------
   // skip events with less jets than partons or invalid match
   // --------------------------------------------------------
  **/

  if( jets.size()<nPartons || invalidMatch_ ) {
    // indices referring to the jet combination
    std::vector<int> invalidCombi;
    for(unsigned int i = 0; i < nPartons; ++i) invalidCombi.push_back( -1 );
    
    KinFitResult result;
    // status of the fitter
    result.Status   = -1;
    // chi2
    result.Chi2     = -1.;
    // chi2 probability
    result.Prob     = -1.;
    // the kinFit getters return empty objects here
    result.B        = fitter->fittedB();
    result.BBar     = fitter->fittedBBar();
    result.LightQ   = fitter->fittedLightQ();
    result.LightQBar= fitter->fittedLightQBar();
    result.LightP   = fitter->fittedLightP();
    result.LightPBar= fitter->fittedLightPBar();
    result.JetCombi = invalidCombi;
    // push back fit result
    fitResults.push_back( result );
    return fitResults;
  }

  /**
     analyze different jet combinations using the KinFitter
     (or only a given jet combination if useOnlyMatch=true)
  **/

  std::vector<int> jetIndices;
  if(!useOnlyMatch_) {
    for(unsigned int idx=0; idx<jets.size(); ++idx){
      if(maxNJets_>=(int)nPartons && maxNJets_==(int)idx) break;
      jetIndices.push_back(idx);
    }
  }
  
  std::vector<int> combi;
  for(unsigned int idx=0; idx<nPartons; ++idx) {
    useOnlyMatch_?combi.push_back(match_[idx]):combi.push_back(idx);
  }

  
  unsigned int bJetCounter = 0;
  for(std::vector<pat::Jet>::const_iterator jet = jets.begin(); jet < jets.end(); ++jet){
    if(jet->bDiscriminator(bTagAlgo_) >= minBTagValueBJet_) ++bJetCounter;
  }

  do{
    for(int cnt=0; cnt<TMath::Factorial(combi.size()); ++cnt){
      // take into account indistinguishability of the two jets from the two W decays,
      // and the two decay branches, this reduces the combinatorics by a factor of 2*2*2
      if( ((combi[TtFullHadEvtPartons::LightQ] < combi[TtFullHadEvtPartons::LightQBar] &&
	    combi[TtFullHadEvtPartons::LightP] < combi[TtFullHadEvtPartons::LightPBar] &&
	    combi[TtFullHadEvtPartons::B]      < combi[TtFullHadEvtPartons::BBar]    ) ||
	   useOnlyMatch_) && doBTagging(jets, bJetCounter, combi) ) {

	std::vector<pat::Jet> jetCombi;
	jetCombi.resize(nPartons);
	jetCombi[TtFullHadEvtPartons::LightQ   ] = corJet(jets[combi[TtFullHadEvtPartons::LightQ   ]], "wMix");
	jetCombi[TtFullHadEvtPartons::LightQBar] = corJet(jets[combi[TtFullHadEvtPartons::LightQBar]], "wMix");
	jetCombi[TtFullHadEvtPartons::B        ] = corJet(jets[combi[TtFullHadEvtPartons::B        ]], "bottom");
	jetCombi[TtFullHadEvtPartons::BBar     ] = corJet(jets[combi[TtFullHadEvtPartons::BBar     ]], "bottom");
	jetCombi[TtFullHadEvtPartons::LightP   ] = corJet(jets[combi[TtFullHadEvtPartons::LightP   ]], "wMix");
	jetCombi[TtFullHadEvtPartons::LightPBar] = corJet(jets[combi[TtFullHadEvtPartons::LightPBar]], "wMix");
	  
	// do the kinematic fit
	int status = fitter->fit(jetCombi);
	  
	if( status == 0 ) { 
	  // fill struct KinFitResults if converged
	  TtFullHadKinFitter::KinFitResult result;
	  result.Status   = status;
	  result.Chi2     = fitter->fitS();
	  result.Prob     = fitter->fitProb();
	  result.B        = fitter->fittedB();
	  result.BBar     = fitter->fittedBBar();
	  result.LightQ   = fitter->fittedLightQ();
	  result.LightQBar= fitter->fittedLightQBar();
	  result.LightP   = fitter->fittedLightP();
	  result.LightPBar= fitter->fittedLightPBar();
	  result.JetCombi = combi;
	  // push back fit result
	  fitResults.push_back( result );
	}
      }
      // don't go through combinatorics if useOnlyMatch was chosen
      if(useOnlyMatch_){
	break; 
      }
      // next permutation
      std::next_permutation( combi.begin(), combi.end() );
    }
    // don't go through combinatorics if useOnlyMatch was chosen
    if(useOnlyMatch_){
      break;
    }
  }
  while( stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ) );


  // sort results w.r.t. chi2 values
  fitResults.sort();

  /**
     feed out result starting with the 
     JetComb having the smallest chi2
  **/

  if( (unsigned)fitResults.size() < 1 ) { 
    // in case no fit results were stored in the list (i.e. when all fits were aborted)

    KinFitResult result;
    // status of the fitter
    result.Status   = -1;
    // chi2
    result.Chi2     = -1.;
    // chi2 probability
    result.Prob     = -1.;
    // the kinFit getters return empty objects here
    result.B        = fitter->fittedB();
    result.BBar     = fitter->fittedBBar();
    result.LightQ   = fitter->fittedLightQ();
    result.LightQBar= fitter->fittedLightQBar();
    result.LightP   = fitter->fittedLightP();
    result.LightPBar= fitter->fittedLightPBar();
    // indices referring to the jet combination
    std::vector<int> invalidCombi(nPartons, -1);
    result.JetCombi = invalidCombi;
    // push back fit result
    fitResults.push_back( result );
  }
  return fitResults;
}

TtFullHadKinFitter::Param 
TtFullHadKinFitter::KinFit::param(unsigned int configParameter) 
{
  TtFullHadKinFitter::Param result;
  switch(configParameter){
  case TtFullHadKinFitter::kEMom       : result=TtFullHadKinFitter::kEMom;       break;
  case TtFullHadKinFitter::kEtEtaPhi   : result=TtFullHadKinFitter::kEtEtaPhi;   break;
  case TtFullHadKinFitter::kEtThetaPhi : result=TtFullHadKinFitter::kEtThetaPhi; break;
  default: 
    throw cms::Exception("Configuration") 
      << "Chosen jet parametrization is not supported: " << configParameter << "\n";
    break;
  }
  return result;
} 

TtFullHadKinFitter::Constraint 
TtFullHadKinFitter::KinFit::constraint(unsigned configParameter) 
{
  TtFullHadKinFitter::Constraint result;
  switch(configParameter){
  case TtFullHadKinFitter::kWPlusMass      : result=TtFullHadKinFitter::kWPlusMass;      break;
  case TtFullHadKinFitter::kWMinusMass     : result=TtFullHadKinFitter::kWMinusMass;     break;
  case TtFullHadKinFitter::kTopMass        : result=TtFullHadKinFitter::kTopMass;        break;
  case TtFullHadKinFitter::kTopBarMass     : result=TtFullHadKinFitter::kTopBarMass;     break;
  case TtFullHadKinFitter::kEqualTopMasses : result=TtFullHadKinFitter::kEqualTopMasses; break;
  default: 
    throw cms::Exception("Configuration") 
      << "Chosen fit constraint is not supported: " << configParameter << "\n";
    break;
  }
  return result;
} 

std::vector<TtFullHadKinFitter::Constraint>
TtFullHadKinFitter::KinFit::constraints(const std::vector<unsigned>& configParameters)
{
  std::vector<TtFullHadKinFitter::Constraint> result;
  for(unsigned i=0; i<configParameters.size(); ++i){
    result.push_back(constraint(configParameters[i]));
  }
  return result; 
}

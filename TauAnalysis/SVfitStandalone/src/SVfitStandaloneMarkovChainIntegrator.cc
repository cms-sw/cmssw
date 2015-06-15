#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneMarkovChainIntegrator.h"

#include <TMath.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <assert.h>

enum { kMetropolis };

enum { kUniform, kGaus, kNone };

namespace
{
  double square(double x)
  {
    return x*x;
  }

  template <typename T>
  std::string format_vT(const std::vector<T>& vT)
  {
    std::ostringstream os;
    
    os << "{ ";
    
    unsigned numEntries = vT.size();
    for ( unsigned iEntry = 0; iEntry < numEntries; ++iEntry ) {
      os << vT[iEntry];
      if ( iEntry < (numEntries - 1) ) os << ", ";
    }
    
    os << " }";
    
    return os.str();
  }
  
  std::string format_vdouble(const std::vector<double>& vd)
  {
    return format_vT(vd);
  }
}

SVfitStandaloneMarkovChainIntegrator::SVfitStandaloneMarkovChainIntegrator(const std::string& initMode, 
									   unsigned numIterBurnin, unsigned numIterSampling, unsigned numIterSimAnnealingPhase1, unsigned numIterSimAnnealingPhase2,
									   double T0, double alpha, 
									   unsigned numChains, unsigned numBatches,
									   double L, double epsilon0, double nu,
									   int verbose)
  : name_(""),
    integrand_(0),
    startPosition_and_MomentumFinder_(0),
    x_(0),
    useVariableEpsilon0_(false),
    numIntegrationCalls_(0),
    numMovesTotal_accepted_(0),
    numMovesTotal_rejected_(0)
{
  moveMode_ = kMetropolis;

  if      ( initMode == "uniform" ) initMode_ = kUniform;
  else if ( initMode == "Gaus"    ) initMode_ = kGaus;
  else if ( initMode == "none"    ) initMode_ = kNone;
  else {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "Invalid Configuration Parameter 'initMode' = " << initMode << ","
	      << " expected to be either \"uniform\", \"Gaus\" or \"none\" --> ABORTING !!\n";
    assert(0);
  }

//--- get parameters defining number of "stochastic moves" performed per integration
  numIterBurnin_ = numIterBurnin;
  numIterSampling_ = numIterSampling;

//--- get parameters defining maximum number of attempts to find a valid starting-position for the Markov Chain
  maxCallsStartingPos_ = 1000000;

//--- get parameters defining "simulated annealing" stage at beginning of integration
  numIterSimAnnealingPhase1_ = numIterSimAnnealingPhase1;
  numIterSimAnnealingPhase2_ = numIterSimAnnealingPhase2;
  numIterSimAnnealingPhase1plus2_ = numIterSimAnnealingPhase1_ + numIterSimAnnealingPhase2_;
  if ( numIterSimAnnealingPhase1plus2_ > numIterBurnin_ ) {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "Invalid Configuration Parameters 'numIterSimAnnealingPhase1' = " << numIterSimAnnealingPhase1_ << ","
	      << " 'numIterSimAnnealingPhase2' = " << numIterSimAnnealingPhase2_ << ","
	      << " sim. Annealing and Sampling stages must not overlap --> ABORTING !!\n";
    assert(0);
  }
  T0_ = T0;
  sqrtT0_ = TMath::Sqrt(T0_);
  alpha_ = alpha;
  if ( !(alpha_ > 0. && alpha_ < 1.) ) {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "Invalid Configuration Parameter 'alpha' = " << alpha_ << "," 
	      << " value within interval ]0..1[ expected --> ABORTING !!\n";
    assert(0);
  }
  alpha2_ = square(alpha_);
  
//--- get parameter specifying how many Markov Chains are run in parallel
  numChains_ = numChains;
  if ( numChains_ == 0 ) {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "Invalid Configuration Parameter 'numChains' = " << numChains_ << "," 
	      << " value greater 0 expected --> ABORTING !!\n";
    assert(0);
  }

  numBatches_ = numBatches;
  if ( numBatches_ == 0 ) {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "Invalid Configuration Parameter 'numBatches' = " << numBatches_ << "," 
	      << " value greater 0 expected --> ABORTING !!\n";
    assert(0);
  }
  if ( (numIterSampling_ % numBatches_) != 0 ) {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "Invalid Configuration Parameter 'numBatches' = " << numBatches_ << "," 
	      << " factor of numIterSampling = " << numIterSampling_ << " expected --> ABORTING !!\n";
    assert(0);
  }
  
//--- get parameters specific to "dynamic moves" 
  L_ = L;
  epsilon0_ = epsilon0;
  nu_ = nu;

  verbose_ = verbose;
}

SVfitStandaloneMarkovChainIntegrator::~SVfitStandaloneMarkovChainIntegrator()
{
  if ( verbose_ >= 0 ) {
    std::cout << "<SVfitStandaloneMarkovChainIntegrator::~SVfitStandaloneMarkovChainIntegrator>:" << std::endl;
    std::cout << " integration calls = " << numIntegrationCalls_ << std::endl;
    std::cout << " moves: accepted = " << numMovesTotal_accepted_ << ", rejected = " << numMovesTotal_rejected_ 
	      << " (fraction = " << (double)numMovesTotal_accepted_/(numMovesTotal_accepted_ + numMovesTotal_rejected_)*100. 
	      << "%)" << std::endl;
  }

  delete [] x_;
}

void SVfitStandaloneMarkovChainIntegrator::setIntegrand(const ROOT::Math::Functor& integrand)
{
  integrand_ = &integrand;
  numDimensions_ = integrand.NDim();

  delete [] x_;
  x_ = new double[numDimensions_];

  xMin_.resize(numDimensions_); 
  xMax_.resize(numDimensions_);  

  dqDerr_.resize(numDimensions_); 
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    dqDerr_[iDimension] = 1.e-6;
  }

  if ( useVariableEpsilon0_ ) {
    if ( epsilon0s_.size() != numDimensions_ ) {
      std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
		<< "Mismatch in dimensionality between integrand = " << numDimensions_
		<< " and Configuration Parameter 'epsilon0' = " << epsilon0s_.size() << " --> ABORTING !!\n";
      assert(0);
    }
  } else {
    epsilon0s_.resize(numDimensions_); 
    for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
      epsilon0s_[iDimension] = epsilon0_;
    }
  }

  p_.resize(2*numDimensions_);   // first N entries = "significant" components, last N entries = "dummy" components
  q_.resize(numDimensions_);     // "potential energy" E(q) depends in the first N "significant" components only
  gradE_.resize(numDimensions_); 
  prob_ = 0.;

  u_.resize(2*numDimensions_);   // first N entries = "significant" components, last N entries = "dummy" components
  pProposal_.resize(numDimensions_);
  qProposal_.resize(numDimensions_);

  probSum_.resize(numChains_*numBatches_);  
  for ( vdouble::iterator probSum_i = probSum_.begin();
	probSum_i != probSum_.end(); ++probSum_i ) {
    (*probSum_i) = 0.;
  }
  integral_.resize(numChains_*numBatches_);  
}

void SVfitStandaloneMarkovChainIntegrator::setStartPosition_and_MomentumFinder(const ROOT::Math::Functor& startPosition_and_MomentumFinder)
{
  startPosition_and_MomentumFinder_ = &startPosition_and_MomentumFinder;
}

void SVfitStandaloneMarkovChainIntegrator::registerCallBackFunction(const ROOT::Math::Functor& function)
{
  callBackFunctions_.push_back(&function);
}

void SVfitStandaloneMarkovChainIntegrator::integrate(const std::vector<double>& xMin, const std::vector<double>& xMax, 
						     double& integral, double& integralErr, int& errorFlag)
{
  if ( verbose_ >= 2 ) {
    std::cout << "<SVfitStandaloneMarkovChainIntegrator::integrate>:" << std::endl;
    std::cout << " numDimensions = " << numDimensions_ << std::endl;
  }

  if ( !integrand_ ) {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "No integrand function has been set yet --> ABORTING !!\n";
    assert(0);
  }

  if ( !(xMin.size() == numDimensions_ && xMax.size() == numDimensions_) ) {
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
      << "Mismatch in dimensionality between integrand = " << numDimensions_
      << " and integration limits = " << xMin.size() << "/" << xMax.size() << " --> ABORTING !!\n";
    assert(0);
  }
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    xMin_[iDimension] = xMin[iDimension];
    xMax_[iDimension] = xMax[iDimension];
    if ( verbose_ >= 2 ) {
      std::cout << "dimension #" << iDimension << ": min = " << xMin_[iDimension] << ", max = " << xMax_[iDimension] << std::endl;
    }
  }
  
//--- CV: set random number generator used to initialize starting-position
//        for each integration, in order to make integration results independent of processing history
  rnd_.SetSeed(12345);

  numMoves_accepted_ = 0;
  numMoves_rejected_ = 0;

  unsigned k = numChains_*numBatches_;  
  unsigned m = numIterSampling_/numBatches_;

  numChainsRun_ = 0; 

  for ( unsigned iChain = 0; iChain < numChains_; ++iChain ) {
    bool isValidStartPos = false;
    if ( initMode_ == kNone ) {
      prob_ = evalProb(q_);
      if ( prob_ > 0. ) {
	bool isWithinBounds = true;
	for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
	  double q_i = q_[iDimension];
	  if ( !(q_i > 0. && q_i < 1.) ) isWithinBounds = false;
	}
	if ( isWithinBounds ) {
	  isValidStartPos = true;
	} else {
	  if ( verbose_ >= 1 ) {
	    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
		      << "Warning: Requested start-position = " << format_vdouble(q_) << " not within interval ]0..1[ --> searching for valid alternative !!\n";
	  }
	}
      } else {
	if ( verbose_ >= 1 ) {
	  std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
		    << "Warning: Requested start-position = " << format_vdouble(q_) << " returned probability zero --> searching for valid alternative !!";
	}
      }
    }    
    unsigned iTry = 0;
    while ( !isValidStartPos && iTry < maxCallsStartingPos_ ) {
      initializeStartPosition_and_Momentum();
//--- CV: check if start-position is within "valid" (physically allowed) region 
      bool isWithinPhysicalRegion = true;
      if ( startPosition_and_MomentumFinder_ ) {
	updateX(q_);
	isWithinPhysicalRegion = ((*startPosition_and_MomentumFinder_)(x_) > 0.5);
      }
      if ( isWithinPhysicalRegion ) {
	prob_ = evalProb(q_);
	if ( prob_ > 0. ) {
	  isValidStartPos = true;
	} else {
	  if ( iTry > 0 && (iTry % 100000) == 0 ) {
	    if ( iTry == 100000 ) std::cout << "<SVfitStandaloneMarkovChainIntegrator::integrate (name = " << name_ << ")>:" << std::endl;
	    std::cout << "try #" << iTry << ": did not find valid start-position yet." << std::endl;
	    //std::cout << "(q = " << format_vdouble(q_) << ", prob = " << prob_ << ")" << std::endl;
	  }
	}
      }
      ++iTry;
    }
    if ( !isValidStartPos ) continue;

    for ( unsigned iMove = 0; iMove < numIterBurnin_; ++iMove ) {
//--- propose Markov Chain transition to new, randomly chosen, point

      //if ( verbose_ >= 2 ) std::cout << "burn-in move #" << iMove << ":" << std::endl;

      bool isAccepted = false;
      bool isValid = true;
      do {
	makeStochasticMove(iMove, isAccepted, isValid);
      } while ( !isValid );
    }

    unsigned idxBatch = iChain*numBatches_;

    for ( unsigned iMove = 0; iMove < numIterSampling_; ++iMove ) {
//--- propose Markov Chain transition to new, randomly chosen, point;
//    evaluate "call-back" functions at this point

      //if ( verbose_ >= 2 ) std::cout << "sampling move #" << iMove << ":" << std::endl;

      bool isAccepted = false;
      bool isValid = true;
      do {
	makeStochasticMove(numIterBurnin_ + iMove, isAccepted, isValid);
      } while ( !isValid );
      if ( isAccepted ) {
	++numMoves_accepted_;
      } else {
	++numMoves_rejected_;
      }

      updateX(q_);
      for ( std::vector<const ROOT::Math::Functor*>::const_iterator callBackFunction = callBackFunctions_.begin();
	    callBackFunction != callBackFunctions_.end(); ++callBackFunction ) {
	(**callBackFunction)(x_);
      }

      if ( iMove > 0 && (iMove % m) == 0 ) ++idxBatch;
      probSum_[idxBatch] += prob_;
    }

    ++numChainsRun_;
  }

  for ( unsigned idxBatch = 0; idxBatch < probSum_.size(); ++idxBatch ) {  
    integral_[idxBatch] = probSum_[idxBatch]/m;
  }

//--- compute integral value and uncertainty
//   (eqs. (6.39) and (6.40) in [1])   
  integral = 0.;
  for ( unsigned i = 0; i < k; ++i ) {    
    integral += integral_[i];
  }
  integral /= k;

  integralErr = 0.;
  for ( unsigned i = 0; i < k; ++i ) {
    integralErr += square(integral_[i] - integral);
  }
  if ( k >= 2 ) integralErr /= (k*(k - 1));
  integralErr = TMath::Sqrt(integralErr);

  //if ( verbose_ >= 1 ) std::cout << "--> returning integral = " << integral << " +/- " << integralErr << std::endl;

  errorFlag = ( numChainsRun_ >= 0.5*numChains_ ) ?
    0 : 1;

  ++numIntegrationCalls_;
  numMovesTotal_accepted_ += numMoves_accepted_;
  numMovesTotal_rejected_ += numMoves_rejected_;
}

void SVfitStandaloneMarkovChainIntegrator::print(std::ostream& stream) const
{
  stream << "<SVfitStandaloneMarkovChainIntegrator::print>:" << std::endl;
  for ( unsigned iChain = 0; iChain < numChains_; ++iChain ) {
    double integral = 0.;
    for ( unsigned iBatch = 0; iBatch < numBatches_; ++iBatch ) {    
      double integral_i = integral_[iChain*numBatches_ + iBatch];
      integral += integral_i;
    }
    integral /= numBatches_;
    
    double integralErr = 0.;
    for ( unsigned iBatch = 0; iBatch < numBatches_; ++iBatch ) { 
      double integral_i = integral_[iChain*numBatches_ + iBatch];
      integralErr += square(integral_i - integral);
    }
    if ( numBatches_ >= 2 ) integralErr /= (numBatches_*(numBatches_ - 1));
    integralErr = TMath::Sqrt(integralErr);

    std::cout << " chain #" << iChain << ": integral = " << integral << " +/- " << integralErr << std::endl;
  }
  std::cout << "moves: accepted = " << numMoves_accepted_ << ", rejected = " << numMoves_rejected_ 
	    << " (fraction = " << (double)numMoves_accepted_/(numMoves_accepted_ + numMoves_rejected_)*100. 
	    << "%)" << std::endl;
}

//
//-------------------------------------------------------------------------------
//

void SVfitStandaloneMarkovChainIntegrator::initializeStartPosition_and_Momentum(const std::vector<double>& q)
{
//--- set start position of Markov Chain in N-dimensional space to given values
  if ( q.size() == numDimensions_ ) {
    for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
      double q_i = q[iDimension];
      if ( q_i > 0. && q_i < 1. ) {
	q_[iDimension] = q_i;
      } else {
	std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
		  << "Invalid start-position coordinates = " << format_vdouble(q) << " --> ABORTING !!\n";
	assert(0);
      }
    }
  } else { 
    std::cerr << "<SVfitStandaloneMarkovChainIntegrator>:"
	      << "Mismatch in dimensionality between integrand = " << numDimensions_
	      << " and vector of start-position coordinates = " << q.size() << " --> ABORTING !!\n";
    assert(0);
  }
}

void SVfitStandaloneMarkovChainIntegrator::initializeStartPosition_and_Momentum()
{
//--- randomly choose start position of Markov Chain in N-dimensional space
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    bool isInitialized = false;
    while ( !isInitialized ) {
      double q0 = 0.;
      if ( initMode_ == kGaus ) q0 = rnd_.Gaus(0.5, 0.5);
      else q0 = rnd_.Uniform(0., 1.);
      if ( q0 > 0. && q0 < 1. ) {
	q_[iDimension] = q0;
	isInitialized = true;
      }
    }
  }

  if ( verbose_ >= 1 ) {
    std::cout << "<SVfitStandaloneMarkovChainIntegrator::initializeStartPosition_and_Momentum>:" << std::endl;
    std::cout << " q = " << format_vdouble(q_) << std::endl;
  }
}

void SVfitStandaloneMarkovChainIntegrator::sampleSphericallyRandom()
{
//--- compute vector of unit length
//    pointing in random direction in N-dimensional space
//
//    NOTE: the algorithm implemented in this function 
//          uses the fact that a N-dimensional Gaussian is spherically symmetric
//         (u is uniformly distributed over the surface of an N-dimensional hypersphere)
//
  double uMag2 = 0.;
  for ( unsigned iDimension = 0; iDimension < 2*numDimensions_; ++iDimension ) {
    double u_i = rnd_.Gaus(0., 1.);
    u_[iDimension] = u_i;
    uMag2 += (u_i*u_i);
  }
  double uMag = TMath::Sqrt(uMag2);
  for ( unsigned iDimension = 0; iDimension < 2*numDimensions_; ++iDimension ) {
    u_[iDimension] /= uMag;
  }
}

void SVfitStandaloneMarkovChainIntegrator::makeStochasticMove(unsigned idxMove, bool& isAccepted, bool& isValid)
{
//--- perform "stochastic" move
//   (eq. 24 in [2])

  //if ( verbose_ >= 2 ) {
  //  std::cout << "<MarkovChainIntegrator::makeStochasticMove>:" << std::endl;
  //  std::cout << " idx = " << idxMove << std::endl;
  //  std::cout << " q = " << format_vdouble(q_) << std::endl;
  //  std::cout << " prob = " << prob_ << std::endl;
  //}

//--- perform random updates of momentum components
  if ( idxMove < numIterSimAnnealingPhase1_ ) {
    for ( unsigned iDimension = 0; iDimension < 2*numDimensions_; ++iDimension ) {
      p_[iDimension] = sqrtT0_*rnd_.Gaus(0., 1.);
    }
  } else if ( idxMove < numIterSimAnnealingPhase1plus2_ ) {
    double pMag2 = 0.;
    for ( unsigned iDimension = 0; iDimension < 2*numDimensions_; ++iDimension ) {
      double p_i = p_[iDimension];
      pMag2 += p_i*p_i;
    }
    double pMag = TMath::Sqrt(pMag2);
    sampleSphericallyRandom();
    for ( unsigned iDimension = 0; iDimension < 2*numDimensions_; ++iDimension ) {
      p_[iDimension] = alpha_*pMag*u_[iDimension] + (1. - alpha2_)*rnd_.Gaus(0., 1.);
    }
  } else {
    //std::cout << "case 3" << std::endl;
    for ( unsigned iDimension = 0; iDimension < 2*numDimensions_; ++iDimension ) {
      p_[iDimension] = rnd_.Gaus(0., 1.);
    }
  }

  //if ( verbose_ >= 2 ) {
  //  std::cout << "p(updated) = " << format_vdouble(p_) << std::endl;
  //}

//--- choose random step size 
  double exp_nu_times_C = 0.;
  do {
    double C = rnd_.BreitWigner(0., 1.);
    exp_nu_times_C = TMath::Exp(nu_*C);
  } while ( TMath::IsNaN(exp_nu_times_C) || !TMath::Finite(exp_nu_times_C) || exp_nu_times_C > 1.e+6 );
  vdouble epsilon(numDimensions_);
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    epsilon[iDimension] = epsilon0s_[iDimension]*exp_nu_times_C;
  }

  //if ( verbose_ >= 2 ) std::cout << "epsilon = " << format_vdouble(epsilon) << std::endl;

  if        ( moveMode_ == kMetropolis ) { // Metropolis algorithm: move according to eq. (27) in [2]
//--- update position components
//    by single step of chosen size in direction of the momentum components
    for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {    
      qProposal_[iDimension] = q_[iDimension] + epsilon[iDimension]*p_[iDimension];
    }
  } else assert(0);

  //if ( verbose_ >= 2 ) std::cout << "q(proposed) = " << format_vdouble(qProposal_) << std::endl;

//--- ensure that proposed new point is within integration region
//   (take integration region to be "cyclic")
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {         
    double q_i = qProposal_[iDimension];
    q_i = q_i - TMath::Floor(q_i);
    assert(q_i >= 0. && q_i <= 1.);
    qProposal_[iDimension] = q_i;
  }

//--- check if proposed move of Markov Chain to new position is accepted or not:
//    compute change in phase-space volume for "dummy" momentum components
//   (eqs. 25 in [2])
  double probProposal = evalProb(qProposal_);

  //if ( verbose_ >= 2 ) std::cout << "prob(proposed) = " << probProposal << std::endl;

  double deltaE = 0.;
  if      ( probProposal > 0. && prob_ > 0. ) deltaE = -TMath::Log(probProposal/prob_);
  else if ( probProposal > 0.               ) deltaE = -std::numeric_limits<double>::max();
  else if (                      prob_ > 0. ) deltaE = +std::numeric_limits<double>::max();
  else assert(0);

  double pAccept = 0.;
  if        ( moveMode_ == kMetropolis ) { // Metropolis algorithm: move according to eq. (13) in [2]

    //if ( verbose_ >= 2 ) std::cout << " deltaE = " << deltaE << std::endl;

    pAccept = TMath::Exp(-deltaE);
  } else assert(0);

  //if ( verbose_ >= 2 ) std::cout << "p(accept) = " << pAccept << std::endl;

  double u = rnd_.Uniform(0., 1.);

  //if ( verbose_ >= 2 ) std::cout << "u = " << u << std::endl;
  
  if ( u < pAccept ) {
    //if ( verbose_ >= 2 ) std::cout << "move accepted." << std::endl;
    for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {    
      q_[iDimension] = qProposal_[iDimension];
    }
    prob_ = evalProb(q_);
    isAccepted = true;
  } else {
    //if ( verbose_ >= 2 ) std::cout << "move rejected." << std::endl;
    isAccepted = false;
  }
}

void SVfitStandaloneMarkovChainIntegrator::updateX(const std::vector<double>& q)
{
  //std::cout << "<MarkovChainIntegrator::updateX>:" << std::endl;
  //std::cout << " q = " << format_vdouble(q) << std::endl;
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    double q_i = q[iDimension];
    x_[iDimension] = (1. - q_i)*xMin_[iDimension] + q_i*xMax_[iDimension];
    //std::cout << " x[" << iDimension << "] = " << x_[iDimension] << " ";
    //std::cout << "(xMin[" << iDimension << "] = " << xMin_[iDimension] << ","
    //          << " xMax[" << iDimension << "] = " << xMax_[iDimension] << ")";
    //std::cout << std::endl;
  }
}

double SVfitStandaloneMarkovChainIntegrator::evalProb(const std::vector<double>& q)
{
  updateX(q);
  double prob = (*integrand_)(x_);
  return prob;
}

double SVfitStandaloneMarkovChainIntegrator::evalE(const std::vector<double>& q)
{
  double prob = evalProb(q);
  double E = -TMath::Log(prob);
  return E;
}

double SVfitStandaloneMarkovChainIntegrator::evalK(const std::vector<double>& p, unsigned idxFirst, unsigned idxLast)
{
//--- compute "kinetic energy"
//   (of either the "significant" or "dummy" momentum components) 
  assert(idxLast <= p.size());
  double K = 0.;
  for ( unsigned iDimension = idxFirst; iDimension < idxLast; ++iDimension ) {
    double p_i = p[iDimension];
    K += (p_i*p_i);
  }
  K *= 0.5;
  return K;
}

void SVfitStandaloneMarkovChainIntegrator::updateGradE(std::vector<double>& q)
{
//--- numerically compute gradient of "potential energy" E = -log(P(q)) at point q
  //if ( verbose_ >= 1 ) {
  //  std::cout << "<MarkovChainIntegrator::updateGradE>:" << std::endl;
  //  std::cout << " q(1) = " << format_vdouble(q) << std::endl;
  //}

  double prob_q = evalProb(q);  
  //if ( verbose_ >= 1 ) std::cout << " prob(q) = " << prob_q << std::endl;

  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    double q_i = q[iDimension];
    double dqDerr_i = dqDerr_[iDimension];
    double dq = ( (q_i + dqDerr_i) < 1. ) ? +dqDerr_i : -dqDerr_i;
    double q_plus_dq = q_i + dq;
    q[iDimension] = q_plus_dq;
    double prob_q_plus_dq = evalProb(q);
    double gradE_i = -(prob_q_plus_dq - prob_q)/dq;
    if ( prob_q > 0. ) gradE_i /= prob_q;
    gradE_[iDimension] = gradE_i;
    q[iDimension] = q_i;
  }

  //if ( verbose_ >= 1 ) {
  //  std::cout << " q(2) = " << format_vdouble(q) << std::endl;
  //  std::cout << "--> gradE = " << format_vdouble(gradE_) << std::endl;
  //}
}


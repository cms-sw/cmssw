/** \class TrajAnnealing
  *  Allow to save all the traj info
  *  for each annealing cycle 
  *  suitable for algorithm like DAF
  *
  *  \author in May 2014 by brondolin 
  */

#ifndef DataFormats_TrajAnnealing_h
#define DataFormats_TrajAnnealing_h

#include <vector>

//class
class TrajAnnealing 
{
public:
    TrajAnnealing():
      traj_(),
      annealing_(0),
      theWeights(){}
    virtual ~TrajAnnealing(){}

    TrajAnnealing( Trajectory, float );

    void Debug() const;
    float getAnnealing() const { return annealing_; }
    Trajectory getTraj() const { return traj_; }

    //vector of weights
    std::vector<float> const & weights() const {return theWeights;}
    std::vector<float>  & weights() {return theWeights;}

private:
  
    Trajectory traj_;
    float annealing_;
    std::vector<float> theWeights;
    TransientTrackingRecHit::RecHitContainer theHits_;

    std::pair<float, std::vector<float> > getAnnealingWeight( const TrackingRecHit& aRecHit ) const ;
};

// this is our new product, it is simply a 
// collection of TrajAnnealing held in an std::vector
typedef std::vector<TrajAnnealing> TrajAnnealingCollection;

#endif

#ifndef MinHitsTrajectoryFilter_H
#define MinHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */

class MinHitsTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MinHitsTrajectoryFilter( int minHits=5, int seedPairPenalty=0):theMinHits( minHits), theSeedPairPenalty(seedPairPenalty) {}
  //  explicit MinHitsTrajectoryFilter( int minHits=-1, int seedPairPenalty=0):theMinHits( minHits), theSeedPairPenalty(seedPairPenalty) {}

  MinHitsTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC): 
   theMinHits(         pset.getParameter<int>("minimumNumberOfHits") ),
   theSeedPairPenalty( pset.getParameter<int>("seedPairPenalty")     )
 {}
    
  virtual bool qualityFilter( const Trajectory& traj) const { return QF<Trajectory>(traj);}
  virtual bool qualityFilter( const TempTrajectory& traj) const { return QF<TempTrajectory>(traj);}

  virtual bool toBeContinued( TempTrajectory&) const { return TrajectoryFilter::toBeContinuedIfNotContributing ;}
  virtual bool toBeContinued( Trajectory&) const { return TrajectoryFilter::toBeContinuedIfNotContributing ;}

  virtual std::string name() const {return "MinHitsTrajectoryFilter";}

  inline edm::ParameterSetDescription getFilledConfigurationDescription() {
    edm::ParameterSetDescription desc;
    desc.add<int>("minimumNumberOfHits",5);
    desc.add<int>("seedPairPenalty",    0);
    return desc;
  }

protected:

  template<class T> bool QF(const T & traj) const{
    int seedPenalty = (2==traj.seedNHits()) ? theSeedPairPenalty: 0;  // increase by one if seed-doublet...
    return (traj.foundHits() >= theMinHits + seedPenalty);
  }

  int theMinHits;
  int theSeedPairPenalty;

};

#endif

#ifndef SeedExtentionTrajectoryFilter_H
#define SeedExtentionTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class SeedExtentionTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit SeedExtentionTrajectoryFilter() {} 
  
  explicit SeedExtentionTrajectoryFilter(edm::ParameterSet const & pset, edm::ConsumesCollector&) :
     theStrict(pset.existsAs<bool>("strictSeedExtention") ? pset.getParameter<bool>("strictSeedExtention"):false),
     theExtention(pset.existsAs<int>("seedExtention") ? pset.getParameter<int>("seedExtention"):0) {}

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "LostHitsFractionTrajectoryFilter";}

private:

  template<class T> bool TBC(const T& traj) const {
     return theStrict? strictTBC(traj) : looseTBC(traj);
  }
  template<class T> bool looseTBC(const T& traj) const;
  template<class T> bool strictTBC(const T& traj) const;


   bool theStrict=false;
   int theExtention = 0;


};

template<class T> bool SeedExtentionTrajectoryFilter::looseTBC(const T& traj) const {
    return (int(traj.measurements().size())>int(traj.seedNHits())+theExtention) | (0==traj.lostHits());
}


// strict case as a real seeding: do not allow even inactive
template<class T> bool SeedExtentionTrajectoryFilter::strictTBC(const T& traj) const {
    return traj.foundHits()>=int(traj.seedNHits())+theExtention;
}




#endif

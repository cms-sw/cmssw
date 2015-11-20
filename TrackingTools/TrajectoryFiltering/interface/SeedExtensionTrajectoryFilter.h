#ifndef SeedExtensionTrajectoryFilter_H
#define SeedExtensionTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class SeedExtensionTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit SeedExtensionTrajectoryFilter() {} 
  
  explicit SeedExtensionTrajectoryFilter(edm::ParameterSet const & pset, edm::ConsumesCollector&) :
     theStrict(pset.existsAs<bool>("strictSeedExtension") ? pset.getParameter<bool>("strictSeedExtension"):false),
     theExtension(pset.existsAs<int>("seedExtension") ? pset.getParameter<int>("seedExtension"):0) {}

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
   int theExtension = 0;


};

template<class T> bool SeedExtensionTrajectoryFilter::looseTBC(const T& traj) const {
    return (int(traj.measurements().size())>int(traj.seedNHits())+theExtension) | (0==traj.lostHits());
}


// strict case as a real seeding: do not allow even inactive
template<class T> bool SeedExtensionTrajectoryFilter::strictTBC(const T& traj) const {
    return traj.foundHits()>=int(traj.seedNHits())+theExtension;
}




#endif

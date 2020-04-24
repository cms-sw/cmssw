#ifndef CompositeTrajectoryFilter_H
#define CompositeTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */

class CompositeTrajectoryFilter : public TrajectoryFilter {
public:

  explicit CompositeTrajectoryFilter(){filters.clear();}
  explicit CompositeTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC)
  {
    //look for VPSet of filters
    std::vector<edm::ParameterSet> vpset=pset.getParameter<std::vector<edm::ParameterSet> >("filters");
    for (unsigned int i=0;i!= vpset.size();i++)
      {filters.emplace_back(TrajectoryFilterFactory::get()->create(vpset[i].getParameter<std::string>("ComponentType"),
                                                                   vpset[i], iC));}
  }
  
  ~CompositeTrajectoryFilter() override {}

  void setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    for(auto& f: filters) {
      f->setEvent(iEvent, iSetup);
    }
  }

  bool qualityFilter( const Trajectory& traj) const  override{ return QF<Trajectory>(traj);}
  bool qualityFilter( const TempTrajectory& traj) const  override{ return QF<TempTrajectory>(traj);}
 
  bool toBeContinued( Trajectory& traj) const  override{ return TBC<Trajectory>(traj);}
  bool toBeContinued( TempTrajectory& traj) const  override{ return TBC<TempTrajectory>(traj);}
  
  std::string name() const  override{ std::string rname="CompositeTrajectoryFilter";
    unsigned int i=0;
    unsigned int n=filters.size();
    for (;i<n;i++){ rname+="_"+filters[i]->name();}
    return rname;
  }

protected:
  template <class T> bool TBC( T& traj)const{
    unsigned int i=0;
    unsigned int n=filters.size();
    for (;i<n;i++){ if (!filters[i]->toBeContinued(traj)) return false; }
    return true;}

  template <class T> bool QF(const T& traj)const{
    unsigned int i=0;
    unsigned int n=filters.size();
    for (;i<n;i++){ if (!filters[i]->qualityFilter(traj)) return false; }
    return true;}
 protected:
  std::vector<std::unique_ptr<TrajectoryFilter> > filters;
  
};

#endif

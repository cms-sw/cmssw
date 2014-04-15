#ifndef CompositeLogicalTrajectoryFilter_H
#define CompositeLogicalTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class CompositeLogicalTrajectoryFilter : public TrajectoryFilter {
public:
  enum logic { OR, AND };
  explicit CompositeLogicalTrajectoryFilter() {filters.clear();}

  explicit CompositeLogicalTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC)
  {
  //look for VPSet of filters
  std::vector<edm::ParameterSet> vpset=pset.getParameter<std::vector<edm::ParameterSet> >("filters");
  for (unsigned int i=0;i!= vpset.size();i++)
    {
      std::string ls = vpset[i].getParameter<std::string>("logic");
      logic l=OR;
      if (ls == "OR") l=OR;
      else if (ls == "AND") l=AND;
      else{
	edm::LogError("CompositeLogicalTrajectoryFilter")<<"I don't understand the logic: "<<ls
	  ;
      }
      filters.emplace_back(l, std::unique_ptr<TrajectoryFilter>(TrajectoryFilterFactory::get()->create(vpset[i].getParameter<std::string>("ComponentName"), vpset[i], iC)));
    
    }
  }
  
  ~CompositeLogicalTrajectoryFilter() {}

  void setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    for(auto& item: filters) {
      item.second->setEvent(iEvent, iSetup);
    }
  }

  virtual bool qualityFilter( const Trajectory& traj) const { return QF<Trajectory>(traj);}
  virtual bool qualityFilter( const TempTrajectory& traj) const { return QF<TempTrajectory>(traj);}
 
  virtual bool toBeContinued( Trajectory& traj) const { return TBC<Trajectory>(traj);}
  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  
  virtual std::string name() const {return "CompositeLogicalTrajectoryFilter";}

protected:
  template <class T> bool TBC( T& traj)const{
    unsigned int i=0;
    unsigned int n=filters.size();
    for (;i<n;i++){ if (!filters[i].second->toBeContinued(traj)) return false; }
    return true;
  }

  template <class T> bool QF(const T& traj)const{
    bool condition=true;

    unsigned int n=filters.size();
    if (n==0) { edm::LogError("CompositeLogicalTrajectoryFilter")<<n<<" filters !." ; return false;}
    condition=filters[0].second->qualityFilter(traj);

    unsigned int i=1;
    for (;i<n;i++){ 
      bool lcondition =filters[i].second->qualityFilter(traj);
      if (filters[i].first==OR)
	condition= condition || lcondition;
      else if (filters[i].first==AND)
	condition= condition && lcondition;
    }  
    return condition;   
  }

 protected:
  std::vector< std::pair< logic, std::unique_ptr<TrajectoryFilter> > > filters;
};

#endif

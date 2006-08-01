#ifndef MultiTrajectoryStateMerger_H
#define MultiTrajectoryStateMerger_H


class TrajectoryStateOnSurface;

/** Abstract base class for trimming or merging a MultiTrajectoryState into 
 *  one with a smaller number of components.
 */

class MultiTrajectoryStateMerger {

public:

  virtual TrajectoryStateOnSurface merge(const TrajectoryStateOnSurface& tsos) const = 0;
  virtual ~MultiTrajectoryStateMerger() {}
  virtual MultiTrajectoryStateMerger* clone() const = 0;
 protected:

  MultiTrajectoryStateMerger() {}

};  

#endif

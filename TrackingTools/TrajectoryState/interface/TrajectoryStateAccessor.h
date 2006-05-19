#ifndef TrajectoryStateAccessor_H
#define TrajectoryStateAccessor_H

class FreeTrajectoryState;

/** Helper class to obtain the uncertainty on specific
 *  trajectory parameters.
 */

class TrajectoryStateAccessor {
public:

  TrajectoryStateAccessor( const FreeTrajectoryState& fts) :
    theFts(fts) {}

  double inversePtError() const;

private:

  const FreeTrajectoryState& theFts;

};

#endif

#ifndef TkDetLayers_DetGroupMerger_h
#define TkDetLayers_DetGroupMerger_h

#include "TrackingTools/DetLayers/interface/DetGroup.h"

class DetGroupMerger {
public:

  vector<DetGroup> orderAndMergeTwoLevels( const vector<DetGroup>& one,  
					   const vector<DetGroup>& two,
					   int firstIndex, 
					   int firstCrossed) const;

  vector<DetGroup> mergeTwoLevels( const vector<DetGroup>& one,  const vector<DetGroup>& two) const;

  void addSameLevel( const vector<DetGroup>& gvec, vector<DetGroup>& result) const;

  void doubleIndexSize( vector<DetGroup>& vec) const;

  void incrementAndDoubleSize( vector<DetGroup>& vec) const;

};

#endif

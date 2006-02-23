#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"

vector<DetGroup> 
DetGroupMerger::orderAndMergeTwoLevels( const vector<DetGroup>& one,  
					const vector<DetGroup>& two,
					int firstIndex, 
					int firstCrossed) const{
  cout << "dummy implementation of DetGroupMerger::orderAndMergeTwoLevels" << endl;
  return vector<DetGroup>();
}

vector<DetGroup> 
DetGroupMerger::mergeTwoLevels( const vector<DetGroup>& one,  const vector<DetGroup>& two) const{
  cout << "dummy implementation of DetGroupMerger::mergeTwoLevels" << endl;
  return vector<DetGroup>();


}

void 
DetGroupMerger::addSameLevel( const vector<DetGroup>& gvec, vector<DetGroup>& result) const{
  cout << "dummy implementation of DetGroupMerger::addSameLevel" << endl;
}

void 
DetGroupMerger::doubleIndexSize( vector<DetGroup>& vec) const{
  cout << "dummy implementation of DetGroupMerger::doubleIndexSize" << endl;
}

void 
DetGroupMerger::incrementAndDoubleSize( vector<DetGroup>& vec) const{
  cout << "dummy implementation of DetGroupMerger::incrementAndDoubleSize" << endl;
}

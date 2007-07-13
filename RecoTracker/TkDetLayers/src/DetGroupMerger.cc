#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;


DetGroupMerger::orderAndMergeTwoLevels( const vector<DetGroup>& one,  
					const vector<DetGroup>& two,
					std::vector<DetGroup>& result,
					int firstIndex, 
					int firstCrossed) const{
  if (one.empty() && two.empty()) return;


  if (one.empty()) {
    result = two;
    if (firstIndex == firstCrossed) incrementAndDoubleSize(result); 
    else                            doubleIndexSize(result);
  }
  else if (two.empty()) {
    result = one;
    if (firstIndex == firstCrossed) doubleIndexSize(result);
    else                            incrementAndDoubleSize(result);
  }
  else { // both are not empty
    if (firstIndex == firstCrossed) mergeTwoLevels( one, two,result);
    else                            mergeTwoLevels( two, one, result);
  }
}


DetGroupMerger::mergeTwoLevels( const vector<DetGroup>& one,  const vector<DetGroup>& two, std::vector<DetGroup>& result) const{

  result.reserve( one.size() + two.size());

  int indSize1 = one.front().indexSize();
  int indSize2 = two.front().indexSize();

  for (vector<DetGroup>::const_iterator i=one.begin(); i!=one.end(); i++) {
    result.push_back(*i);
    result.back().setIndexSize(indSize1+indSize2);
  }
  for (vector<DetGroup>::const_iterator j=two.begin(); j!=two.end(); j++) {
    result.push_back(*j);
    result.back().incrementIndex(indSize1);
  }
}

void 
DetGroupMerger::addSameLevel( const vector<DetGroup>& gvec, vector<DetGroup>& result) const{
  for (vector<DetGroup>::const_iterator ig=gvec.begin(); ig != gvec.end(); ig++) {
    int gSize = ig->indexSize();
    int index = ig->index(); // at which level it should be inserted
    bool found = false;
    for (vector<DetGroup>::iterator ires=result.begin(); ires!=result.end(); ires++) {
      int resSize = ires->indexSize();
      if (gSize != resSize) {
	LogDebug("TkDetLayers") << "DetGroupMerger::addSameLevel called with groups of different index sizes";
	// throw something appropriate...or handle it properly (may happen in petals?)
      }

      int resIndex = ires->index();
      if (index == resIndex) {
	ires->insert(ires->end(), ig->begin(), ig->end()); // insert in group with same index
	found = true;
	break;
      }
      else if (index < resIndex) {
	// result has no group at index level yet
	result.insert( ires, *ig); // insert a new group, invalidates the iterator ires
	found = true;
	break;
      }
    } // end of loop over result groups
    if (!found) result.insert( result.end(), *ig); // in case the ig index is bigger than any in result
  }
}

void 
DetGroupMerger::doubleIndexSize( vector<DetGroup>& vec) const{
  int indSize = vec.front().indexSize();
  for (vector<DetGroup>::iterator i=vec.begin(); i!=vec.end(); i++) {
    i->setIndexSize( 2*indSize);
  }
}

void 
DetGroupMerger::incrementAndDoubleSize( vector<DetGroup>& vec) const{
  int indSize = vec.front().indexSize();
  for (vector<DetGroup>::iterator i=vec.begin(); i!=vec.end(); i++) {
    i->incrementIndex( indSize);
  }
}

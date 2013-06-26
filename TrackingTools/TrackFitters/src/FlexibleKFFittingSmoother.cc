#include "TrackingTools/TrackFitters/interface/FlexibleKFFittingSmoother.h"

using namespace std;

FlexibleKFFittingSmoother::~FlexibleKFFittingSmoother() 
{
  delete theStandardFitter;
  delete theLooperFitter;
}

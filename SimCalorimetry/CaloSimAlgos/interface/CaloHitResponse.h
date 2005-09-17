#ifndef CaloHitResponse_h
#define CaloHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <map>
#include<vector>

/**

   \class CaloHitResponse

   \brief Creates electronics signals from hits 

*/

namespace cms {

  class CaloHit;
  class CaloVShape;
  class CaloVSimParameterMap;

  class CaloHitResponse 
  {
  public:
    typedef std::map<DetId, CaloSamples> AnalogSignalMap;
    // get this from somewhere external
    enum {BUNCHSPACE=25};

    CaloHitResponse(CaloVSimParameterMap * parameterMap, CaloVShape * shape);

    /// doesn't delete the pointers passed in
    ~CaloHitResponse() {}

    /// tell sit which pileup bunches to do
    void setBunchRange(int minBunch, int maxBunch);

    /// Complete cell digitization.
    void run(const std::vector<CaloHit> & hits);

    /// frees up memory
    void clear() {theAnalogSignalMap.clear();}
 
    /// adds the amplitude for a single hit to the frame
    void addHit(const CaloHit * hit, CaloSamples & frame) const;

    /// creates a new frame from this hit
    CaloSamples makeFrame(const CaloHit & hit) const;

    /// users can look for the signal for a given cell
    CaloSamples findSignal(const DetId & cell) const;

  protected:

    AnalogSignalMap theAnalogSignalMap;
    // a little prototype to return when there's no signal
    mutable CaloSamples theBlankFrame;

    CaloVSimParameterMap * theParameterMap;
    CaloVShape * theShape;

    int theMinBunch;
    int theMaxBunch;

  };
}

#endif



#ifndef CaloHitResponse_h
#define CaloHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include<map>
#include<vector>

/**

 \class CaloHitResponse

 \brief Creates electronics signals from hits 

*/


namespace cms {

class CaloVShape;
class CaloVSimParameterMap;
class CaloVHitCorrection;
class CaloVHitFilter;
class CaloSimParameters;

class CaloHitResponse 
{
public:
  typedef std::map<DetId, CaloSamples> AnalogSignalMap;
  // get this from somewhere external
  enum {BUNCHSPACE=25};

  CaloHitResponse(const CaloVSimParameterMap * parameterMap, const CaloVShape * shape);

  /// doesn't delete the pointers passed in
  ~CaloHitResponse() {}

  /// tells it which pileup bunches to do
  void setBunchRange(int minBunch, int maxBunch);

  /// geometry needed for time-of-flight
  void setGeometry(const CaloGeometry * geometry) { theGeometry = geometry; }

  /// Complete cell digitization.
  void run(MixCollection<PCaloHit> & hits);

  /// if you want to reject hits, for example, from a certain subdetector, set this
  void setHitFilter(const CaloVHitFilter * filter) {
    theHitFilter = filter;
  }

  /// If you want to correct hits, for attenuation or delay, set this.
  void setHitCorrection(const CaloVHitCorrection * hitCorrection) {
    theHitCorrection = hitCorrection;
  }

  /// frees up memory
  void clear() {theAnalogSignalMap.clear();}
 
  /// adds the amplitude for a single hit to the frame
  void addHit(const PCaloHit * hit, CaloSamples & frame) const;

  /// creates a new frame from this hit
  CaloSamples makeAnalogSignal(const PCaloHit & hit) const;

  /// finds the amplitude contribution from this hit, applying
  /// photostatistics, if needed
  double analogSignalAmplitude(const PCaloHit & hit, const CaloSimParameters & parameters) const;

  /// users can look for the signal for a given cell
  CaloSamples findSignal(const DetId & cell) const;

  /// time-of-flight, in ns, to get to this cell
  /// returns 0 if no geometry has been set
  double timeOfFlight(const DetId & detId) const;

protected:

  AnalogSignalMap theAnalogSignalMap;

  const CaloVSimParameterMap * theParameterMap;
  const CaloVShape * theShape;
  const CaloVHitCorrection * theHitCorrection;
  const CaloVHitFilter * theHitFilter;

  const CaloGeometry * theGeometry;

  int theMinBunch;
  int theMaxBunch;

};

}

#endif



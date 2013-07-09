#ifndef CaloSimAlgos_CaloHitResponse_h
#define CaloSimAlgos_CaloHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVPECorrection.h"

#include<map>
#include<vector>

/**

 \class CaloHitResponse

 \brief Creates electronics signals from hits 

*/
#define ChangeHcalEnergyScale

class CaloVShape;
class CaloShapes;
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
  CaloHitResponse(const CaloVSimParameterMap * parameterMap, const CaloShapes * shapes);

  /// doesn't delete the pointers passed in
  virtual ~CaloHitResponse();

  // change HBHE scale
  void initHBHEScale();
  void setHBHEScale(std::string &); //GMA

  /// tells it which pileup bunches to do
  void setBunchRange(int minBunch, int maxBunch);

  /// geometry needed for time-of-flight
  void setGeometry(const CaloGeometry * geometry) { theGeometry = geometry; }

  virtual bool keepBlank() const { return true ; }

  /// Initialize hits
  virtual void initializeHits() {}

  /// Finalize hits
  virtual void finalizeHits() {}

  /// Complete cell digitization.
  virtual void run(MixCollection<PCaloHit> & hits);

  /// process a single SimHit
  virtual void add(const PCaloHit & hit);

  /// add a signal, in units of pe
  void add(const CaloSamples & signal);

  /// if you want to reject hits, for example, from a certain subdetector, set this
  void setHitFilter(const CaloVHitFilter * filter) {
    theHitFilter = filter;
  }

  /// If you want to correct hits, for attenuation or delay, set this.
  void setHitCorrection(const CaloVHitCorrection * hitCorrection) {
    theHitCorrection = hitCorrection;
  }

  /// if you want to correct the photoelectrons
  void setPECorrection(const CaloVPECorrection * peCorrection) {
    thePECorrection = peCorrection;
  }

  virtual void setRandomEngine(CLHEP::HepRandomEngine & engine);

  /// frees up memory
  void clear() {theAnalogSignalMap.clear();}
 
  /// adds the amplitude for a single hit to the frame
  void addHit(const PCaloHit * hit, CaloSamples & frame) const;

  /// creates the signal corresponding to this hit
  virtual CaloSamples makeAnalogSignal(const PCaloHit & inputHit) const;

  /// finds the amplitude contribution from this hit, applying
  /// photostatistics, if needed.  Results are in photoelectrons
  double analogSignalAmplitude(const DetId & id, float energy, const CaloSimParameters & parameters) const;

  /// users can look for the signal for a given cell
  CaloSamples * findSignal(const DetId & detId);

  /// number of signals in the current cache
  int nSignals() const {return theAnalogSignalMap.size();}

  /// creates an empty signal for this DetId
  CaloSamples makeBlankSignal(const DetId & detId) const;


  /// time-of-flight, in ns, to get to this cell
  /// returns 0 if no geometry has been set
  double timeOfFlight(const DetId & detId) const;

  /// setting the phase shift for asynchronous trigger (e.g. test beams)
  void setPhaseShift(const double & thePhaseShift) { thePhaseShift_ = thePhaseShift; }

  /// check if crossing is within bunch range:

  bool withinBunchRange(int bunchCrossing) const {
    return(bunchCrossing >= theMinBunch && bunchCrossing <= theMaxBunch);
  }

protected:

  AnalogSignalMap theAnalogSignalMap;

  const CaloVSimParameterMap * theParameterMap;
  const CaloShapes * theShapes;
  const CaloVShape * theShape;
  const CaloVHitCorrection * theHitCorrection;
  const CaloVPECorrection * thePECorrection;
  const CaloVHitFilter * theHitFilter;

  const CaloGeometry * theGeometry;

  mutable CLHEP::RandPoissonQ * theRandPoisson;

  int theMinBunch;
  int theMaxBunch;

  double thePhaseShift_;

  // private : 
  bool  changeScale;
#ifdef ChangeHcalEnergyScale
  float hcal_en_scale[100][72][4];
#endif  
};

#endif



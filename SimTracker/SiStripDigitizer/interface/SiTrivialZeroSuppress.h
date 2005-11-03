#ifndef Tracker_SiTrivialZeroSuppress_H
#define Tracker_SiTrivialZeroSuppress_H

#include "SimTracker/SiStripDigitizer/interface/SiZeroSuppress.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 * Trivial zero suppression algorithm, implemented in the trkFEDclusterizer method.
 * The class publically inherits from the SiZeroSuppress class, which requires 
 * the use of a method named zeroSuppress.
 *
 * There are four possible algorithms, the default of which (4)
 * has different thresholds for isolated strips and strips in clusters.
 * It also merges clusters (single or multi strip) that are only separated
 * by one strip. This strip is selected as signal even though it is below
 * both thresholds.
 *
 * When a strip satisfying only the lower threshold is at the edge of an APV or module, 
 * the trkFEDclusterizer method assumes that every strip just outside an APV or module has a hit on it. 
 * This is to avoid cluster inefficiencies at the edges of APVs and modules.   
 */
class SiTrivialZeroSuppress : public SiZeroSuppress{
 public:
   
  /**Constructor. This reads in the noise in the strips.*/
  SiTrivialZeroSuppress(const edm::ParameterSet const& conf, float noise);
  
  /** This calculates the lower and high signal thresholds using the noise. It also
   *  checks for a valid choice of zero suppression algorithm.*/
  void initParams(edm::ParameterSet const& conf_);
 
  /** This simply calls the method to do zero suppression (trkFEDclusterizer).*/
  SiZeroSuppress::DigitalMapType zeroSuppress(const DigitalMapType&);
  
  /** Zero suppression method. The code is placed here, and not in zeroSuppress to
   * enable it to be inherited by another class which may also have a zeroSuppress method (ie. SiFedZeroSuppress)*/
  SiZeroSuppress::DigitalMapType trkFEDclusterizer(const DigitalMapType&); 
  
 private:
  float noiseInAdc;
  short theFEDalgorithm;
  float theFEDlowThresh;
  float theFEDhighThresh;
  short theNumFEDalgos;

  edm::ParameterSet conf_;
  int algoConf;
  double lowthreshConf;
  double highthreshConf;
};
 
#endif

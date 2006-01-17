#ifndef Tracker_SiLinearChargeDivider_H
#define Tracker_SiLinearChargeDivider_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/SiStripDigitizer/interface/SiChargeDivider.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "GeneratorInterface/HepPDT/interface/HepPDTable.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"

/**
 * Concrete implementation of SiChargeDivider. 
 * It divides the charge on the line connecting entry and exit point of the SimTrack in the Silicon.
 */
class SiLinearChargeDivider : public SiChargeDivider{
 public:

  SiLinearChargeDivider(){
    //    particleTable = & HepPDT::theTable(); // aggiungere dopo (AG)
  }
  SiLinearChargeDivider(const edm::ParameterSet& conf);

  virtual ~SiLinearChargeDivider(){
    //    delete particleTable;
  }

  //  SiChargeDivider::ionization_type divide(const SimHit&, const StripDet& det);
  SiChargeDivider::ionization_type divide(const PSimHit&, const StripGeomDetUnit& det);
  
 private:
  edm::ParameterSet conf_;
  
  float PeakShape(const PSimHit&);
  float DeconvolutionShape( const PSimHit&);
  float TimeResponse( const PSimHit&) ; 
  void fluctuateEloss(int particleId, float momentum, float eloss, float length, int NumberOfSegmentation, float elossVector[]);   
  //  static SimpleConfigurable<bool> peakMode;
  bool peakMode;
  //  static SimpleConfigurable<bool> fluctuateCharge;
  bool fluctuateCharge;
  //  static SimpleConfigurable<int>  chargeDivisionsPerStrip;
  int  chargedivisionsPerStrip;
  //  static SimpleConfigurable<double> deltaCut ;
  double deltaCut ;
  SiG4UniversalFluctuation fluctuate; 
  //  HepPDTable * particleTable;
};

#endif

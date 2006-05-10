#ifndef Tracker_SiLinearChargeDivider_H
#define Tracker_SiLinearChargeDivider_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/SiStripDigitizer/interface/SiChargeDivider.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "SimGeneral/HepPDT/interface/HepPDTable.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

/**
 * Concrete implementation of SiChargeDivider. 
 * It divides the charge on the line connecting entry and exit point of the SimTrack in the Silicon.
 */
class SiLinearChargeDivider : public SiChargeDivider{
 public:

  SiLinearChargeDivider(){
    //    particleTable = & HepPDT::theTable(); 
  }
  SiLinearChargeDivider(const edm::ParameterSet& conf);

  virtual ~SiLinearChargeDivider(){
    //    delete particleTable;
  }

  SiChargeDivider::ionization_type divide(const PSimHit&, const StripGeomDetUnit& det);
  
 private:
  edm::ParameterSet conf_;
  
  float PeakShape(const PSimHit&);
  float DeconvolutionShape( const PSimHit&);
  float TimeResponse( const PSimHit&) ; 
  void fluctuateEloss(int particleId, float momentum, float eloss, float length, int NumberOfSegmentation, float elossVector[]);   
  bool peakMode;
  bool fluctuateCharge;
  int  chargedivisionsPerStrip;
  double deltaCut ;
  SiG4UniversalFluctuation fluctuate; 
  //  HepPDTable * particleTable;
};

#endif

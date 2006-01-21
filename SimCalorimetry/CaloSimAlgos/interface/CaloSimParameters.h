#ifndef CaloSimAlgos_CaloSimParameters_h
#define CaloSimAlgos_CaloSimParameters_h

#include <iosfwd>

/**

   \class CaloSimParameters

   \brief Main class for Parameters in different subdetectors.
   
*/
class CaloSimParameters
{
public:
  CaloSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog, 
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
  : simHitToPhotoelectrons_(simHitToPhotoelectrons),
    photoelectronsToAnalog_(photoelectronsToAnalog),
    samplingFactor_(samplingFactor),
    timePhase_(timePhase),
    readoutFrameSize_(readoutFrameSize),
    binOfMaximum_(binOfMaximum),
    doPhotostatistics_(doPhotostatistics)
  {
  }

  ~CaloSimParameters() {};

  /// the factor which goes from whatever units the SimHit amplitudes
  /// are in (could be deposited GeV, real GeV, or photoelectrons)
  /// and converts to photoelectrons
  double simHitToPhotoelectrons() const { return simHitToPhotoelectrons_;}

  /// the factor which goes from photoelectrons to whatever gets read by ADCs
 double photoelectronsToAnalog() const {return photoelectronsToAnalog_;}

  /// the ratio of actual incident energy to deposited energy
  /// in the SimHit
  double samplingFactor() const {return samplingFactor_;}

  /// the adjustment you need to apply to get the signal where you want it
  double timePhase() const {return timePhase_;}

  /// the factor that goes from linearized ADC counts to energy
  /// (GeV/dGeV) / (pe/dGeV) / (signal/pe)
  double calibrationConstant() const {
    return samplingFactor_ / simHitToPhotoelectrons_ / photoelectronsToAnalog_;
  }

  /// for now, the LinearFrames and trhe digis will be one-to-one.
  int readoutFrameSize() const {return readoutFrameSize_;}

  int binOfMaximum() const {return binOfMaximum_;}

  /// whether or not to apply Poisson statistics to photoelectrons
  bool doPhotostatistics() const {return doPhotostatistics_;}


private:
  double simHitToPhotoelectrons_;
  double photoelectronsToAnalog_;
  double samplingFactor_;
  double timePhase_;
  int readoutFrameSize_;
  int binOfMaximum_;
  bool doPhotostatistics_;

};

std::ostream & operator<<(std::ostream & os, const CaloSimParameters & p);

#endif



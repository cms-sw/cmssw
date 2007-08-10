#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/CSCDigitizer/src/CSCWireElectronicsSim.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include <iostream>


CSCWireElectronicsSim::CSCWireElectronicsSim(const edm::ParameterSet & p) 
 : CSCBaseElectronicsSim(p),
   theFraction(0.5),
   theWireNoise(0.0),
   theWireThreshold(0.),
   theTimingCalibrationError(p.getParameter<double>("wireTimingError"))
{
  fillAmpResponse();
}


void CSCWireElectronicsSim::initParameters() {
  theLayerGeometry = theLayer->geometry();
  nElements = theLayerGeometry->numberOfWireGroups();
  theWireNoise = theSpecs->wireNoise(theShapingTime)
                * e_SI * pow(10.0,15);
  theWireThreshold = theWireNoise * 8;
}


int CSCWireElectronicsSim::readoutElement(int element) const {
  return theLayerGeometry->wireGroup(element);
}

void CSCWireElectronicsSim::fillDigis(CSCWireDigiCollection & digis) {

  if(theSignalMap.empty()) {
    return;
  }

  // Loop over analog signals, run the fractional discriminator on each one,
  // and save the DIGI in the layer.
  for(CSCSignalMap::iterator mapI = theSignalMap.begin(),
      lastSignal = theSignalMap.end();
      mapI != lastSignal; ++mapI) 
  {
    int wireGroup = (*mapI).first;
    const CSCAnalogSignal & signal = (*mapI).second;
    LogTrace("CSCWireElectronicsSim") << "CSCWireElectronicsSim: dump of wire signal follows... " 
       <<  signal;
    int signalSize = signal.getSize();
    // the way we handle noise in this chamber is by randomly varying
    // the threshold
    float threshold = theWireThreshold + theRandGaussQ->fire() * theWireNoise;
    for(int ibin = 0; ibin < signalSize; ++ibin)
      if(signal.getBinValue(ibin) > threshold) {
        // jackpot.  Now define this signal as everything up until
        // the signal goes below zero.
        int lastbin = signalSize;
        int i;
        for(i = ibin; i < signalSize; ++i) {
          if(signal.getBinValue(i) < 0.) {
            lastbin = i;
            break;
          }
        }

      float qMax = 0.0;
      // in this loop, find the max charge and the 'fifth' electron arrival
      for ( i = ibin; i < lastbin; ++i)
      {
        float next_charge = signal.getBinValue(i);
        if(next_charge > qMax) {
          qMax = next_charge;
        }
      }
     
      int bin_firing_FD = 0;
      for ( i = ibin; i < lastbin; ++i)
      {
        if( bin_firing_FD == 0 && signal.getBinValue(i) >= qMax * theFraction )
        {
           bin_firing_FD = i;
        }
      } 
      float tofOffset = timeOfFlightCalibration(wireGroup);
      int chamberType = theSpecs->chamberType();
      // fill in the rest of the information, for everything that survives the fraction discrim.

      // Shouldn't one measure from theTimeOffset of the CSCAnalogSignal?
      // Well, yes, but unfortunately it seems CSCAnalogSignal::superimpose
      // fails to reset theTimeOffset properly. In any case, if everything
      // is self-consistent, the overall theTimeOffset should BE
      // theSignalStartTime. There is big trouble if any of the
      // overlapping MEAS's happen to have a timeOffset earlier than
      // theSignalStartTime (which is why it defaults to -10bx = -250ns).
      // That could only happen in the case of signals from pile-up events
      // arising way earlier than 10bx before the signal event crossing
      // and so only if pile-up were simulated over an enormous range of
      // bx earlier than the signal bx.
      // (Comments from Tim, Aug-2005)

      float fdTime = theRandGaussQ->fire(theSignalStartTime + theSamplingTime*bin_firing_FD,
                                         theTimingCalibrationError);

      int beamCrossingTag = 
         static_cast<int>( (fdTime - tofOffset - theBunchTimingOffsets[chamberType])
                            / theBunchSpacing );

      // Wire digi as of Oct-2006 adapted to real data: time word has 16 bits with set bit
      // flagging appropriate bunch crossing, and bx 0 corresponding to 7th bit i.e.

      //      1st bit set (bit 0) <-> bx -6
      //      2nd              1  <-> bx -5
      //      ...           ...       ....
      //      7th              6  <-> bx  0
      //      8th              7  <-> bx +1
      //      ...           ...       ....
      //     16th             15  <-> bx +9

      // Parameter theOffsetOfBxZero = 6 @@WARNING! This offset may be changed (hardware)!

      int nBitsToOffset = beamCrossingTag + theOffsetOfBxZero;
      int timeWord = 0; // and this will remain if too early or late (<bx-6 or >bx+9)
      if ( (nBitsToOffset>= 0) && (nBitsToOffset<16) ) 
 	 timeWord = (1 << nBitsToOffset ); // set appropriate bit

      CSCWireDigi newDigi(wireGroup, timeWord);
      LogTrace("CSCWireElectronicsSim") << newDigi;
      digis.insertDigi(layerId(), newDigi);

      // we code the channels so strips start at 1, wire groups at 101
      addLinks(channelIndex(wireGroup));
      // skip over all the time bins used for this digi
      ibin = lastbin;
    } // loop over time bins in signal
  } // loop over wire signals   
}


float CSCWireElectronicsSim::calculateAmpResponse(float t) const {
  static const float fC_by_ns = 1000000;
  static const float resistor = 20000;
  static const float amplifier_pole               = 1/7.5;
  static const float fastest_chamber_exp_risetime = 10.;
  static const float p0=amplifier_pole;
  static const float p1=1/fastest_chamber_exp_risetime;

  static const float dp = p0 - p1;

  // ENABLE DISC:

  static const float norm = -12 * resistor * p1 * pow(p0/dp, 4) / fC_by_ns;

  float enable_disc_volts = norm*(  exp(-p0*t) *(1          +
						 t*dp       +
						 pow(t*dp,2)/2 +
						 pow(t*dp,3)/6  )
				    - exp(-p1*t) );
  static const float collectionFraction = 0.12;
  static const float igain = 1./0.005; // volts per fC
  return enable_disc_volts * igain * collectionFraction;
}                                                                               


float CSCWireElectronicsSim::signalDelay(int element, float pos) const {
  // readout is on right edge of chamber, signal speed is c
  // zero calibrated to chamber center
  // pos is assumed to be in wire coordinates, not local
  float distance = -1. * pos; // in cm
  float speed = c_light / cm;
  float delay = distance / speed;
  return delay;
}

float CSCWireElectronicsSim::timeOfFlightCalibration(int wireGroup) const {
  // calibration is done for groups of 8 wire groups, facetiously
  // called wireGroupGroups
  int middleWireGroup = wireGroup - wireGroup%8 + 4;
  int numberOfWireGroups = theLayerGeometry->numberOfWireGroups();
  if(middleWireGroup > numberOfWireGroups) 
     middleWireGroup = numberOfWireGroups;

//  LocalPoint centerOfGroupGroup = theLayerGeometry->centerOfWireGroup(middleWireGroup);
//  float averageDist = theLayer->surface().toGlobal(centerOfGroupGroup).mag();
  GlobalPoint centerOfGroupGroup = theLayer->centerOfWireGroup(middleWireGroup);
  float averageDist = centerOfGroupGroup.mag();


  float averageTOF  = averageDist * cm / c_light; // Units of c_light: mm/ns

  LogTrace("CSCWireElectronicsSim") << "CSCWireElectronicsSim: TofCalib  wg = " << wireGroup << 
       " mid wg = " << middleWireGroup << 
       " av dist = " << averageDist << 
      " av tof = " << averageTOF;
  
  return averageTOF;
}
 

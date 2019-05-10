#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "SimMuon/CSCDigitizer/src/CSCWireElectronicsSim.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>

CSCWireElectronicsSim::CSCWireElectronicsSim(const edm::ParameterSet &p)
    : CSCBaseElectronicsSim(p), theFraction(0.5), theWireNoise(0.0), theWireThreshold(0.) {
  fillAmpResponse();
}

void CSCWireElectronicsSim::initParameters() {
  nElements = theLayerGeometry->numberOfWireGroups();
  theWireNoise = theSpecs->wireNoise(theShapingTime) * e_SI * pow(10.0, 15);
  theWireThreshold = theWireNoise * 8;
}

int CSCWireElectronicsSim::readoutElement(int element) const { return theLayerGeometry->wireGroup(element); }

void CSCWireElectronicsSim::fillDigis(CSCWireDigiCollection &digis, CLHEP::HepRandomEngine *engine) {
  if (theSignalMap.empty()) {
    return;
  }

  // Loop over analog signals, run the fractional discriminator on each one,
  // and save the DIGI in the layer.
  for (CSCSignalMap::iterator mapI = theSignalMap.begin(), lastSignal = theSignalMap.end(); mapI != lastSignal;
       ++mapI) {
    int wireGroup = (*mapI).first;
    const CSCAnalogSignal &signal = (*mapI).second;
    LogTrace("CSCWireElectronicsSim") << "CSCWireElectronicsSim: dump of wire signal follows... " << signal;
    int signalSize = signal.getSize();

    int timeWord = 0;  // and this will remain if too early or late (<bx-6 or >bx+9)

    // the way we handle noise in this chamber is by randomly varying
    // the threshold
    float threshold = theWireThreshold;
    if (doNoise_) {
      threshold += CLHEP::RandGaussQ::shoot(engine) * theWireNoise;
    }
    for (int ibin = 0; ibin < signalSize; ++ibin) {
      if (signal.getBinValue(ibin) > threshold) {
        // jackpot.  Now define this signal as everything up until
        // the signal goes below zero.
        int lastbin = signalSize;
        int i;
        for (i = ibin; i < signalSize; ++i) {
          if (signal.getBinValue(i) < 0.) {
            lastbin = i;
            break;
          }
        }

        float qMax = 0.0;
        // in this loop, find the max charge and the 'fifth' electron arrival
        for (i = ibin; i < lastbin; ++i) {
          float next_charge = signal.getBinValue(i);
          if (next_charge > qMax) {
            qMax = next_charge;
          }
        }

        int bin_firing_FD = 0;
        for (i = ibin; i < lastbin; ++i) {
          if (signal.getBinValue(i) >= qMax * theFraction) {
            bin_firing_FD = i;
            //@@ Long-standing but unlikely minor bug, I (Tim) think - following
            //'break' was missing...
            //@@ ... So if both ibins 0 and 1 could fire FD, we'd flag the
            // firing bin as 1 not 0
            //@@ (since the above test was restricted to bin_firing_FD==0 too).
            break;
          }
        }

        float tofOffset = timeOfFlightCalibration(wireGroup);
        int chamberType = theSpecs->chamberType();

        // Note that CSCAnalogSignal::superimpose does not reset theTimeOffset
        // to the earliest of the two signal's time offsets. If it did then we
        // could handle signals from any time range e.g. form pileup events many
        // bx's from the signal bx (bx=0). But then we would be wastefully
        // storing signals over times which we can never see in the real
        // detector, because only hits within a few bx's of bx=0 are read out.
        // Instead, the working time range for wire hits is always started from
        // theSignalStartTime, set as a parameter in the config file.
        // On the other hand, if any of the overlapped CSCAnalogSignals happens
        // to have a timeOffset earlier than theSignalStartTime (which is
        // currently set to -100 ns) then we're in trouble. For pileup events
        // this would mean events from collisions earlier than 4 bx before the
        // signal bx.

        float fdTime = theSignalStartTime + theSamplingTime * bin_firing_FD;
        if (doNoise_) {
          fdTime += theTimingCalibrationError[chamberType] * CLHEP::RandGaussQ::shoot(engine);
        }

        float bxFloat = (fdTime - tofOffset - theBunchTimingOffsets[chamberType]) / theBunchSpacing + theOffsetOfBxZero;
        int bxInt = static_cast<int>(bxFloat);
        if (bxFloat >= 0 && bxFloat < 16) {
          timeWord |= (1 << bxInt);
          // discriminator stays high for 35 ns
          if (bxFloat - bxInt > 0.6) {
            timeWord |= (1 << (bxInt + 1));
          }
        }

        // Wire digi as of Oct-2006 adapted to real data: time word has 16 bits
        // with set bit flagging appropriate bunch crossing, and bx 0
        // corresponding to the 7th bit, 'bit 6':

        //      1st bit set (bit 0) <-> bx -6
        //      2nd              1  <-> bx -5
        //      ...           ...       ....
        //      7th              6  <-> bx  0
        //      8th              7  <-> bx +1
        //      ...           ...       ....
        //     16th             15  <-> bx +9

        // skip over all the time bins used for this digi
        ibin = lastbin;
      }  // if over threshold
    }    // loop over time bins in signal

    // Only create a wire digi if there is a wire hit within [-6 bx, +9 bx]
    if (timeWord != 0) {
      CSCWireDigi newDigi(wireGroup, timeWord);
      LogTrace("CSCWireElectronicsSim") << newDigi;
      digis.insertDigi(layerId(), newDigi);
      addLinks(channelIndex(wireGroup));
    }
  }  // loop over wire signals
}

float CSCWireElectronicsSim::calculateAmpResponse(float t) const {
  static const float fC_by_ns = 1000000;
  static const float resistor = 20000;
  static const float amplifier_pole = 1 / 7.5;
  static const float fastest_chamber_exp_risetime = 10.;
  static const float p0 = amplifier_pole;
  static const float p1 = 1 / fastest_chamber_exp_risetime;

  static const float dp = p0 - p1;

  // ENABLE DISC:

  static const float norm = -12 * resistor * p1 * pow(p0 / dp, 4) / fC_by_ns;

  float enable_disc_volts =
      norm * (exp(-p0 * t) * (1 + t * dp + pow(t * dp, 2) / 2 + pow(t * dp, 3) / 6) - exp(-p1 * t));
  static const float collectionFraction = 0.12;
  static const float igain = 1. / 0.005;  // volts per fC
  return enable_disc_volts * igain * collectionFraction;
}

float CSCWireElectronicsSim::timeOfFlightCalibration(int wireGroup) const {
  // calibration is done for groups of 8 wire groups, facetiously
  // called wireGroupGroups
  int middleWireGroup = wireGroup - wireGroup % 8 + 4;
  int numberOfWireGroups = theLayerGeometry->numberOfWireGroups();
  if (middleWireGroup > numberOfWireGroups)
    middleWireGroup = numberOfWireGroups;

  GlobalPoint centerOfGroupGroup = theLayer->centerOfWireGroup(middleWireGroup);
  float averageDist = centerOfGroupGroup.mag();
  float averageTOF = averageDist * cm / c_light;  // Units of c_light: mm/ns

  LogTrace("CSCWireElectronicsSim") << "CSCWireElectronicsSim: TofCalib  wg = " << wireGroup
                                    << " mid wg = " << middleWireGroup << " av dist = " << averageDist
                                    << " av tof = " << averageTOF;

  return averageTOF;
}

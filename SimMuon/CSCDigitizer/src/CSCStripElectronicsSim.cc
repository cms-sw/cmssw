#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "SimMuon/CSCDigitizer/src/CSCCrosstalkGenerator.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "SimMuon/CSCDigitizer/src/CSCStripElectronicsSim.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <cassert>
#include <list>

// This is CSCStripElectronicsSim.cc

CSCStripElectronicsSim::CSCStripElectronicsSim(const edm::ParameterSet &p)
    : CSCBaseElectronicsSim(p),
      theAmpResponse(theShapingTime, CSCStripAmpResponse::RADICAL),
      theComparatorThreshold(20.),
      theComparatorNoise(0.),
      theComparatorRMSOffset(2.),
      theComparatorSaturation(1057.),
      theComparatorWait(50.),
      theComparatorDeadTime(100.),
      theDaqDeadTime(200.),
      theTimingOffset(0.),
      nScaBins_(p.getParameter<int>("nScaBins")),
      doSuppression_(p.getParameter<bool>("doSuppression")),
      doCrosstalk_(p.getParameter<bool>("doCrosstalk")),
      theStripConditions(nullptr),
      theCrosstalkGenerator(nullptr),
      theComparatorClockJump(2),
      sca_time_bin_size(50.),
      sca_peak_bin(p.getParameter<int>("scaPeakBin")),
      theComparatorTimeBinOffset(p.getParameter<double>("comparatorTimeBinOffset")),
      theComparatorTimeOffset(p.getParameter<double>("comparatorTimeOffset")),
      theComparatorSamplingTime(p.getParameter<double>("comparatorSamplingTime")),
      theSCATimingOffsets(p.getParameter<std::vector<double>>("scaTimingOffsets")) {
  if (doCrosstalk_) {
    theCrosstalkGenerator = new CSCCrosstalkGenerator();
  }

  fillAmpResponse();
}

CSCStripElectronicsSim::~CSCStripElectronicsSim() {
  if (doCrosstalk_) {
    delete theCrosstalkGenerator;
  }
}

void CSCStripElectronicsSim::initParameters() {
  nElements = theLayerGeometry->numberOfStrips();
  theComparatorThreshold = 20.;
  // selfTest();

  // calculate the offset to the peak
  float averageDistance = theLayer->surface().position().mag();
  theAverageTimeOfFlight = averageDistance * cm / c_light;  // Units of c_light: mm/ns
  int chamberType = theSpecs->chamberType();
  theTimingOffset = theShapingTime + theAverageTimeOfFlight + theBunchTimingOffsets[chamberType];
  // TODO make sure config gets overridden
  theSignalStartTime = theTimingOffset - (sca_peak_bin - 1) * sca_time_bin_size;
  theSignalStopTime = theSignalStartTime + nScaBins_ * sca_time_bin_size;
  theNumberOfSamples = nScaBins_ * static_cast<int>(sca_time_bin_size / theSamplingTime);
}

int CSCStripElectronicsSim::readoutElement(int strip) const { return theLayerGeometry->channel(strip); }

float CSCStripElectronicsSim::calculateAmpResponse(float t) const { return theAmpResponse.calculateAmpResponse(t); }

CSCAnalogSignal CSCStripElectronicsSim::makeNoiseSignal(int element, CLHEP::HepRandomEngine *engine) {
  std::vector<float> noiseBins(nScaBins_);
  CSCAnalogSignal tmpSignal(element, sca_time_bin_size, noiseBins);
  if (doNoise_) {
    theStripConditions->noisify(layerId(), tmpSignal, engine);
  }
  // now rebin it
  std::vector<float> binValues(theNumberOfSamples);
  for (int ibin = 0; ibin < theNumberOfSamples; ++ibin) {
    binValues[ibin] = tmpSignal.getValue(ibin * theSamplingTime);
  }
  CSCAnalogSignal finalSignal(element, theSamplingTime, binValues, 0., theSignalStartTime);
  return finalSignal;
}

float CSCStripElectronicsSim::comparatorReading(const CSCAnalogSignal &signal,
                                                float time,
                                                CLHEP::HepRandomEngine *engine) const {
  return std::min(signal.getValue(time), theComparatorSaturation) +
         theComparatorRMSOffset * CLHEP::RandGaussQ::shoot(engine);
}

void CSCStripElectronicsSim::runComparator(std::vector<CSCComparatorDigi> &result, CLHEP::HepRandomEngine *engine) {
  // first, make a list of all the comparators we actually
  // need to run
  std::list<int> comparatorsWithSignal;
  CSCSignalMap::iterator signalMapItr;
  for (signalMapItr = theSignalMap.begin(); signalMapItr != theSignalMap.end(); ++signalMapItr) {
    // Elements in signal map count from 1
    // 1,2->0,  3,4->1,  5,6->2, ...
    comparatorsWithSignal.push_back(((*signalMapItr).first - 1) / 2);
  }
  // no need to sort
  comparatorsWithSignal.unique();
  for (std::list<int>::iterator listItr = comparatorsWithSignal.begin(); listItr != comparatorsWithSignal.end();
       ++listItr) {
    int iComparator = *listItr;
    // find signal1 and signal2
    // iComparator counts from 0
    // icomp =0->1,2,  =1->3,4,  =2->5,6, ...
    const CSCAnalogSignal &signal1 = find(readoutElement(iComparator * 2 + 1), engine);
    const CSCAnalogSignal &signal2 = find(readoutElement(iComparator * 2 + 2), engine);
    for (float time = theSignalStartTime + theComparatorTimeOffset; time < theSignalStopTime - theComparatorWait;
         time += theComparatorSamplingTime) {
      if (comparatorReading(signal1, time, engine) > theComparatorThreshold ||
          comparatorReading(signal2, time, engine) > theComparatorThreshold) {
        // wait a bit, so we can run the comparator at the signal peak
        float comparatorTime = time;
        time += theComparatorWait;

        float height1 = comparatorReading(signal1, time, engine);
        float height2 = comparatorReading(signal2, time, engine);
        int output = 0;
        int strip = 0;
        // distrip logic; comparator output is for pairs of strips:
        // hit  bin  dec
        // x--- 100   4
        // -x-- 101   5
        // --x- 110   6
        // ---x 111   7
        // just to prevent a copy
        const CSCAnalogSignal *mainSignal = nullptr;
        // pick the higher of the two strips in the pair
        if (height1 > height2) {
          mainSignal = &signal1;
          float leftStrip = 0.;
          if (iComparator > 0) {
            leftStrip = comparatorReading(find(readoutElement(iComparator * 2), engine), time, engine);
          }
          // if this strip is higher than either of its neighbors, make a
          // comparator digi
          if (leftStrip < height1 && height1 > theComparatorThreshold) {
            output = (leftStrip < height2);
            strip = iComparator * 2 + 1;
          }
        } else {
          mainSignal = &signal2;
          float rightStrip = 0.;
          if (iComparator * 2 + 3 <= nElements) {
            rightStrip = comparatorReading(find(readoutElement(iComparator * 2 + 3), engine), time, engine);
          }
          if (rightStrip < height2 && height2 > theComparatorThreshold) {
            output = (height1 < rightStrip);
            strip = iComparator * 2 + 2;
          }
        }
        if (strip != 0) {
          float bxFloat =
              (comparatorTime - theTimingOffset) / theBunchSpacing + theComparatorTimeBinOffset + theOffsetOfBxZero;

          // Comparator digi as of Nov-2006 adapted to real data: time word has
          // 16 bits with set bit flagging appropriate bunch crossing, and bx 0
          // corresponding to 9th bit i.e.

          //      1st bit set (bit 0) <-> bx -9
          //      2nd              1  <-> bx -8
          //      ...           ...       ....
          //      8th              9  <-> bx  0
          //      9th             10  <-> bx +1
          //      ...           ...       ....
          //     16th             15  <-> bx +6

          // Parameter theOffsetOfBxZero = 9 @@WARNING! This offset may be
          // changed (hardware)!

          int timeWord = 0;  // and this will remain if too early or late
          if ((bxFloat >= 0) && (bxFloat < 16))
            timeWord = (1 << static_cast<int>(bxFloat));  // set appropriate bit

          CSCComparatorDigi newDigi(strip, output, timeWord);
          result.push_back(newDigi);
        }

        // wait for the comparator to reset
        time += theComparatorDeadTime;
        // really should be zero, but strip signal doesn't go negative yet
        float resetThreshold = 1;
        while (time < theSignalStopTime && mainSignal->getValue(time) > resetThreshold) {
          time += theComparatorSamplingTime;
        }

      }  // if over threshold
    }    // loop over time samples
  }      // loop over comparators
  // sort by time
  sort(result.begin(), result.end());
}

std::list<int> CSCStripElectronicsSim::getKeyStrips(const std::vector<CSCComparatorDigi> &comparators) const {
  std::list<int> result;
  for (std::vector<CSCComparatorDigi>::const_iterator compItr = comparators.begin(); compItr != comparators.end();
       ++compItr) {
    if (std::abs(compItr->getTimeBin() - theOffsetOfBxZero) <= 2) {
      result.push_back(compItr->getStrip());
    }
  }
  // need sort for unique to work.
  result.sort();
  result.unique();
  return result;
}

std::list<int> CSCStripElectronicsSim::getKeyStripsFromMC() const {
  // assumes the detector hit map is filled
  std::list<int> result;
  transform(theDetectorHitMap.begin(),
            theDetectorHitMap.end(),
            //   back_inserter(result),
            //   boost::bind(&DetectorHitMap::value_type::first,_1));
            // suggested code from Chris Jones
            back_inserter(result),
            std::bind(&DetectorHitMap::value_type::first, std::placeholders::_1));
  //  back_inserter(result), [](DetectorHitMap::value_type const& iValue) {
  //  return iValue.first; } );
  result.sort();
  result.unique();
  return result;
}

std::list<int> CSCStripElectronicsSim::channelsToRead(const std::list<int> &keyStrips, int window) const {
  std::list<int> result;
  std::list<int>::const_iterator keyStripItr = keyStrips.begin();
  if (doSuppression_) {
    for (; keyStripItr != keyStrips.end(); ++keyStripItr) {
      // pick the five strips around the comparator
      for (int istrip = (*keyStripItr) - window; istrip <= (*keyStripItr) + window; ++istrip) {
        if (istrip > 0 && istrip <= nElements) {
          result.push_back(readoutElement(istrip));
        }
      }
    }
    result.sort();
    result.unique();
  } else {
    // read the whole CFEB, 16 strips
    std::list<int> cfebsToRead;
    for (; keyStripItr != keyStrips.end(); ++keyStripItr) {
      int cfeb = (readoutElement(*keyStripItr) - 1) / 16;
      cfebsToRead.push_back(cfeb);
      int remainder = (readoutElement(*keyStripItr) - 1) % 16;
      // if we're within 3 strips of an edge, take neighboring CFEB, too
      if (remainder < window && cfeb != 0) {
        cfebsToRead.push_back(cfeb - 1);
      }
      // the 'readouElement' makes it so that ME1/1 has just one CFEB
      int maxCFEBs = readoutElement(nElements) / 16 - 1;
      if (remainder >= 16 - window && cfeb != maxCFEBs) {
        cfebsToRead.push_back(cfeb + 1);
      }
    }
    cfebsToRead.sort();
    cfebsToRead.unique();

    // now convert the CFEBS to strips
    for (std::list<int>::const_iterator cfebItr = cfebsToRead.begin(); cfebItr != cfebsToRead.end(); ++cfebItr) {
      for (int i = 1; i <= 16; ++i) {
        result.push_back((*cfebItr) * 16 + i);
      }
    }
  }
  return result;
}

bool SortSignalsByTotal(const CSCAnalogSignal &s1, const CSCAnalogSignal &s2) {
  return (s1.getTotal() > s2.getTotal());
}

void CSCStripElectronicsSim::fillDigis(CSCStripDigiCollection &digis,
                                       CSCComparatorDigiCollection &comparators,
                                       CLHEP::HepRandomEngine *engine) {
  if (doCrosstalk_) {
    addCrosstalk(engine);
  }

  std::vector<CSCComparatorDigi> comparatorOutputs;
  runComparator(comparatorOutputs, engine);
  // copy these to the result
  if (!comparatorOutputs.empty()) {
    CSCComparatorDigiCollection::Range range(comparatorOutputs.begin(), comparatorOutputs.end());
    comparators.put(range, layerId());
  }

  // std::list<int> keyStrips = getKeyStrips(comparatorOutputs);
  std::list<int> keyStrips = getKeyStripsFromMC();
  fillStripDigis(keyStrips, digis, engine);
}

void CSCStripElectronicsSim::fillStripDigis(const std::list<int> &keyStrips,
                                            CSCStripDigiCollection &digis,
                                            CLHEP::HepRandomEngine *engine) {
  std::list<int> stripsToDo = channelsToRead(keyStrips, 3);
  std::vector<CSCStripDigi> stripDigis;
  stripDigis.reserve(stripsToDo.size());
  for (std::list<int>::const_iterator stripItr = stripsToDo.begin(); stripItr != stripsToDo.end(); ++stripItr) {
    createDigi(*stripItr, find(*stripItr, engine), stripDigis, engine);
  }

  CSCStripDigiCollection::Range stripRange(stripDigis.begin(), stripDigis.end());
  digis.put(stripRange, layerId());
}

void CSCStripElectronicsSim::addCrosstalk(CLHEP::HepRandomEngine *engine) {
  // this is needed so we can add a noise signal to the map
  // without messing up any iterators
  std::vector<CSCAnalogSignal> realSignals;
  realSignals.reserve(theSignalMap.size());
  CSCSignalMap::iterator mapI = theSignalMap.begin(), mapEnd = theSignalMap.end();
  for (; mapI != mapEnd; ++mapI) {
    realSignals.push_back((*mapI).second);
  }
  sort(realSignals.begin(), realSignals.end(), SortSignalsByTotal);
  std::vector<CSCAnalogSignal>::iterator realSignalItr = realSignals.begin(), realSignalsEnd = realSignals.end();
  for (; realSignalItr != realSignalsEnd; ++realSignalItr) {
    int thisStrip = (*realSignalItr).getElement();
    // add it to each neighbor
    if (thisStrip > 1) {
      int otherStrip = thisStrip - 1;
      addCrosstalk(*realSignalItr, thisStrip, otherStrip, engine);
    }
    if (thisStrip < nElements) {
      int otherStrip = thisStrip + 1;
      addCrosstalk(*realSignalItr, thisStrip, otherStrip, engine);
    }
  }
}

void CSCStripElectronicsSim::addCrosstalk(const CSCAnalogSignal &signal,
                                          int thisStrip,
                                          int otherStrip,
                                          CLHEP::HepRandomEngine *engine) {
  float capacitiveCrosstalk, resistiveCrosstalk;
  bool leftRight = (otherStrip > thisStrip);
  theStripConditions->crosstalk(
      layerId(), thisStrip, theLayerGeometry->length(), leftRight, capacitiveCrosstalk, resistiveCrosstalk);
  theCrosstalkGenerator->setParameters(capacitiveCrosstalk, 0., resistiveCrosstalk);
  CSCAnalogSignal crosstalkSignal(theCrosstalkGenerator->getCrosstalk(signal));
  find(readoutElement(otherStrip), engine).superimpose(crosstalkSignal);

  // Now subtract the crosstalk signal from the original signal
  crosstalkSignal *= -1.;
  find(thisStrip, engine).superimpose(crosstalkSignal);
}

void CSCStripElectronicsSim::createDigi(int channel,
                                        const CSCAnalogSignal &signal,
                                        std::vector<CSCStripDigi> &result,
                                        CLHEP::HepRandomEngine *engine) {
  // fill in the sca information
  std::vector<int> scaCounts(nScaBins_);

  float pedestal = theStripConditions->pedestal(layerId(), channel);
  float gain = theStripConditions->smearedGain(layerId(), channel, engine);
  int chamberType = theSpecs->chamberType();
  float timeSmearing = CLHEP::RandGaussQ::shoot(engine) * theTimingCalibrationError[chamberType];
  // undo the correction for TOF, instead, using some nominal
  // value from ME2/1
  float t0 = theSignalStartTime + theSCATimingOffsets[chamberType] + timeSmearing + 29. - theAverageTimeOfFlight;
  for (int scaBin = 0; scaBin < nScaBins_; ++scaBin) {
    float t = t0 + scaBin * sca_time_bin_size;
    scaCounts[scaBin] = static_cast<int>(pedestal + signal.getValue(t) * gain);
  }
  CSCStripDigi newDigi(channel, scaCounts);

  // do saturation of 12-bit ADC
  doSaturation(newDigi);

  result.push_back(newDigi);
  addLinks(channelIndex(channel));
  LogTrace("CSCStripElectronicsSim") << newDigi;
}

void CSCStripElectronicsSim::doSaturation(CSCStripDigi &digi) {
  std::vector<int> scaCounts(digi.getADCCounts());
  for (unsigned scaBin = 0; scaBin < scaCounts.size(); ++scaBin) {
    scaCounts[scaBin] = std::min(scaCounts[scaBin], 4095);
  }
  digi.setADCCounts(scaCounts);
}

void CSCStripElectronicsSim::fillMissingLayer(const CSCLayer *layer,
                                              const CSCComparatorDigiCollection &comparators,
                                              CSCStripDigiCollection &digis,
                                              CLHEP::HepRandomEngine *engine) {
  theSignalMap.clear();
  setLayer(layer);
  CSCDetId chamberId(theLayerId.chamberId());
  // find all comparator key strips in this chamber
  std::list<int> chamberKeyStrips;
  for (CSCComparatorDigiCollection::DigiRangeIterator comparatorItr = comparators.begin();
       comparatorItr != comparators.end();
       ++comparatorItr) {
    // could be more efficient
    if (CSCDetId((*comparatorItr).first).chamberId() == chamberId) {
      std::vector<CSCComparatorDigi> layerComparators((*comparatorItr).second.first, (*comparatorItr).second.second);
      std::list<int> layerKeyStrips = getKeyStrips(layerComparators);
      chamberKeyStrips.insert(chamberKeyStrips.end(), layerKeyStrips.begin(), layerKeyStrips.end());
    }
  }
  chamberKeyStrips.sort();
  chamberKeyStrips.unique();
  fillStripDigis(chamberKeyStrips, digis, engine);
}

void CSCStripElectronicsSim::selfTest() const {
  // make sure the zero suppression algorithms work
  std::list<int> keyStrips, stripsRead;
  //
  bool isGanged = (readoutElement(nElements) == 16);
  keyStrips.push_back(readoutElement(19));
  keyStrips.push_back(readoutElement(30));
  keyStrips.push_back(readoutElement(32));
  stripsRead = channelsToRead(keyStrips, 3);
  if (doSuppression_) {
    unsigned int expectedSize = isGanged ? 10 : 12;
    assert(stripsRead.size() == expectedSize);
    assert(stripsRead.front() == readoutElement(17));
  } else {
    unsigned int expectedSize = isGanged ? 16 : 48;
    assert(stripsRead.size() == expectedSize);
    assert(stripsRead.front() == 1);
  }
}

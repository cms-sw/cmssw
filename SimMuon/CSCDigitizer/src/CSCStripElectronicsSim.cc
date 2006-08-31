#include "Utilities/Timing/interface/TimingReport.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/CSCDigitizer/src/CSCStripElectronicsSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "SimMuon/CSCDigitizer/src/CSCCrosstalkGenerator.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "SimMuon/CSCDigitizer/src/CSCScaNoiseReader.h"
#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGaussian.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include<list>

// This is CSCStripElectronicsSim.cc

CSCStripElectronicsSim::CSCStripElectronicsSim(const edm::ParameterSet & p)
: CSCBaseElectronicsSim(p),
  theComparatorThreshold(20.),
  theComparatorNoise(0.),
  theComparatorRMSOffset(2.),
  theComparatorSaturation(1057.),
  theComparatorWait(50.),
  theComparatorDeadTime(100.),
  theDaqDeadTime(200.),
  nScaBins_(p.getParameter<int>("nScaBins")),
  doCrosstalk_(p.getParameter<bool>("doCrosstalk")),
  theCrosstalkGenerator(0),
  theScaNoiseGenerator(0),
  theComparatorClockJump(2),
  sca_time_bin_size(50.),
  sca_noise(0.),
  theAnalogNoise(p.getParameter<double>("analogNoise")),
  thePedestal(p.getParameter<double>("pedestal")),
  thePedestalWidth(p.getParameter<double>("pedestalWidth")),
  sca_peak_bin(p.getParameter<int>("scaPeakBin")),
  theComparatorTimeBinOffset(5),
  scaNoiseMode_(p.getParameter<std::string>("scaNoiseMode"))
{

  if(doCrosstalk_) {
    theCrosstalkGenerator = new CSCCrosstalkGenerator();
  }
  if(doNoise_) {
    if(scaNoiseMode_ == "file") {
      theScaNoiseGenerator = new CSCScaNoiseReader(thePedestal, thePedestalWidth);
    }
    else if(scaNoiseMode_ == "simple") {
      theScaNoiseGenerator = new CSCScaNoiseGaussian(theAnalogNoise, thePedestal, thePedestalWidth);
    }
    else {
      edm::LogError("CSCStripElectronicsSim") << "Bad value for SCA noise mode";
    }
  }

  fillAmpResponse();
}


CSCStripElectronicsSim::~CSCStripElectronicsSim() {
  if(doNoise_) {
    delete theScaNoiseGenerator;
  }
  if(doCrosstalk_) {
    delete theCrosstalkGenerator;
  }
}

void CSCStripElectronicsSim::initParameters() {
  TimeMe t("CSCStripEl:init");
  theLayerGeometry = theLayer->geometry();
  nElements = theLayerGeometry->numberOfStrips();
  theComparatorThreshold = 20.;
  if(doCrosstalk_) {
    float capacativeConstant = theLayerGeometry->length()/70;
    theCrosstalkGenerator->setParameters(capacativeConstant, 0., 0.02);
  }
}  


int CSCStripElectronicsSim::readoutElement(int strip) const {
  return theLayerGeometry->channel(strip);
}

CSCAnalogSignal CSCStripElectronicsSim::makeNoiseSignal(int element) {
  // assume the noise is measured every 50 ns
  int nNoiseBins = (int)((theSignalStopTime-theSignalStartTime)/sca_time_bin_size);
  std::vector<float> noiseBins(nNoiseBins);
  CSCAnalogSignal tmpSignal(element, sca_time_bin_size, noiseBins);
  theScaNoiseGenerator->noisify(layerId(), tmpSignal);
  tmpSignal *= theSpecs->chargePerCount();
  // now rebin it
  std::vector<float> binValues(theNumberOfSamples);
  for(int ibin=0; ibin < theNumberOfSamples; ++ibin) {
     binValues[ibin] = tmpSignal.getValue(ibin*theSamplingTime);
  }
  CSCAnalogSignal finalSignal(element, theSamplingTime, binValues, 0., theSignalStartTime);
  return finalSignal;
}


float CSCStripElectronicsSim::signalDelay(int element, float pos) const {
  // readout is on top edge of chamber, signal speed is 0.8 c
  // zero calibrated to chamber center
  float distance = -1. * pos;
  float speed = 0.8 * c_light / cm;
  return distance / speed;;
}


float CSCStripElectronicsSim::comparatorReading(const CSCAnalogSignal & signal,
                                                  float time) const {
   return std::min(signal.getValue(time), theComparatorSaturation)
       +  theComparatorRMSOffset*RandGaussQ::shoot();  
}


std::vector<CSCComparatorDigi> 
CSCStripElectronicsSim::runComparator() {
  // first, make a list of all the comparators we actually
  // need to run
  std::vector<CSCComparatorDigi> result;
  std::list<int> comparatorsWithSignal;
  MESignalMap::iterator signalMapItr;
  for(signalMapItr = theSignalMap.begin();
      signalMapItr != theSignalMap.end(); ++signalMapItr) {
	comparatorsWithSignal.push_back( ((*signalMapItr).first-1)/2 );
  }
  comparatorsWithSignal.unique();
  for(std::list<int>::iterator listItr = comparatorsWithSignal.begin();
      listItr != comparatorsWithSignal.end(); ++listItr) {
    int iComparator = *listItr;
    // find signal1 and signal2
    CSCAnalogSignal signal1 = find(readoutElement(iComparator*2 + 1));
    CSCAnalogSignal signal2 = find(readoutElement(iComparator*2 + 2));
    for(float time = theSignalStartTime; time < theSignalStopTime; time += theSamplingTime) {
      if(comparatorReading(signal1, time) > theComparatorThreshold
      || comparatorReading(signal2, time) > theComparatorThreshold) {
	 // wait a bit, so we can run the comparator at the signal peak
        float comparatorTime = time;
        time += theComparatorWait;
		  
        float height1 = comparatorReading(signal1, time);
        float height2 = comparatorReading(signal2, time);
        int output = 0; 
        int strip = 0;
	 // distrip logic; comparator output is for pairs of strips:
         // hit  bin  dec
         // x--- 100   4
         // -x-- 101   5
         // --x- 110   6
         // ---x 111   7
         CSCAnalogSignal mainSignal;
         // pick the higher of the two strips in the pair
         if(height1 > height2) {
           float leftStrip = 0.;
           if(iComparator > 0)  {
             leftStrip = comparatorReading(find(readoutElement(iComparator*2)), time); 
           }
           // if this strip is higher than either of its neighbors, make a comparator digi
           if(leftStrip < height1) {
             output = (leftStrip < height2);
             strip = iComparator*2 + 1;
             mainSignal = signal1;
           }
         } else {
           float rightStrip = 0.;
           if(iComparator*2+3 <= nElements) {
             rightStrip = comparatorReading(find(readoutElement(iComparator*2+3)), time);
           }
           if(rightStrip < height2) {
             output = (height1 < rightStrip);
             strip = iComparator*2 + 2;
             mainSignal = signal2;
           }
         }
         if(strip != 0) {

           // decide how long this comparator and neighboring ones will be locked for
           float lockingTime = time + theComparatorDeadTime;
           // really should be zero, but strip signal doesn't go negative yet
           float resetThreshold = theComparatorThreshold/2.;
           while(lockingTime < theNumberOfSamples*theSamplingTime-theComparatorWait
              && mainSignal.getValue(lockingTime) > resetThreshold) {
             lockingTime += theSamplingTime;
           }
           int timeBin = (int)(comparatorTime/theBunchSpacing) + theComparatorTimeBinOffset;
           CSCComparatorDigi newDigi(strip, output, timeBin);
           result.push_back(newDigi);
           time = lockingTime;
         }
       } // if over threshold
    } // loop over time samples
  }  // loop over comparators  
  // sort by time
  sort(result.begin(), result.end());
  return result;
}


void CSCStripElectronicsSim::getReadoutRange(int strip, int & minStrip, int & maxStrip) {
  // read out bunches of 16 strips
  int bunch = int((strip-1)/16);  //count from zero
  minStrip = 16 * bunch + 1;
  // if it's close to an edge, take the previous group
  if(strip-minStrip < 3 && strip > 16) {
    minStrip -= 16;
  }

  maxStrip = 16 * (bunch + 1);
  maxStrip = std::min(maxStrip, nElements);
  if(maxStrip - strip < 3 && maxStrip <= nElements-16) {
    maxStrip += 16;
  }
}


bool SortSignalsByTotal(const CSCAnalogSignal & s1,
                        const CSCAnalogSignal & s2) {
  return ( s1.getTotal() > s2.getTotal() );
}


void CSCStripElectronicsSim::fillDigis(CSCStripDigiCollection & digis, 
                                       CSCComparatorDigiCollection & comparators)
{
  TimeMe t("CSCStripEl:filldigi");
  if(doCrosstalk_) {
    addCrosstalk();
  } 
    
  // OK, in real life, an LCT causes a group of 16 strips in
  // all 6 layers to be read out for a 200 ns period.  We don't
  // have LCT's right now, so whenever a comparator fires, we
  // read out the five surrounding strips, then lock them from being
  // read out again for 200 ns.
  std::vector<float> lockedUntil(nElements+1, theSignalStartTime);

  std::vector<CSCComparatorDigi> comparatorOutputs = runComparator();

  // copy these to the result
  CSCComparatorDigiCollection::Range range(comparatorOutputs.begin(), comparatorOutputs.end());
  comparators.put(range, layerId());

  for(std::vector<CSCComparatorDigi>::const_iterator comparatorItr = comparatorOutputs.begin();
      comparatorItr != comparatorOutputs.end(); ++comparatorItr) {
    int channel = readoutElement((*comparatorItr).getStrip());
    float comparatorTime = ( (*comparatorItr).getTimeBin()-theComparatorTimeBinOffset ) * theBunchSpacing;
    LogDebug("CSCStripElectronicsSim") << "comparator " << channel <<" " << comparatorTime 
                    << " lockedUntil " << lockedUntil[channel];

    // make sure the readout isn't locked.
    if(comparatorTime > lockedUntil[channel]) { 
      CSCAnalogSignal signal = find(channel);

      // This strip will be the center of a cluster.
      // Next we have to make it so the max goes in
      // the sca_peak_bin.
      float peakTime = theSignalStartTime;
      float maxValue = 0.;
      // we can start when the comparator fires, and end in the time defined
      // by the Daq time between readouts
      float stoppingTime = std::min(comparatorTime+theDaqDeadTime, theSignalStopTime);

      // No, in fact we must start on an sca sampling time...

      //@@ The following code is wrong if sca time bin is not the same width as
      //@@ comparator-logic time bin, or 2x comparator time bin
      float startingTime = comparatorTime - theSamplingTime; // one comp time bin earlier
      if ( (static_cast<int>( startingTime ))%(static_cast<int>( sca_time_bin_size )) != 0 ) 
        startingTime -= theSamplingTime; // need to move to a multiple of sca sampling

      // Find the peak sca bin
      for(float scaTime = startingTime; scaTime < stoppingTime; scaTime += sca_time_bin_size) {
        float thisValue = signal.getValue(scaTime);
        if(thisValue > maxValue) {
          peakTime = scaTime;
          maxValue = thisValue;
        }
      }

      // Adjust so the SCA bins readout have max value in bin 'sca_peak_bin' (typically, 5th bin, bin=4)
      float sca_start_time = peakTime - sca_peak_bin*sca_time_bin_size;

      // create the digi for the strip that fired the comparator
      digis.insertDigi( layerId(), createDigi(channel, sca_start_time, false) );
      lockedUntil[channel] = comparatorTime + theDaqDeadTime;

      // now do the other strips
      int minStrip, maxStrip;
      getReadoutRange(channel, minStrip, maxStrip);
      for(int ichannel = minStrip; ichannel <= readoutElement(maxStrip); ++ichannel) {
        if(comparatorTime > lockedUntil[ichannel] && ichannel != channel) { 
          bool suppress = true;
          // see if this strip has a neighbor comparator firing in this time window
          for(std::vector<CSCComparatorDigi>::const_iterator otherComparatorItr = comparatorOutputs.begin();
              otherComparatorItr != comparatorOutputs.end(); ++otherComparatorItr) {
            // see if there's a comparator output on this strip, or nearby
            float otherComparatorTime = ((*otherComparatorItr).getTimeBin() - theComparatorTimeBinOffset) * theBunchSpacing;
            if(otherComparatorTime >= comparatorTime &&
               otherComparatorTime < comparatorTime+theDaqDeadTime) 
            {
              int leftComparator =  (readoutElement(ichannel-2)-1)/2;
              int rightComparator = (readoutElement(ichannel+2)-1)/2;
              int thisComparator =  (readoutElement(ichannel  )-1)/2;
              int otherDistrip = ((*otherComparatorItr).getStrip()-1) / 2;
              if(otherDistrip == thisComparator ||
                 otherDistrip == leftComparator ||
                 otherDistrip == rightComparator) {
                suppress = false;
              }
            }
          }
          if(!suppress) {
            digis.insertDigi(layerId(), createDigi(ichannel, sca_start_time, true) );
          }
          // even zero-suppressed strips get deadtime
          lockedUntil[ichannel] = (comparatorItr->getTimeBin()-theComparatorTimeBinOffset) * theBunchSpacing + theComparatorDeadTime;
        }
      } // loop over neighbor strips
    } // if main strip not locked
  } // loop over comparator outputs
}


void CSCStripElectronicsSim::addCrosstalk() {
  // this is needed so we can add a noise signal to the map
  // without messing up any iterators
  std::vector<CSCAnalogSignal> realSignals;
  realSignals.reserve(theSignalMap.size());
  for(MESignalMap::iterator mapI = theSignalMap.begin(); mapI != theSignalMap.end(); ++mapI) {
    realSignals.push_back((*mapI).second);
  }
  sort(realSignals.begin(), realSignals.end(), SortSignalsByTotal);
  for(std::vector<CSCAnalogSignal>::iterator realSignalItr = realSignals.begin();
      realSignalItr != realSignals.end(); ++realSignalItr) {
    CSCAnalogSignal crosstalkSignal = theCrosstalkGenerator->getCrosstalk(*realSignalItr);
    int thisStrip = (*realSignalItr).getElement();
    // add it to each neighbor
    if(thisStrip > 1) {
      find(readoutElement(thisStrip-1)).superimpose(crosstalkSignal);
    }
    if(thisStrip < nElements) {
      find(readoutElement(thisStrip+1)).superimpose(crosstalkSignal);
    }

    // Now subtract twice the crosstalk signal from the original signal
    crosstalkSignal *= -2.;
    find(readoutElement(thisStrip)).superimpose(crosstalkSignal);
  }
}


CSCStripDigi CSCStripElectronicsSim::createDigi(int channel, 
                                     float sca_start_time,
                                     bool addScaNoise) 
{
  // correct for Time of Flight
  float averageDistance = theLayer->surface().position().mag();
  float averageTimeOfFlight = averageDistance * cm / c_light; // Units of c_light: mm/ns
  double startTime = sca_start_time - averageTimeOfFlight;

  CSCAnalogSignal signal = find(channel);
  // fill in the sca information
  std::vector<int> scaCounts(nScaBins_);
  for(int scaBin = 0; scaBin < nScaBins_; ++scaBin) {
    scaCounts[scaBin] = static_cast< int >
      ( signal.getValue(startTime+scaBin*sca_time_bin_size) / theSpecs->chargePerCount() );
    if(addScaNoise) {
      scaCounts[scaBin] += static_cast< int >( RandGaussQ::shoot() * sca_noise 
						 / theSpecs->chargePerCount() );
    }
  }
  //int adcCounts = static_cast< int >( signal.getTotal() / theSpecs->chargePerCount() );
  CSCStripDigi newDigi(channel, scaCounts);
  if(theScaNoiseGenerator != 0) {
    theScaNoiseGenerator->addPedestal(layerId(), newDigi);
  }

  // do saturation of 12-bit ADC
  doSaturation(newDigi);

  addLinks(channelIndex(channel));
  //LogDebug("CSCStripElectronicsSim") << newDigi;
  //newDigi.print();

  return newDigi;
}


void CSCStripElectronicsSim::doSaturation(CSCStripDigi & digi)
{
  std::vector<int> scaCounts(digi.getADCCounts());
  for(unsigned scaBin = 0; scaBin < scaCounts.size(); ++scaBin) {
    scaCounts[scaBin] = std::min(scaCounts[scaBin], 4095);
  }
  digi.setADCCounts(scaCounts);
}



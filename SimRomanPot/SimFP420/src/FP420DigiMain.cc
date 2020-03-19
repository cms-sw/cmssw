///////////////////////////////////////////////////////////////////////////////
// File: FP420DigiMain.cc
// Date: 12.2006
// Description: FP420DigiMain for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"

#include "SimRomanPot/SimFP420/interface/ChargeDividerFP420.h"
#include "SimRomanPot/SimFP420/interface/ChargeDrifterFP420.h"

#include "DataFormats/FP420Digi/interface/HDigiFP420.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <gsl/gsl_sf_erf.h>

#include <iostream>
#include <vector>
using namespace std;

//#define CBOLTZ (1.38E-23)
//#define e_SI (1.6E-19)

FP420DigiMain::FP420DigiMain(const edm::ParameterSet &conf) : conf_(conf) {
  edm::LogInfo("SimRomanPotSimFP420") << "Creating a FP420DigiMain";
  ndigis = 0;
  verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
  theElectronPerADC = conf_.getParameter<double>("ElectronFP420PerAdc");
  theThreshold = conf_.getParameter<double>("AdcFP420Threshold");
  noNoise = conf_.getParameter<bool>("NoFP420Noise");
  addNoisyPixels = conf_.getParameter<bool>("AddNoisyPixels");
  thez420 = conf_.getParameter<double>("z420");
  thezD2 = conf_.getParameter<double>("zD2");
  thezD3 = conf_.getParameter<double>("zD3");
  theApplyTofCut = conf_.getParameter<bool>("ApplyTofCut");
  tofCut = conf_.getParameter<double>("LowtofCutAndTo200ns");
  //  sn0              = 3;// number of stations
  // pn0              = 6;// number of superplanes
  // rn0              = 3; // number of sensors in superlayer
  xytype = 2;

  edm::LogInfo("SimRomanPotSimFP420") << "theApplyTofCut=" << theApplyTofCut << " tofCut=" << tofCut << "\n"
                                      << "FP420DigiMain theElectronPerADC=" << theElectronPerADC
                                      << " theThreshold=" << theThreshold << " noNoise=" << noNoise;

  // X (or Y)define type of sensor (xytype=1 or 2 used to derive it: 1-Y, 2-X)
  // for every type there is normal pixel size=0.05 and Wide 0.400 mm
  Thick300 = 0.300;  // = 0.300 mm  normalized to 300micron Silicon
  // ENC= 2160.;             //          EquivalentNoiseCharge300um = 2160. +
  // other sources of noise
  ENC = 960.;  //          EquivalentNoiseCharge300um = 2160. + other sources of
               //          noise

  ldriftX = 0.050;          // in mm(xytype=1)
  ldriftY = 0.050;          // in mm(xytype=2)
  moduleThickness = 0.250;  // mm(xytype=1)(xytype=2)

  pitchY = 0.050;  // in mm(xytype=1)
  pitchX = 0.050;  // in mm(xytype=2)
  // numStripsY = 200;        // Y plate number of strips:200*0.050=10mm
  // (xytype=1) numStripsX = 400;        // X plate number of
  // strips:400*0.050=20mm (xytype=2)
  numStripsY = 144;  // Y plate number of strips:144*0.050=7.2mm (xytype=1)
  numStripsX = 160;  // X plate number of strips:160*0.050=8.0mm (xytype=2)

  pitchYW = 0.400;  // in mm(xytype=1)
  pitchXW = 0.400;  // in mm(xytype=2)
  // numStripsYW = 50;        // Y plate number of W strips:50 *0.400=20mm
  // (xytype=1) - W have ortogonal projection numStripsXW = 25;        // X
  // plate number of W strips:25 *0.400=10mm (xytype=2) - W have ortogonal
  // projection
  numStripsYW = 20;  // Y plate number of W strips:20 *0.400=8.0mm (xytype=1) - W
                     // have ortogonal projection
  numStripsXW = 18;  // X plate number of W strips:18 *0.400=7.2mm (xytype=2) - W
                     // have ortogonal projection

  //  tofCut = 1350.;           // Cut on the particle TOF range  = 1380 - 1500
  elossCut = 0.00003;  // Cut on the particle TOF   = 100 or 50

  edm::LogInfo("SimRomanPotSimFP420") << "FP420DigiMain moduleThickness=" << moduleThickness;

  float noiseRMS = ENC * moduleThickness / Thick300;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Y:
  if (xytype == 1) {
    numStrips = numStripsY;  // Y plate number of strips:200*0.050=10mm -->
                             // 100*0.100=10mm
    pitch = pitchY;
    ldrift = ldriftX;          // because drift is in local coordinates which 90 degree
                               // rotated ( for correct timeNormalization & noiseRMS
                               // calculations)
    numStripsW = numStripsYW;  // Y plate number of strips:200*0.050=10mm -->
                               // 100*0.100=10mm
    pitchW = pitchYW;
  }
  // X:
  if (xytype == 2) {
    numStrips = numStripsX;  // X plate number of strips:400*0.050=20mm -->
                             // 200*0.100=20mm
    pitch = pitchX;
    ldrift = ldriftY;          // because drift is in local coordinates which 90 degree
                               // rotated ( for correct timeNormalization & noiseRMS
                               // calculations)
    numStripsW = numStripsXW;  // X plate number of strips:400*0.050=20mm -->
                               // 200*0.100=20mm
    pitchW = pitchXW;
  }

  theHitDigitizerFP420 =
      new HitDigitizerFP420(moduleThickness, ldrift, ldriftY, ldriftX, thez420, thezD2, thezD3, verbosity);
  int numPixels = numStrips * numStripsW;
  theGNoiseFP420 = new GaussNoiseFP420(numPixels, noiseRMS, theThreshold, addNoisyPixels, verbosity);
  theZSuppressFP420 = new ZeroSuppressFP420(conf_, noiseRMS / theElectronPerADC);
  thePileUpFP420 = new PileUpFP420();
  theDConverterFP420 = new DigiConverterFP420(theElectronPerADC, verbosity);

  edm::LogInfo("SimRomanPotSimFP420") << "FP420DigiMain end of constructor";
}

FP420DigiMain::~FP420DigiMain() {
  delete theGNoiseFP420;
  delete theZSuppressFP420;
  delete theHitDigitizerFP420;
  delete thePileUpFP420;
  delete theDConverterFP420;
}

//  Run the algorithm
//  ------------------

vector<HDigiFP420> FP420DigiMain::run(const std::vector<PSimHit> &input, const G4ThreeVector &bfield, unsigned int iu) {
  thePileUpFP420->reset();
  bool first = true;

  // main loop
  //
  // First: loop on the SimHits
  //
  std::vector<PSimHit>::const_iterator simHitIter = input.begin();
  std::vector<PSimHit>::const_iterator simHitIterEnd = input.end();
  for (; simHitIter != simHitIterEnd; ++simHitIter) {
    const PSimHit ihit = *simHitIter;

    if (first) {
      //       detID = ihit.detUnitId();
      first = false;
    }

    // main part here:
    double losenergy = ihit.energyLoss();
    float tof = ihit.tof();

    if (verbosity > 1) {
      unsigned int unitID = ihit.detUnitId();
      std::cout << " *******FP420DigiMain:                           intindex= " << iu << std::endl;
      std::cout << " tof= " << tof << "  EnergyLoss= " << losenergy << "  pabs= " << ihit.pabs() << std::endl;
      std::cout << " EntryLocalP x= " << ihit.entryPoint().x() << " EntryLocalP y= " << ihit.entryPoint().y()
                << std::endl;
      std::cout << " ExitLocalP x= " << ihit.exitPoint().x() << " ExitLocalP y= " << ihit.exitPoint().y() << std::endl;
      std::cout << " EntryLocalP z= " << ihit.entryPoint().z() << " ExitLocalP z= " << ihit.exitPoint().z()
                << std::endl;
      std::cout << " unitID= " << unitID << "  xytype= " << xytype << " particleType= " << ihit.particleType()
                << " trackId= " << ihit.trackId() << std::endl;
      //    std::cout << " det= " << det << " sector= " << sector << "  zmodule=
      //    " << zmodule << std::endl;
      std::cout << "  bfield= " << bfield << std::endl;
    }
    if (verbosity > 0) {
      std::cout << " *******FP420DigiMain:                           theApplyTofCut= " << theApplyTofCut << std::endl;
      std::cout << " tof= " << tof << "  EnergyLoss= " << losenergy << "  pabs= " << ihit.pabs() << std::endl;
      std::cout << " particleType= " << ihit.particleType() << std::endl;
    }

    if ((!(theApplyTofCut) || (theApplyTofCut && abs(tof) > tofCut && abs(tof) < (tofCut + 200.))) &&
        losenergy > elossCut) {
      if (verbosity > 0)
        std::cout << " inside tof: OK " << std::endl;

      HitDigitizerFP420::hit_map_type _temp = theHitDigitizerFP420->processHit(
          ihit, bfield, xytype, numStrips, pitch, numStripsW, pitchW, moduleThickness, verbosity);

      thePileUpFP420->add(_temp, ihit, verbosity);
    }
    // main part end (AZ)
  }  // for
  // main loop end (AZ)

  PileUpFP420::signal_map_type theSignal = thePileUpFP420->dumpSignal();
  PileUpFP420::HitToDigisMapType theLink = thePileUpFP420->dumpLink();

  PileUpFP420::signal_map_type afterNoise;

  if (noNoise) {
    afterNoise = theSignal;
  } else {
    afterNoise = theGNoiseFP420->addNoise(theSignal);
    //    add_noise();
  }
  digis.clear();

  push_digis(theZSuppressFP420->zeroSuppress(theDConverterFP420->convert(afterNoise), verbosity), theLink, afterNoise);

  return digis;  // to HDigiFP420
}

void FP420DigiMain::push_digis(const DigitalMapType &dm,
                               const HitToDigisMapType &htd,
                               const PileUpFP420::signal_map_type &afterNoise) {
  //
  if (verbosity > 0) {
    std::cout << " ****FP420DigiMain: push_digis start: do for loop dm.size()=" << dm.size() << std::endl;
  }

  digis.reserve(50);
  digis.clear();
  //   link_coll.clear();
  for (DigitalMapType::const_iterator i = dm.begin(); i != dm.end(); i++) {
    // Load digis
    // push to digis the content of first and second words of HDigiFP420 vector
    // for every strip pointer (*i)
    digis.push_back(HDigiFP420((*i).first, (*i).second));
    ndigis++;
    // very useful check:
    if (verbosity > 0) {
      std::cout << " ****FP420DigiMain:push_digis:  ndigis = " << ndigis << std::endl;
      std::cout << "push_digis: strip  = " << (*i).first << "  adc = " << (*i).second << std::endl;
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////

  // reworked to access the fraction of amplitude per simhit

  for (HitToDigisMapType::const_iterator mi = htd.begin(); mi != htd.end(); mi++) {
    //
    if (dm.find((*mi).first) != dm.end()) {
      // --- For each channel, sum up the signals from a simtrack
      //
      map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
      for (std::vector<std::pair<const PSimHit *, Amplitude>>::const_iterator simul = (*mi).second.begin();
           simul != (*mi).second.end();
           simul++) {
        totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
      }
      /*
      //--- include the noise as well

      PileUpFP420::signal_map_type& temp1 =
      const_cast<PileUpFP420::signal_map_type&>(afterNoise); float
      totalAmplitude1 = temp1[(*mi).first];

      //--- digisimlink
      for (std::map<const PSimHit *, Amplitude>::const_iterator iter =
      totalAmplitudePerSimHit.begin(); iter != totalAmplitudePerSimHit.end();
      iter++){

      float threshold = 0.;
      if (totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1 >= threshold) {
      float fraction = totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1;

      //Noise fluctuation could make fraction>1. Unphysical, set it by hand.
      if(fraction >1.) fraction = 1.;

      link_coll.push_back(StripDigiSimLink( (*mi).first,   //channel
      ((*iter).first)->trackId(), //simhit
      fraction));    //fraction
      }//if
      }//for
      */
      //
    }  // if
  }    // for
}

#ifndef SimPPS_RPDigiProducer_RP_DET_DIGITIZER_H
#define SimPPS_RPDigiProducer_RP_DET_DIGITIZER_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"
#include "SimPPS/RPDigiProducer/plugins/RPHitChargeConverter.h"
#include "SimPPS/RPDigiProducer/plugins/RPVFATSimulator.h"
#include "SimPPS/RPDigiProducer/plugins/RPDisplacementGenerator.h"
#include "SimPPS/RPDigiProducer/plugins/RPGaussianTailNoiseAdder.h"
#include "SimPPS/RPDigiProducer/plugins/RPPileUpSignals.h"

#include <vector>
#include <string>

namespace CLHEP {
  class HepRandomEngine;
}
class CTPPSRPAlignmentCorrectionsData;
class CTPPSGeometry;

class RPDetDigitizer {
public:
  RPDetDigitizer(const edm::ParameterSet &params,
                 CLHEP::HepRandomEngine &eng,
                 RPDetId det_id,
                 const CTPPSRPAlignmentCorrectionsData *alignments,
                 const CTPPSGeometry &geom);
  void run(const std::vector<PSimHit> &input,
           const std::vector<int> &input_links,
           std::vector<TotemRPDigi> &output_digi,
           simromanpot::DigiPrimaryMapType &output_digi_links);

private:
  std::unique_ptr<RPGaussianTailNoiseAdder> theRPGaussianTailNoiseAdder;
  std::unique_ptr<RPPileUpSignals> theRPPileUpSignals;
  std::unique_ptr<RPHitChargeConverter> theRPHitChargeConverter;
  std::unique_ptr<RPVFATSimulator> theRPVFATSimulator;
  std::unique_ptr<RPDisplacementGenerator> theRPDisplacementGenerator;

private:
  int numStrips_;
  double theNoiseInElectrons;   // Noise (RMS) in units of electrons.
  double theStripThresholdInE;  // Strip noise treshold in electorns.
  bool noNoise_;                //if the nos is included
  RPDetId det_id_;
  bool misalignment_simulation_on_;
  int verbosity_;
  bool links_persistence_;
};

#endif  //SimCTPPS_RPDigiProducer_RP_DET_DIGITIZER_H

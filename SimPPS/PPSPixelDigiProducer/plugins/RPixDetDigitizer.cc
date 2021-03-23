#include "SimPPS/PPSPixelDigiProducer/interface/RPixDetDigitizer.h"

RPixDetDigitizer::RPixDetDigitizer(const edm::ParameterSet &params,
                                   CLHEP::HepRandomEngine &eng,
                                   uint32_t det_id,
                                   const PPSPixelTopology &ppt)
    : det_id_(det_id) {
  verbosity_ = params.getParameter<int>("RPixVerbosity");
  theNoiseInElectrons = params.getParameter<double>("RPixEquivalentNoiseCharge");
  thePixelThresholdInE = params.getParameter<double>("RPixDummyROCThreshold");
  noNoise = params.getParameter<bool>("RPixNoNoise");

  links_persistence_ = params.getParameter<bool>("CTPPSPixelDigiSimHitRelationsPersistence");

  theRPixPileUpSignals = std::make_unique<RPixPileUpSignals>(params, det_id_);
  theRPixDummyROCSimulator = std::make_unique<RPixDummyROCSimulator>(params, det_id_);
  theRPixHitChargeConverter = std::make_unique<RPixHitChargeConverter>(params, eng, det_id_, ppt);
}

RPixDetDigitizer::~RPixDetDigitizer() {}

void RPixDetDigitizer::run(const std::vector<PSimHit> &input,
                           const std::vector<int> &input_links,
                           std::vector<CTPPSPixelDigi> &output_digi,
                           std::vector<std::vector<std::pair<int, double> > > &output_digi_links,
                           const CTPPSPixelGainCalibrations *pcalibrations,
                           const PPSPixelTopology *pixelTopology) {
  if (verbosity_)
    edm::LogInfo("PPS") << "RPixDetDigitizer " << det_id_ << " received input.size()=" << input.size();
  theRPixPileUpSignals->reset();
  bool links_persistence_checked = links_persistence_ && input_links.size() == input.size();
  int input_size = input.size();
  for (int i = 0; i < input_size; ++i) {
    std::map<unsigned short, double> the_pixel_charge_map;
    the_pixel_charge_map = theRPixHitChargeConverter->processHit(input[i], *pixelTopology);

    if (verbosity_)
      edm::LogInfo("PPS") << "RPixDetDigitizer " << det_id_ << " returned hits=" << the_pixel_charge_map.size();
    if (links_persistence_checked)
      theRPixPileUpSignals->add(the_pixel_charge_map, input_links[i]);
    else
      theRPixPileUpSignals->add(the_pixel_charge_map, 0);
  }
  const std::map<unsigned short, double> &theSignal = theRPixPileUpSignals->dumpSignal();
  std::map<unsigned short, std::vector<std::pair<int, double> > > &theSignalProvenance =
      theRPixPileUpSignals->dumpLinks();
  std::map<unsigned short, double> afterNoise;
  afterNoise = theSignal;
  theRPixDummyROCSimulator->ConvertChargeToHits(
      afterNoise, theSignalProvenance, output_digi, output_digi_links, pcalibrations);
}

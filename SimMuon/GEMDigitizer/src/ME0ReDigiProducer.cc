#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimMuon/GEMDigitizer/interface/ME0ReDigiProducer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"
#include <sstream>
#include <string>
#include <map>
#include <vector>


ME0ReDigiProducer::ME0ReDigiProducer(const edm::ParameterSet& ps)
{
  produces<ME0DigiPreRecoCollection>();

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "ME0ReDigiProducer::ME0PreRecoDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  std::string collection_(ps.getParameter<std::string>("inputCollection"));

  token_ = consumes<ME0DigiPreRecoCollection>(edm::InputTag(collection_));
  timeResolution_ = ps.getParameter<double>("timeResolution");
  minBunch_ = ps.getParameter<int>("minBunch");
  maxBunch_ = ps.getParameter<int>("maxBunch");
  smearTiming_ = ps.getParameter<bool>("smearTiming");
  discretizeTiming_  = ps.getParameter<bool>("discretizeTiming");
  radialResolution_ = ps.getParameter<double>("radialResolution");
  smearRadial_ = ps.getParameter<bool>("smearRadial");
  oldXResolution_ = ps.getParameter<double>("oldXResolution");
  newXResolution_ = ps.getParameter<double>("newXResolution");
  newYResolution_ = ps.getParameter<double>("newYResolution");
  discretizeX_ = ps.getParameter<bool>("discretizeX");
  reDigitizeOnlyMuons_ = ps.getParameter<bool>("reDigitizeOnlyMuons");
  reDigitizeNeutronBkg_ = ps.getParameter<bool>("reDigitizeNeutronBkg");
  instLumi_ = ps.getParameter<double>("instLumi");
}


ME0ReDigiProducer::~ME0ReDigiProducer()
{
}


void ME0ReDigiProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup)
{
  // set geometry
  edm::ESHandle<ME0Geometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_= &*hGeom;

  LogDebug("ME0ReDigiProducer")
    << "Extracting central TOFs:" << std::endl;
  // get the central TOFs for the eta partitions
  for(auto &roll: geometry_->etaPartitions()){
    const ME0DetId detId(roll->id());
    if (detId.chamber() != 1 or detId.region() != 1) continue;
    const LocalPoint centralLP(0., 0., 0.);
    const GlobalPoint centralGP(roll->toGlobal(centralLP));
    const float centralTOF(centralGP.mag() / 29.98); //speed of light
    centralTOF_.push_back(centralTOF);
    LogDebug("ME0ReDigiProducer")
      << "ME0DetId " << detId << " central TOF " << centralTOF << std::endl;
  }
  nPartitions_ = centralTOF_.size()/6;
}


void ME0ReDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::Handle<ME0DigiPreRecoCollection> input_digis;
  e.getByToken(token_, input_digis);

  std::unique_ptr<ME0DigiPreRecoCollection> output_digis(new ME0DigiPreRecoCollection());

  // build the clusters
  buildDigis(*(input_digis.product()), *output_digis, engine);

  // store them in the event
  e.put(std::move(output_digis));
}


void ME0ReDigiProducer::buildDigis(const ME0DigiPreRecoCollection & input_digis,
                                   ME0DigiPreRecoCollection & output_digis,
                                   CLHEP::HepRandomEngine* engine)
{
  /*
    Starting form the incoming pseudo-digi, which has sigma_x=300um, sigma_t=sigma_R=0, do the following
    1A. Smear time using sigma_t by 7 ns (native resolution of GEM)
    1B. Correct the smeared time with the central arrival time for partition
    1C. Apply discretization: if the smeared time is outside the BX window (-12.5ns;+12.5ns),
    the hit should be assigned to the next (or previous) BX

    2A. Apply smearing in the radial direction (not in local Y!) sigma_R of 100 um
    2B. Apply discretization in radial direction to see which eta partition it belongs to.
    Assign the hit to have Y-position equal to the middle of the partition.

    3A. Apply smearing in x-direction (not required if we stick with sigma=300 um, which is what
    is used in pseudo-digis) to obtain desired x-resolution sigma_x_desired:
    - use gaussian smear with sigma_eff=sqrt(sigma_desired^2-300^2)
  */

  for(auto &roll: geometry_->etaPartitions()){
    const ME0DetId detId(roll->id());
    //const uint32_t rawId(detId.rawId());
    auto digis = input_digis.get(detId);
    for (auto d = digis.first; d != digis.second; ++d) {
      const ME0DigiPreReco me0Digi = *d;
      edm::LogVerbatim("ME0ReDigiProducer")
        << "Check detId " << detId << " digi " << me0Digi << std::endl;

      // selection
      if (reDigitizeOnlyMuons_ and fabs(me0Digi.pdgid()) != 13) continue;
      if (!reDigitizeNeutronBkg_ and !me0Digi.prompt()) continue;

      // scale for luminosity
      if (CLHEP::RandFlat::shoot(engine) > instLumi_*1.0/10) continue;

      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tPassed selection" << std::endl;

      // time resolution
      float newTof(me0Digi.tof());
      if (smearTiming_) newTof += CLHEP::RandGaussQ::shoot(engine, 0, timeResolution_);

      // arrival time in ns
      const float t0(centralTOF_[ nPartitions_ * (detId.layer() -1) + detId.roll() - 1 ]);
      const float correctedNewTof(newTof - t0);

      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tnew TOF " << newTof << " corrected new TOF " << correctedNewTof << std::endl;

      // calculate the new time in ns
      int newTime = correctedNewTof;
      if (discretizeTiming_){
        for (int iBunch = minBunch_ - 2; iBunch <= maxBunch_ + 2; ++iBunch){
          if (-12.5 + iBunch*25 < newTime and newTime <= 12.5 + iBunch*25){
            newTime = iBunch * 25;
            break;
          }
        }
      }

      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tBX " << newTime << std::endl;

      // calculate the position in global coordinates
      const LocalPoint oldLP(me0Digi.x(), me0Digi.y(), 0);
      const GlobalPoint oldGP(roll->toGlobal(oldLP));
      const GlobalPoint centralGP(roll->toGlobal(LocalPoint(0.,0.,0.)));
      const std::vector<float> parameters(roll->specs()->parameters());
      const float height(parameters[2]); // G4 uses half-dimensions!

      // smear the new radial with gaussian
      const float oldR(oldGP.perp());
      const float newR(CLHEP::RandGaussQ::shoot(engine, oldR, radialResolution_));

      // calculate the new position in local coordinates
      const GlobalPoint newGP(GlobalPoint::Cylindrical(newR, oldGP.phi(), oldGP.z()));

      // check if the smeared hit remains in its partition, or moves one up or down
      const float deltaY(newGP.y() - centralGP.y());
      int newRoll;
      if (deltaY > height)  newRoll = detId.roll() - 1;
      if (deltaY < -height) newRoll = detId.roll() + 1;
      else newRoll = detId.roll();

      // sanity-check
      if (newRoll == nPartitions_+1) newRoll = nPartitions_;
      if (newRoll == 0) newRoll = 1;

      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tnew roll " << newRoll << std::endl;

      // new hit has y coordinate in the center of the roll
      const float newY(0.);

      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tnew Y " << newY << std::endl;

      // new x position
      const float targetResolution(sqrt(newXResolution_*newXResolution_ - oldXResolution_ * oldXResolution_));
      float newX(CLHEP::RandGaussQ::shoot(engine, me0Digi.x(), targetResolution));

      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tnew X " << newX << std::endl;

      // discretize the new X
      if (discretizeX_){
        const LocalPoint lp(newX, newY, 0);
        int strip(roll->strip(lp));
        float stripF(float(strip) - 0.5);
        const LocalPoint newLP(roll->centreOfStrip(stripF));
        newX = newLP.x();
        edm::LogVerbatim("ME0ReDigiProducer")
          << "\t\tdiscretized X " << newX << std::endl;
      }

      // make a new ME0DetId
      ME0DigiPreReco out_digi(newX, newY, targetResolution, newYResolution_, me0Digi.corr(), newTime, me0Digi.pdgid(), me0Digi.prompt());
      ME0DetId out_detId(detId.region(), detId.layer(), detId.chamber(), newRoll);

      output_digis.insertDigi(out_detId, out_digi);
    }
  }
}

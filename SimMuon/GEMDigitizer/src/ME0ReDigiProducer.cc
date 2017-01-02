#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimMuon/GEMDigitizer/interface/ME0ReDigiProducer.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
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
  oldYResolution_ = ps.getParameter<double>("oldYResolution");
  newXResolution_ = ps.getParameter<double>("newXResolution");
  newYResolution_ = ps.getParameter<double>("newYResolution");
  discretizeX_ = ps.getParameter<bool>("discretizeX");
  discretizeY_ = ps.getParameter<bool>("discretizeY");
  reDigitizeOnlyMuons_ = ps.getParameter<bool>("reDigitizeOnlyMuons");
  reDigitizeNeutronBkg_ = ps.getParameter<bool>("reDigitizeNeutronBkg");
  rateFact_ = ps.getParameter<double>("rateFact");
  instLumiDefault_ = ps.getParameter<double>("instLumiDefault");
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
      if (reDigitizeOnlyMuons_ and std::abs(me0Digi.pdgid()) != 13) continue;
      if (!reDigitizeNeutronBkg_ and !me0Digi.prompt()) continue;

      // scale background hits for luminosity
      if (!me0Digi.prompt())
	  if (CLHEP::RandFlat::shoot(engine) > instLumi_*1.0/(instLumiDefault_*rateFact_)) continue;
      
      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tPassed selection" << std::endl;

      // time resolution
      float newTof(me0Digi.tof());
      if (me0Digi.prompt() and smearTiming_) newTof += CLHEP::RandGaussQ::shoot(engine, 0, timeResolution_);

      // arrival time in ns
      //const float t0(centralTOF_[ nPartitions_ * (detId.layer() -1) + detId.roll() - 1 ]);
      int index = nPartitions_ * (detId.layer() -1) + detId.roll() - 1;
      if(detId.roll() == 0) index = nPartitions_ * (detId.layer() -1) + detId.roll();
      edm::LogVerbatim("ME0ReDigiProducer")
	<<"size "<<centralTOF_.size()<<" nPartitions "<<nPartitions_<<" layer "<<detId.layer()<<" roll "<<detId.roll()<<" index "<<index<<std::endl;
      
      const float t0(centralTOF_[ index ]);      
      const float correctedNewTof(newTof - t0);

      edm::LogVerbatim("ME0ReDigiProducer")
        <<" t0 "<< t0 << " originalTOF " << me0Digi.tof() << "\tnew TOF " << newTof << " corrected new TOF " << correctedNewTof << std::endl;

      // calculate the new time in ns
      float newTime = correctedNewTof;
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

      float newR = oldR;
      if (me0Digi.prompt() and smearRadial_  and detId.roll() > 0)
	newR = CLHEP::RandGaussQ::shoot(engine, oldR, radialResolution_);
      
      // calculate the new position in local coordinates
      const GlobalPoint radialSmearedGP(GlobalPoint::Cylindrical(newR, oldGP.phi(), oldGP.z()));
      LocalPoint radialSmearedLP = roll->toLocal(radialSmearedGP);
      
      // new y position after smearing
      const float targetYResolution(sqrt(newYResolution_*newYResolution_ - oldYResolution_ * oldYResolution_));
      float newLPy = radialSmearedLP.y();
      if (me0Digi.prompt())
	newLPy = CLHEP::RandGaussQ::shoot(engine, radialSmearedLP.y(), targetYResolution);
      
      const ME0EtaPartition* newPart = roll;
      LocalPoint newLP(radialSmearedLP.x(), newLPy, radialSmearedLP.z());
      GlobalPoint newGP(newPart->toGlobal(newLP));
      	
      // check if digi moves one up or down roll
      int newRoll = detId.roll();
      if (newLP.y() > height)  --newRoll;
      if (newLP.y() < -height) ++newRoll;

      if (newRoll != detId.roll()){
	// check if new roll is possible
	if (newRoll < ME0DetId::minRollId || newRoll > ME0DetId::maxRollId){
	  newRoll = detId.roll();
	}
	else {	 
	  // roll changed, get new ME0EtaPartition
	  newPart = geometry_->etaPartition(ME0DetId(detId.region(), detId.layer(), detId.chamber(), newRoll));
	}

	// if new ME0EtaPartition fails or roll not changed
	if (!newPart or newRoll == detId.roll()){
	  newPart = roll;
	  // set local y to edge of etaPartition
	  if (newLP.y() > height)  newLP = LocalPoint(newLP.x(), height, newLP.z());
	  if (newLP.y() < -height) newLP = LocalPoint(newLP.x(), -height, newLP.z());	
	}
	else {// new partiton, get new local point
	  newLP = newPart->toLocal(newGP);
	}
      }	

      // smearing in X
      double newXResolutionCor = correctSigmaU(roll, newLP.y());
      
      // new x position after smearing
      const float targetXResolution(sqrt(newXResolutionCor*newXResolutionCor - oldXResolution_ * oldXResolution_));
      float newLPx = newLP.x();
      if (me0Digi.prompt())
	newLPx = CLHEP::RandGaussQ::shoot(engine, newLP.x(), targetXResolution);

      // update local point after x smearing
      newLP = LocalPoint(newLPx, newLP.y(), newLP.z());
      
      float newY(newLP.y());
      // new hit has y coordinate in the center of the roll when using discretizeY
      if (discretizeY_ and detId.roll() > 0) newY = 0;
      edm::LogVerbatim("ME0ReDigiProducer")
	<< "\tnew Y " << newY << std::endl;

      float newX(newLP.x());
      edm::LogVerbatim("ME0ReDigiProducer")
        << "\tnew X " << newX << std::endl;      
      // discretize the new X
      if (discretizeX_){
        int strip(newPart->strip(newLP));
        float stripF(float(strip) - 0.5);
        const LocalPoint newLP(newPart->centreOfStrip(stripF));
        newX = newLP.x();
        edm::LogVerbatim("ME0ReDigiProducer")
          << "\t\tdiscretized X " << newX << std::endl;
      }

      // make a new ME0DetId
      ME0DigiPreReco out_digi(newX, newY, targetXResolution, targetYResolution, me0Digi.corr(), newTime, me0Digi.pdgid(), me0Digi.prompt());

      output_digis.insertDigi(newPart->id(), out_digi);
    }
  }
}

double ME0ReDigiProducer::correctSigmaU(const ME0EtaPartition* roll, double y) {
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  auto& parameters(roll->specs()->parameters());
  double height(parameters[2]);       // height     = height from Center of Roll
  double rollRadius = top_->radius(); // rollRadius = Radius at Center of Roll
  double Rmax = rollRadius+height;    // MaxRadius  = Radius at top of Roll
  double Rx = rollRadius+y;           // y in [-height,+height]
  double sigma_u_new = Rx/Rmax*newXResolution_;
  return sigma_u_new;
}

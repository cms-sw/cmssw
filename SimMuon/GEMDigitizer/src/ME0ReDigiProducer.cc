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



ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry(const ME0Geometry* geometry, const unsigned int numberOfStrips, const unsigned int numberOfPartitions) {
	//First test geometry to make sure that it is compatible with our assumptions
	const auto& chambers = geometry->chambers();
	if(!chambers.size())
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - No ME0Chambers in geometry.";
	const auto* mainChamber = chambers.front();
	const unsigned int nLayers = chambers.front()->nLayers();
	if(!nLayers)
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - ME0Chamber has no layers.";
	const auto* mainLayer = mainChamber->layers()[0];
	if(!mainLayer->nEtaPartitions())
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - ME0Layer has no partitions.";
	if(mainLayer->nEtaPartitions() != 1)
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - This module is only compatitble with geometries that contain only one partiton per ME0Layer.";

	const auto* mainPartition = mainLayer->etaPartitions()[0];
	const TrapezoidalStripTopology * mainTopo = dynamic_cast<const TrapezoidalStripTopology*>(&mainPartition->topology());
	if(!mainTopo)
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - ME0 strip topology must be of type TrapezoidalStripTopology. This module cannot be used";

	for(const auto& chamber : geometry->chambers() ){
		if(chamber->nLayers() != int(nLayers))
			throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - Not all ME0Chambers have the same number of layers. This module cannot be used.";
		for(unsigned int iL = 0; iL < nLayers; ++iL){
			if(chamber->layers()[iL]->nEtaPartitions() != mainLayer->nEtaPartitions())
				throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - Not all ME0Layers have the same number of partitions. This module cannot be used.";
			if(chamber->layers()[iL]->etaPartitions()[0]->specs()->parameters() != mainPartition->specs()->parameters())
				throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - Not all ME0 ETA partitions have the same properties. This module cannot be used.";
			if(std::fabs(chamber->layers()[iL]->etaPartitions()[0]->position().z()) != std::fabs(mainChamber->layers()[iL]->etaPartitions()[0]->position().z()))
				throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - Not all ME0 ETA partitions in a single layer have the same Z position. This module cannot be used.";
		}
	}

	//Calculate radius to center of partition
	middleDistanceFromBeam = mainTopo->radius();

	//calculate the top of each eta partition, assuming equal distance in eta between partitions
	auto localTop     = LocalPoint(0,mainTopo->stripLength()/2);
	auto localBottom  = LocalPoint(0,-1*mainTopo->stripLength()/2);
	auto globalTop    = mainPartition->toGlobal(localTop);
	auto globalBottom = mainPartition->toGlobal(localBottom);
	double etaTop     = globalTop.eta();
	double etaBottom  = globalBottom.eta();
	double zBottom    = globalBottom.z();
	partionTops.reserve(numberOfPartitions);
	for(unsigned int iP = 0; iP < numberOfPartitions; ++iP){
		double eta = (etaTop -etaBottom)*double(iP + 1)/double(numberOfPartitions) + etaBottom;
		double distFromBeam = std::fabs(2 * zBottom /(std::exp(-1*eta) - std::exp(eta)  ));
		partionTops.push_back(distFromBeam - middleDistanceFromBeam);
		LogDebug("ME0ReDigiProducer::TemporaryGeometry") << "Top of new partition: " <<partionTops.back() << std::endl;
	}

	//Build topologies
	stripTopos.reserve(numberOfPartitions);
	const auto& mainPars = mainPartition->specs()->parameters();
	for(unsigned int iP = 0; iP < numberOfPartitions; ++iP){
		std::vector<float> params(4,0);

		//half width of trapezoid at local coordinate Y
		auto getWidth = [&] ( float locY ) -> float { return (mainPars[2]*(mainPars[1]+mainPars[0]) +locY*(mainPars[1] - mainPars[0]) )/(2*mainPars[2]);};

		params[0] = iP == 0 ?  mainPars[0] : getWidth(partionTops[iP -1]); // Half width of bottom of chamber
		params[1] = iP +1 == numberOfPartitions ?  mainPars[1] : getWidth(partionTops[iP]); // Half width of top of chamber
		params[2] = ((iP + 1 == numberOfPartitions ? localTop.y() : partionTops[iP] ) - (iP  == 0 ? localBottom.y() : partionTops[iP-1] ) )/2; // Half width of height of chamber
		params[3] = numberOfStrips;

		stripTopos.push_back(buildTopo(params));
	}

	//Get TOF at center of each partition
	tofs.resize(nLayers);
	LogDebug("ME0ReDigiProducer::TemporaryGeometry") << "TOF numbers [layer][partition]: " ;
	for(unsigned int iL = 0; iL < nLayers; ++iL){
		tofs[iL].resize(numberOfPartitions);
		for(unsigned int iP = 0; iP < numberOfPartitions; ++iP){
			const LocalPoint partCenter(0., getPartCenter(iP), 0.);
			const GlobalPoint centralGP(mainChamber->layers()[iL]->etaPartitions()[0]->toGlobal(partCenter));
			tofs[iL][iP] = (centralGP.mag() / 29.9792); //speed of light [cm/ns]
			LogDebug("ME0ReDigiProducer::TemporaryGeometry") << "["<<iL<<"]["<<iP<<"]="<< tofs[iL][iP] <<" "<<std::endl;
		}
	}
}

unsigned int ME0ReDigiProducer::TemporaryGeometry::findEtaPartition(float locY) const {
	unsigned int etaPart = stripTopos.size() -1;
	for(unsigned int iP = 0; iP < stripTopos.size(); ++iP ){
		if(locY <  partionTops[iP]) {etaPart = iP; break;}
	}
	return etaPart;
}

float ME0ReDigiProducer::TemporaryGeometry::getPartCenter(const unsigned int partIdx) const {return stripTopos[partIdx]->radius() - middleDistanceFromBeam ;}

ME0ReDigiProducer::TemporaryGeometry::~TemporaryGeometry() {
	for(auto * p : stripTopos) { delete p;}
}

TrapezoidalStripTopology * ME0ReDigiProducer::TemporaryGeometry::buildTopo(const std::vector<float>& _p) const {
	float b = _p[0];
	float B = _p[1];
	float h = _p[2];
	float r0 = h*(B + b)/(B - b);
	float striplength = h*2;
	float strips = _p[3];
	float pitch = (b + B)/strips;
	int nstrip =static_cast<int>(strips);

	LogDebug("ME0ReDigiProducer::TemporaryGeometry") << "New partition parameters: " <<
			"bottom width("<< 2*b <<") top width("<<2*B<<") height("<< 2*h <<") radius to center("<< r0 <<") nStrips("<< strips <<") pitch(" << pitch<<")"<< std::endl;

	return new TrapezoidalStripTopology(nstrip, pitch, striplength, r0);
}

ME0ReDigiProducer::ME0ReDigiProducer(const edm::ParameterSet& ps) :
		numberOfSrips      (ps.getParameter<unsigned int>("numberOfSrips")),
		numberOfPartitions (ps.getParameter<unsigned int>("numberOfPartitions")),
		neutronAcceptance  (ps.getParameter<double>("neutronAcceptance")),
		timeResolution     (ps.getParameter<double>("timeResolution")),
		minBXReadout       (ps.getParameter<int>("minBXReadout")),
		maxBXReadout       (ps.getParameter<int>("maxBXReadout")),
		layerReadout       (ps.getParameter<std::vector<int>>("layerReadout")),
		mergeDigis         (ps.getParameter<bool>("mergeDigis")),
		token(consumes<ME0DigiPreRecoCollection>(edm::InputTag(ps.getParameter<std::string>("inputCollection"))))
{
	produces<ME0DigiPreRecoCollection>();
	produces<ME0DigiPreRecoMap>();

	edm::Service<edm::RandomNumberGenerator> rng;
	if (!rng.isAvailable()){
		throw cms::Exception("Configuration")
		<< "ME0ReDigiProducer::ME0PreRecoDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
		<< "Add the service in the configuration file or remove the modules that require it.";
	}
	geometry = 0;
	tempGeo = 0;

	if(numberOfSrips == 0)
		throw cms::Exception("Setup") << "ME0ReDigiProducer::ME0PreRecoDigiProducer() - Must have at least one strip.";
	if(numberOfPartitions == 0)
		throw cms::Exception("Setup") << "ME0ReDigiProducer::ME0PreRecoDigiProducer() - Must have at least one partition.";
	if(neutronAcceptance < 0 )
		throw cms::Exception("Setup") << "ME0ReDigiProducer::ME0PreRecoDigiProducer() - neutronAcceptance must be >= 0.";




}


ME0ReDigiProducer::~ME0ReDigiProducer()
{
	if(tempGeo) delete tempGeo;
}


void ME0ReDigiProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup)
{
	// set geometry
	edm::ESHandle<ME0Geometry> hGeom;
	eventSetup.get<MuonGeometryRecord>().get(hGeom);
	geometry= &*hGeom;

	LogDebug("ME0ReDigiProducer")
	<< "Building temporary geometry:" << std::endl;
	tempGeo = new TemporaryGeometry(geometry,numberOfSrips,numberOfPartitions);
	LogDebug("ME0ReDigiProducer")
	<< "Done building temporary geometry!" << std::endl;

	if(tempGeo->numLayers() != layerReadout.size() )
		throw cms::Exception("Configuration") << "ME0ReDigiProducer::beginRun() - The geoemtry has "<<tempGeo->numLayers()
		<< " layers, but the readout of "<<layerReadout.size() << " were specified with the layerReadout parameter."  ;

}


void ME0ReDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
	edm::Service<edm::RandomNumberGenerator> rng;
	CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

	edm::Handle<ME0DigiPreRecoCollection> input_digis;
	e.getByToken(token, input_digis);

	std::unique_ptr<ME0DigiPreRecoCollection> output_digis(new ME0DigiPreRecoCollection());
	std::unique_ptr<ME0DigiPreRecoMap>       output_digimap(new ME0DigiPreRecoMap());

	// build the digis
	buildDigis(*(input_digis.product()),
			*output_digis,
			*output_digimap,
			engine);

	// store them in the event
	e.put(std::move(output_digis));
	e.put(std::move(output_digimap));

//	produces< edm::ValueMap<edm::Ref(ME0DigiPreReco)> >("ptrToNewDigi");

}


void ME0ReDigiProducer::buildDigis(const ME0DigiPreRecoCollection & input_digis,
		ME0DigiPreRecoCollection & output_digis,
		ME0DigiPreRecoMap & output_digimap,
		CLHEP::HepRandomEngine* engine)
{

	LogDebug("ME0ReDigiProducer::buildDigis") << "Begin bulding digis."<<std::endl;
	ME0DigiPreRecoCollection::DigiRangeIterator me0dgIt;
	for (me0dgIt = input_digis.begin(); me0dgIt != input_digis.end();
			++me0dgIt){

		const auto& me0Id = (*me0dgIt).first;
		LogTrace("ME0ReDigiProducer::buildDigis") << "Starting with chamber: "<< me0Id<<std::endl;

		//setup map for this chamber
		typedef std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, unsigned int > > > ChamberDigiMap;
		ChamberDigiMap chDigiMap;
		//fills map...returns -1 if digi is not already in the map
		auto fillDigiMap = [&] (unsigned int bx, unsigned int part, unsigned int strip, unsigned int currentIDX) -> int {
			auto it1 = chDigiMap.find(bx);
			if (it1 == chDigiMap.end()){
				chDigiMap[bx][part][strip] = currentIDX;
				return -1;
			}
			auto it2 = it1->second.find(part);
			if (it2 == it1->second.end()){
				it1->second[part][strip] = currentIDX;
				return -1;
			}
			auto it3 = it2->second.find(strip);
			if (it3 == it2->second.end()){
				it2->second[strip] = currentIDX;
				return -1;
			}
			return it3->second;
		};

		int newDigiIdx = 0;
		const ME0DigiPreRecoCollection::Range& range = (*me0dgIt).second;
		for (ME0DigiPreRecoCollection::const_iterator digi = range.first;
				digi != range.second;digi++) {
			LogTrace("ME0ReDigiProducer::buildDigis") << std::endl<< "(" <<digi->x() <<","<< digi->y()<<","<<digi->tof()<<","<<digi->pdgid()<<","<<digi->prompt()<<")-> ";

			//If we don't readout this layer skip
			if(!layerReadout[me0Id.layer() -1 ]) {
				output_digimap.insertDigi(me0Id, -1);
				continue;
			}

			//if neutron and we are filtering skip
			if(!digi->prompt() && neutronAcceptance < 1.0 )
				if (CLHEP::RandFlat::shoot(engine) > neutronAcceptance){
					output_digimap.insertDigi(me0Id, -1);
					continue;
				}

			const unsigned int partIdx = tempGeo->findEtaPartition(digi->y());
			LogTrace("ME0ReDigiProducer::buildDigis") << partIdx <<" ";
			float tof = digi->tof() + (timeResolution < 0 ? 0.0 : CLHEP::RandGaussQ::shoot(engine, 0, timeResolution));
			const float partMeanTof = tempGeo->getCentralTOF(me0Id,partIdx);
			//convert to relative to partition
			tof -= partMeanTof;
			const int bxIdx = std::round(tof/25.0);
			LogTrace("ME0ReDigiProducer::buildDigis") << tof <<"("<<bxIdx<<") ";
			//filter if outside of readout window
			if(bxIdx < minBXReadout) {output_digimap.insertDigi(me0Id, -1); continue; }
			if(bxIdx > maxBXReadout) {output_digimap.insertDigi(me0Id, -1); continue; }
			tof = bxIdx*25;

			//get coordinates and errors
			const float partCenter = tempGeo->getPartCenter(partIdx);
			const auto* topo = tempGeo->getTopo(partIdx);

			//find channel
			const LocalPoint partLocalPoint(digi->x(), digi->y() - partCenter ,0.);
			const int strip = topo->channel(partLocalPoint);
			const float stripF = float(strip)+0.5;

			LogTrace("ME0ReDigiProducer::buildDigis") << "("<<bxIdx<<","<<partIdx<<","<<strip<<") ";

			//If we are merging check to see if it already exists
			if(mergeDigis){
				int matchIDX = fillDigiMap(bxIdx,partIdx,strip,newDigiIdx);
				if(matchIDX >= 0){
					output_digimap.insertDigi(me0Id, matchIDX);
					continue;
				}
			}

			//get digitized location
			LocalPoint  digiPartLocalPoint = topo->localPosition(stripF);
			LocalError  digiPartLocalError = topo->localError(stripF, 1./sqrt(12.));
			LocalPoint  digiChamberLocalPoint(digiPartLocalPoint.x(),digiPartLocalPoint.y() + partCenter,0);

			//Digis store sigmaX,sigmaY, correlationCoef
			const float sigmaX = std::sqrt(digiPartLocalError.xx());
			const float sigmaY = std::sqrt(digiPartLocalError.yy());
			const float corrCoef = digiPartLocalError.xy() /(sigmaX*sigmaY);

			//Fill in the new collection
			ME0DigiPreReco out_digi(digiChamberLocalPoint.x(), digiChamberLocalPoint.y(),
					sigmaX, sigmaY, corrCoef, tof, digi->pdgid(), digi->prompt());
			output_digis.insertDigi(me0Id, out_digi);

			// store index of previous detid and digi
			output_digimap.insertDigi(me0Id, newDigiIdx);
			newDigiIdx++;

			LogTrace("ME0ReDigiProducer::buildDigis") << "("<<digiChamberLocalPoint.x()<<","<<digiChamberLocalPoint.y()<<","<<sigmaX<<","<<sigmaY<<","<< tof<<") ";
		}

		chDigiMap.clear();


	}

}

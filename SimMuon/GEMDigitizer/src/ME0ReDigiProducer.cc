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
#include "CLHEP/Units/PhysicalConstants.h"
#include <sstream>
#include <string>
#include <map>
#include <vector>



ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry(const ME0Geometry* geometry, const unsigned int numberOfStrips, const unsigned int numberOfPartitions) {
	//First test geometry to make sure that it is compatible with our assumptions
	const auto& chambers = geometry->chambers();
	if(chambers.empty())
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - No ME0Chambers in geometry.";
	const auto* mainChamber = chambers.front();
	const unsigned int nLayers = chambers.front()->nLayers();
	if(!nLayers)
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - ME0Chamber has no layers.";
	const auto* mainLayer = mainChamber->layers()[0];
	if(!mainLayer->nEtaPartitions())
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - ME0Layer has no partitions.";
	if(mainLayer->nEtaPartitions() != 1)
		throw cms::Exception("Setup") << "ME0ReDigiProducer::TemporaryGeometry::TemporaryGeometry() - This module is only compatitble with geometries that contain only one partition per ME0Layer.";

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
	const auto localTop     = LocalPoint(0,mainTopo->stripLength()/2);
	const auto localBottom  = LocalPoint(0,-1*mainTopo->stripLength()/2);
	const auto globalTop    = mainPartition->toGlobal(localTop);
	const auto globalBottom = mainPartition->toGlobal(localBottom);
	const double etaTop     = globalTop.eta();
	const double etaBottom  = globalBottom.eta();
	const double zBottom    = globalBottom.z();

	//Build topologies
	partitionTops.reserve(numberOfPartitions);
	stripTopos.reserve(numberOfPartitions);
	const auto& mainPars = mainPartition->specs()->parameters();
	for(unsigned int iP = 0; iP < numberOfPartitions; ++iP){
		const double eta = (etaTop -etaBottom)*double(iP + 1)/double(numberOfPartitions) + etaBottom;
		const double distFromBeam = std::fabs(zBottom /std::sinh(eta));
		partitionTops.push_back(distFromBeam - middleDistanceFromBeam);
		LogDebug("ME0ReDigiProducer::TemporaryGeometry") << "Top of new partition: " <<partitionTops.back() << std::endl;

		std::vector<float> params(4,0);

		//half width of trapezoid at local coordinate Y
		auto getWidth = [&] ( float locY ) -> float { return (mainPars[2]*(mainPars[1]+mainPars[0]) +locY*(mainPars[1] - mainPars[0]) )/(2*mainPars[2]);};

		params[0] = iP == 0 ?  mainPars[0] : getWidth(partitionTops[iP -1]); // Half width of bottom of chamber
		params[1] = iP +1 == numberOfPartitions ?  mainPars[1] : getWidth(partitionTops[iP]); // Half width of top of chamber
		params[2] = ((iP + 1 == numberOfPartitions ? localTop.y() : partitionTops[iP] ) - (iP  == 0 ? localBottom.y() : partitionTops[iP-1] ) )/2; // Half width of height of chamber
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
			tofs[iL][iP] = (centralGP.mag() / (CLHEP::c_light/CLHEP::cm)); //speed of light [cm/ns]
			LogDebug("ME0ReDigiProducer::TemporaryGeometry") << "["<<iL<<"]["<<iP<<"]="<< tofs[iL][iP] <<" "<<std::endl;
		}
	}
}

unsigned int ME0ReDigiProducer::TemporaryGeometry::findEtaPartition(float locY) const {
	unsigned int etaPart = stripTopos.size() -1;
	for(unsigned int iP = 0; iP < stripTopos.size(); ++iP ){
		if(locY <  partitionTops[iP]) {etaPart = iP; break;}
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
		bxWidth            (25.0),
		useCusGeoFor1PartGeo(ps.getParameter<bool>("useCusGeoFor1PartGeo")),
		usePads(ps.getParameter<bool>("usePads")),
		numberOfStrips      (ps.getParameter<unsigned int>("numberOfStrips")),
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
	useBuiltinGeo = true;

	if(useCusGeoFor1PartGeo){
	        if (usePads)
		        numberOfStrips = numberOfStrips/2;
		if(numberOfStrips == 0)
			throw cms::Exception("Setup") << "ME0ReDigiProducer::ME0PreRecoDigiProducer() - Must have at least one strip if using custom geometry.";
		if(numberOfPartitions == 0)
			throw cms::Exception("Setup") << "ME0ReDigiProducer::ME0PreRecoDigiProducer() - Must have at least one partition if using custom geometry.";
	}

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

	const auto& chambers = geometry->chambers();
	if(chambers.empty())
		throw cms::Exception("Setup") << "ME0ReDigiProducer::beginRun() - No ME0Chambers in geometry.";

	const unsigned int nLayers = chambers.front()->nLayers();
	if(!nLayers) throw cms::Exception("Setup") << "ME0ReDigiProducer::beginRun() - No layers in ME0 geometry.";

	const unsigned int nPartitions = chambers.front()->layers()[0]->nEtaPartitions();

	if(useCusGeoFor1PartGeo && nPartitions == 1){
		useBuiltinGeo = false;
	}

	if(useBuiltinGeo){
		if(nLayers != layerReadout.size() )
			throw cms::Exception("Configuration") << "ME0ReDigiProducer::beginRun() - The geometry has "<<nLayers
			<< " layers, but the readout of "<<layerReadout.size() << " were specified with the layerReadout parameter."  ;
		fillCentralTOFs();
	} else {
		LogDebug("ME0ReDigiProducer")
				<< "Building temporary geometry:" << std::endl;
		tempGeo = new TemporaryGeometry(geometry,numberOfStrips,numberOfPartitions);
		LogDebug("ME0ReDigiProducer")
		<< "Done building temporary geometry!" << std::endl;

		if(tempGeo->numLayers() != layerReadout.size() )
			throw cms::Exception("Configuration") << "ME0ReDigiProducer::beginRun() - The geometry has "<<tempGeo->numLayers()
			<< " layers, but the readout of "<<layerReadout.size() << " were specified with the layerReadout parameter."  ;
	}
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
}


void ME0ReDigiProducer::buildDigis(const ME0DigiPreRecoCollection & input_digis,
		ME0DigiPreRecoCollection & output_digis,
		ME0DigiPreRecoMap & output_digimap,
		CLHEP::HepRandomEngine* engine)
{

    /*
      Starting form the incoming pseudo-digi, which has perfect time and position resolution:
      1A. Smear time using sigma_t by some value
      1B. Correct the smeared time with the central arrival time for partition
      1C. Apply discretization: if the smeared time is outside the BX window (-12.5ns;+12.5ns),
      the hit should be assigned to the next (or previous) BX

      2A. Find strip that the digi belongs to
      2B. Get the center of this strip and the error on the position assuming the geometry

      3A. Filter event if a digi at this partition/strip/BX already exists
      3B. Add to collection
    */

	LogDebug("ME0ReDigiProducer::buildDigis") << "Begin building digis."<<std::endl;
	ME0DigiPreRecoCollection::DigiRangeIterator me0dgIt;
	for (me0dgIt = input_digis.begin(); me0dgIt != input_digis.end();
			++me0dgIt){

		const auto& me0Id = (*me0dgIt).first;
		LogTrace("ME0ReDigiProducer::buildDigis") << "Starting with roll: "<< me0Id<<std::endl;

		//setup map for this chamber/eta partition
		ChamberDigiMap chDigiMap;

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

			//smear TOF if necessary
			float tof = digi->tof() + (timeResolution < 0 ? 0.0 : CLHEP::RandGaussQ::shoot(engine, 0, timeResolution));

			//Values used to fill objet
			int mapPartIDX = me0Id.roll() -1;
			int strip = 0;
			LocalPoint  digiLocalPoint;
			LocalError  digiLocalError;
			if(useBuiltinGeo){
				getStripProperties(geometry->etaPartition(me0Id),&*digi,tof,strip,digiLocalPoint,digiLocalError);
			} else {
				mapPartIDX = getCustomStripProperties(me0Id,&*digi,tof,strip,digiLocalPoint,digiLocalError);

			}

			//filter if outside of readout window
			const int bxIdx = std::round(tof/bxWidth);
			LogTrace("ME0ReDigiProducer::buildDigis") << tof <<"("<<bxIdx<<") ";
			if(bxIdx < minBXReadout) {output_digimap.insertDigi(me0Id, -1); continue; }
			if(bxIdx > maxBXReadout) {output_digimap.insertDigi(me0Id, -1); continue; }
			tof = bxIdx*bxWidth;


			//If we are merging check to see if it already exists
			LogTrace("ME0ReDigiProducer::buildDigis") << "("<<bxIdx<<","<<mapPartIDX<<","<<strip<<") ";
			if(mergeDigis){
				int matchIDX = fillDigiMap(chDigiMap, bxIdx,mapPartIDX,strip,newDigiIdx);
				if(matchIDX >= 0){
					output_digimap.insertDigi(me0Id, matchIDX);
					continue;
				}
			}

			//Digis store sigmaX,sigmaY, correlationCoef
			const float sigmaX = std::sqrt(digiLocalError.xx());
			const float sigmaY = std::sqrt(digiLocalError.yy());
			const float corrCoef = digiLocalError.xy() /(sigmaX*sigmaY);

			//Fill in the new collection
			ME0DigiPreReco out_digi(digiLocalPoint.x(), digiLocalPoint.y(),
					sigmaX, sigmaY, corrCoef, tof, digi->pdgid(), digi->prompt());
			output_digis.insertDigi(me0Id, out_digi);

			// store index of previous detid and digi
			output_digimap.insertDigi(me0Id, newDigiIdx);
			newDigiIdx++;

			LogTrace("ME0ReDigiProducer::buildDigis") << "("<<digiLocalPoint.x()<<","<<digiLocalPoint.y()<<","<<sigmaX<<","<<sigmaY<<","<< tof<<") ";
		}

		chDigiMap.clear();


	}

}

void ME0ReDigiProducer::fillCentralTOFs() {
	const auto* mainChamber = geometry->chambers().front();
	const unsigned int nLayers = mainChamber->nLayers();
	//Get TOF at center of each partition
	tofs.clear();
	tofs.resize(nLayers);
	LogDebug("ME0ReDigiProducer::fillCentralTOFs()") << "TOF numbers [layer][partition]: " ;
	for(unsigned int iL = 0; iL < nLayers; ++iL){
		const auto* layer = mainChamber->layers()[iL];
		const unsigned int mapLayIDX = layer->id().layer() -1;
		const unsigned int nPartitions = layer->nEtaPartitions();
		if(!nPartitions)
			throw cms::Exception("Setup") << "ME0ReDigiProducer::fillCentralTOFs() - ME0Layer has no partitions.";
		tofs[mapLayIDX].resize(nPartitions);
		for(unsigned int iP = 0; iP < nPartitions; ++iP){
			const unsigned int mapPartIDX = layer->etaPartitions()[iP]->id().roll() -1;
			const GlobalPoint centralGP(layer->etaPartitions()[iP]->position());
			tofs[mapLayIDX][mapPartIDX] = (centralGP.mag() / (CLHEP::c_light/CLHEP::cm)); //speed of light [cm/ns]
			LogDebug("ME0ReDigiProducer::fillCentralTOFs()") << "["<<mapLayIDX<<"]["<<mapPartIDX<<"]="<< tofs[mapLayIDX][mapPartIDX] <<" "<<std::endl;
		}
	}
}
int ME0ReDigiProducer::getCustomStripProperties(const ME0DetId& detId,const ME0DigiPreReco* inDigi, float& tof,int& strip,  LocalPoint&  digiLocalPoint, LocalError&  digiLocalError ) const {
	const unsigned int partIdx = tempGeo->findEtaPartition(inDigi->y());
	LogTrace("ME0ReDigiProducer::buildDigis") << partIdx <<" ";
	const float partMeanTof = tempGeo->getCentralTOF(detId,partIdx);

	//convert to relative to partition
	tof -= partMeanTof;

	//get coordinates and errors
	const float partCenter = tempGeo->getPartCenter(partIdx);
	const auto* topo = tempGeo->getTopo(partIdx);

	//find channel
	const LocalPoint partLocalPoint(inDigi->x(), inDigi->y() - partCenter ,0.);
	strip = topo->channel(partLocalPoint);
	const float stripF = float(strip)+0.5;

	//get digitized location
	LocalPoint  digiPartLocalPoint = topo->localPosition(stripF);
	digiLocalError = topo->localError(stripF, 1./sqrt(12.)); //std dev. flat distribution with length L is L/sqrt(12). The strip topology expects the error in units of strips.
	digiLocalPoint = LocalPoint(digiPartLocalPoint.x(),digiPartLocalPoint.y() + partCenter,0.0);
	return partIdx;


}
void ME0ReDigiProducer::getStripProperties(const ME0EtaPartition* etaPart, const ME0DigiPreReco* inDigi, float& tof,int& strip, LocalPoint&  digiLocalPoint, LocalError&  digiLocalError) const {
	//convert to relative to partition
	tof -= tofs[etaPart->id().layer()-1][etaPart->id().roll() -1];

	const TrapezoidalStripTopology * origTopo = (const TrapezoidalStripTopology*)(&etaPart->specificTopology());
	TrapezoidalStripTopology padTopo(origTopo->nstrips()/2,origTopo->pitch()*2,origTopo->stripLength(),origTopo->radius());
	const auto & topo = usePads ?  padTopo : etaPart->specificTopology();

	//find channel
	const LocalPoint partLocalPoint(inDigi->x(), inDigi->y(),0.);
	strip = topo.channel(partLocalPoint);
	const float stripF = float(strip)+0.5;

	//get digitized location
	digiLocalPoint = topo.localPosition(stripF);
	digiLocalError = topo.localError(stripF, 1./sqrt(12.));
}

unsigned int ME0ReDigiProducer::fillDigiMap(ChamberDigiMap& chDigiMap, unsigned int bx, unsigned int part, unsigned int strip, unsigned int currentIDX) const {
	DigiIndicies newIDX(bx,part,strip);
	auto it1 = chDigiMap.find(newIDX);
	if (it1 == chDigiMap.end()){
		chDigiMap[newIDX] = currentIDX;
		return -1;
	}
	return it1->second;
}



// --------------------------------------------------------
// A class to read HPD noise from the library.
// The deliverable of the class is the collection of
// noisy HcalDetIds with associated noise in units of fC for
// 10 time samples. During the library production a higher
// theshold is used to find a noisy HPD. A lower threshold is
// used to eliminate adding unnecessary quite channels to HPD 
// noise event collection. Therefore user may not see whole 18 
// channels for noisy HPD.
//
// Project: HPD noise library reader
// Author: T.Yetkin University of Iowa, Feb. 7, 2008
// $Id: $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseLibraryReader.h"

using namespace edm;
using namespace std;

HPDNoiseLibraryReader::HPDNoiseLibraryReader(const edm::ParameterSet& iConfig)
: theNoiseRate(0), theNoisyPhi(0), theRandFlat(0), theRandGaussQ(0)
{
    setRandomEngine();
    
    ParameterSet pSet     = iConfig.getParameter<edm::ParameterSet>("HPDNoiseLibrary");
    FileInPath   filepath = pSet.getParameter<edm::FileInPath>("FileName");
    theHPDName            = pSet.getUntrackedParameter<string> ("HPDName", "HPD");
    string pName          = filepath.fullPath();
    if (pName.find(".") == 0) pName.erase(0,2);
    theReader = new HPDNoiseReader(pName);
    theNames = theReader->allNames();   // all 72x2 HPDs

    fillRate();
}

HPDNoiseLibraryReader::~HPDNoiseLibraryReader() 
{
    if (theRandFlat)delete theRandFlat;
    if (theRandGaussQ)delete theRandGaussQ;
}

void HPDNoiseLibraryReader::initializeServices() 
{
    if (not edmplugin::PluginManager::isAvailable()) {
        edmplugin::PluginManager::configure(edmplugin::standard::config());
    }

    std::string config =
      "process CorrNoise = {"
          "service = RandomNumberGeneratorService" 
             "{" 
	          "untracked uint32 sourceSeed = 123456789" 
	      "}" 
      "}";

    // create the services
    edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(config);

    // make the services available
    edm::ServiceRegistry::Operate operate(tempToken);
}
void HPDNoiseLibraryReader::setRandomEngine() 
{
    edm::Service < edm::RandomNumberGenerator > rng;
    if (!rng.isAvailable()) {
        throw cms::Exception("Configuration") << "HcalHPDNoiseLibrary requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
    }
    setRandomEngine(rng->getEngine());
}
void HPDNoiseLibraryReader::setRandomEngine(CLHEP::HepRandomEngine & engine) 
{
    if(theRandGaussQ) delete theRandGaussQ;
    if(theRandFlat) delete theRandFlat;
    theRandGaussQ = new CLHEP::RandGaussQ(engine);
    theRandFlat = new CLHEP::RandFlat(engine);
}

void HPDNoiseLibraryReader::fillRate() 
{
    for (size_t i = 0; i < theNames.size(); ++i) {
        HPDNoiseReader::Handle hpdObj = theReader->getHandle(theNames[i]);
        if (theReader->valid(hpdObj)) {
            theNoiseRate.push_back(theReader->rate(hpdObj));
        } else {
            std::cerr << "HPD Handle Object is not valid!" << endl;
        }
    }
}

HPDNoiseData* HPDNoiseLibraryReader::getNoiseData(int iphi) 
{
    
    
    HPDNoiseData* data;
    // make sure that iphi from HcalDetId is found noisy at first.
    // In other words, be sure that iphi is in the collection of
    // noisy Phis
    if (!(applyNoise(iphi))) return data;
    
    int zside = 1;
    if(iphi>72){
       iphi = iphi-72;
       zside = -1;
    }
    std::string name;
    if (zside == 1) {
        name = "ZPlus" + theHPDName + itos(iphi);
    } else if (zside == -1){
        name = "ZMinus" + theHPDName + itos(iphi);
    }else {
        cerr << " ZSide Calculation Error." << endl;
    }
    HPDNoiseReader::Handle hpdObj = theReader->getHandle(name);
    if (theReader->valid(hpdObj)) {
        // randomly select one entry from library for this HPD
        unsigned long entry = theRandFlat->fireInt( theReader->totalEntries(hpdObj));
	theReader->getEntry(hpdObj, entry, &data);
    }else{
        std::cerr << " HPD Name in the library is not valid." << std::endl;
    }
    return data;
}


void HPDNoiseLibraryReader::getNoisyPhis() 
{

    clearPhi();
    double rndm[144];

    theRandFlat->shootArray(144, rndm);

    for (int i = 0; i < 144; ++i) {
        if (rndm[i] < theNoiseRate[i]) {
            theNoisyPhi.push_back(i + 1);
        }
    }
}

void HPDNoiseLibraryReader::getBiasedNoisyPhis() 
{

    clearPhi();
    double rndm[144];

    theRandFlat->shootArray(144, rndm);
    for (int i = 0; i < 144; ++i) {
        if (rndm[i] < theNoiseRate[i]) {
            theNoisyPhi.push_back(i + 1);
        }
    }
    // make sure one HPD is always noisy
    if (theNoisyPhi.size() == 0) {
        int iPhi = (theRandFlat->fireInt(144)) + 1; // integer from interval [0-144[ + 1

        theNoisyPhi.push_back(iPhi);
    }
}

vector<pair <HcalDetId, const float* > > HPDNoiseLibraryReader::getNoisyHcalDetIds() 
{
    
    vector< pair<HcalDetId, const float* > > result;
    // decide which phi are noisy by using noise rates 
    // and random numbers (to be called for each event)
    getNoisyPhis();
    for (int i = 0; i < int(theNoisyPhi.size()); ++i) {
	int iphi = theNoisyPhi[i];
	HPDNoiseData* data;
	data = getNoiseData(iphi);
	for(unsigned int i=0; i<data->size();++i)
	{
	    pair < HcalDetId, const float* >tmp_pair( data->getDataFrame(i).id(), data->getDataFrame(i).getFrame());
	    result.push_back(tmp_pair);
	}
    }
    return result;
}

vector<pair <HcalDetId, const float* > > HPDNoiseLibraryReader::getBiasedNoisyHcalDetIds() 
{
    
    vector< pair<HcalDetId, const float* > > result;
    // decide which phi are noisy by using noise rates 
    // and random numbers (to be called for each event)
    // at least one Phi is always noisy.
    getBiasedNoisyPhis();
    for (int i = 0; i < int(theNoisyPhi.size()); ++i) {
	int iphi = theNoisyPhi[i];
	HPDNoiseData* data;
	data = getNoiseData(iphi);
	for(unsigned int i=0; i<data->size();++i)
	{
	    pair < HcalDetId, const float* >tmp_pair( data->getDataFrame(i).id(), data->getDataFrame(i).getFrame());
	    result.push_back(tmp_pair);
	}
    }
    return result;
}

bool HPDNoiseLibraryReader::applyNoise(int iphi) 
{

    bool isAccepted = false;
    vector < int >::iterator phi_iter;
    phi_iter = find(theNoisyPhi.begin(), theNoisyPhi.end(), iphi);
    if (phi_iter != theNoisyPhi.end()){
        isAccepted = true;
    }
    return isAccepted;
}
string HPDNoiseLibraryReader::itos(int i) 
{
    stringstream s;
    s << i;
    return s.str();
}

void HPDNoiseLibraryReader::clearPhi() {
    theNoisyPhi.clear();
}

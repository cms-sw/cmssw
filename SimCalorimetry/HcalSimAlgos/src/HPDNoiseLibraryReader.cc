
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
// $Id: HPDNoiseLibraryReader.cc,v 1.5 2012/06/07 18:12:43 wmtan Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseLibraryReader.h"

using namespace edm;
using namespace std;

HPDNoiseLibraryReader::HPDNoiseLibraryReader(const edm::ParameterSet & iConfig)
:theDischargeNoiseRate(0),
theIonFeedbackFirstPeakRate(0),
theIonFeedbackSecondPeakRate(0),
theNoisyPhi(0),
theRandFlat(0),
theRandGaussQ(0) {
    setRandomEngine();

    ParameterSet pSet = iConfig.getParameter < edm::ParameterSet > ("HPDNoiseLibrary");
    FileInPath filepath = pSet.getParameter < edm::FileInPath > ("FileName");

    theHPDName = pSet.getUntrackedParameter < string > ("HPDName", "HPD");
    string pName = filepath.fullPath();

    if (pName.find(".") == 0)
        pName.erase(0, 2);
    theReader = new HPDNoiseReader(pName);
    theNames = theReader->allNames();   // all 72x2 HPDs

    fillRates();
}

HPDNoiseLibraryReader::~HPDNoiseLibraryReader() {
    if (theRandFlat)
        delete theRandFlat;

    if (theRandGaussQ)
        delete theRandGaussQ;
}

void HPDNoiseLibraryReader::initializeServices() {
    if (not edmplugin::PluginManager::isAvailable()) {
        edmplugin::PluginManager::configure(edmplugin::standard::config());
    }

    std::string config =
      "process CorrNoise = {"
      "service = RandomNumberGeneratorService" "{" "untracked uint32 sourceSeed = 123456789" "}" "}";

    // create the services
    edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(config);

    // make the services available
    edm::ServiceRegistry::Operate operate(tempToken);
}
void HPDNoiseLibraryReader::setRandomEngine() {
    edm::Service < edm::RandomNumberGenerator > rng;
    if (!rng.isAvailable()) {
        throw cms::Exception("Configuration") << "HcalHPDNoiseLibrary requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
    }
    setRandomEngine(rng->getEngine());
}
void HPDNoiseLibraryReader::setRandomEngine(CLHEP::HepRandomEngine & engine) {
    if (theRandGaussQ)
        delete theRandGaussQ;

    if (theRandFlat)
        delete theRandFlat;

    theRandGaussQ = new CLHEP::RandGaussQ(engine);
    theRandFlat = new CLHEP::RandFlat(engine);
}

void HPDNoiseLibraryReader::fillRates() {
    for (size_t i = 0; i < theNames.size(); ++i) {
        HPDNoiseReader::Handle hpdObj = theReader->getHandle(theNames[i]);
        if (theReader->valid(hpdObj)) {
            theDischargeNoiseRate.push_back(theReader->dischargeRate(hpdObj));
            theIonFeedbackFirstPeakRate.push_back(theReader->ionFeedbackFirstPeakRate(hpdObj));
            theIonFeedbackSecondPeakRate.push_back(theReader->ionFeedbackSecondPeakRate(hpdObj));
        } else {
            std::cerr << "HPD Handle Object is not valid!" << endl;
        }
    }
}

HPDNoiseData *HPDNoiseLibraryReader::getNoiseData(int iphi) {


    HPDNoiseData *data = 0; //data->size() is checked wherever actually used  

    // make sure that iphi from HcalDetId is found noisy at first.
    // In other words, be sure that iphi is in the collection of
    // noisy Phis
    if (!(IsNoiseApplicable(iphi)))         
        return data;

    int zside = 1;

    if (iphi > 72) {
        iphi = iphi - 72;
        zside = -1;
    }
    std::string name;
    if (zside == 1) {
        name = "ZPlus" + theHPDName + itos(iphi);
    } else if (zside == -1) {
        name = "ZMinus" + theHPDName + itos(iphi);
    } else {
        cerr << " ZSide Calculation Error." << endl;
    }
    HPDNoiseReader::Handle hpdObj = theReader->getHandle(name);
    if (theReader->valid(hpdObj)) {
        // randomly select one entry from library for this HPD
        unsigned long entry = theRandFlat->fireInt(theReader->totalEntries(hpdObj));

        theReader->getEntry(hpdObj, entry, &data);
    } else {
        std::cerr << " HPD Name in the library is not valid." << std::endl;
    }
    return data;
}


void HPDNoiseLibraryReader::getNoisyPhis() {

    clearPhi();
    double rndm[144];

    theRandFlat->shootArray(144, rndm);

    for (int i = 0; i < 144; ++i) {
        if (rndm[i] < theDischargeNoiseRate[i]) {
            theNoisyPhi.push_back(i + 1);
        }
    }
}

void HPDNoiseLibraryReader::getBiasedNoisyPhis() {

    clearPhi();
    double rndm[144];

    theRandFlat->shootArray(144, rndm);
    for (int i = 0; i < 144; ++i) {
        if (rndm[i] < theDischargeNoiseRate[i]) {
            theNoisyPhi.push_back(i + 1);
        }
    }
    // make sure one HPD is always noisy
    if (theNoisyPhi.size() == 0) {
        int iPhi = (theRandFlat->fireInt(144)) + 1; // integer from interval [0-144[ + 1

        theNoisyPhi.push_back(iPhi);
    }
}

vector <pair<HcalDetId, const float *> >HPDNoiseLibraryReader::getNoisyHcalDetIds() {

    vector <pair< HcalDetId, const float *> >result;

    // decide which phi are noisy by using noise rates 
    // and random numbers (to be called for each event)
    getNoisyPhis();
    for (int i = 0; i < int (theNoisyPhi.size()); ++i) {
        int iphi = theNoisyPhi[i];
        HPDNoiseData *data;

        data = getNoiseData(iphi);
        for (unsigned int i = 0; i < data->size(); ++i) {

            result.emplace_back(data->getDataFrame(i).id(), data->getDataFrame(i).getFrame());
        }
    }
    return result;
}

vector <pair<HcalDetId, const float *> >HPDNoiseLibraryReader::getNoisyHcalDetIds(int timeSliceId) 
{
    vector <pair< HcalDetId, const float *> >result;
    // decide which phi are noisy by using noise rates 
    // and random numbers (to be called for each event)
    getNoisyPhis();
    for (int i = 0; i < int (theNoisyPhi.size()); ++i) {
        int iphi = theNoisyPhi[i];
        HPDNoiseData *data;

        data = getNoiseData(iphi);
        for (unsigned int i = 0; i < data->size(); ++i) {
	    float* data_ = const_cast<float*>(data->getDataFrame(i).getFrame());
	    shuffleData(timeSliceId, data_);
	    const float* _data_ =const_cast<const float*>(data_);
            result.emplace_back(data->getDataFrame(i).id(), _data_);
        }
    }
    return result;

}
vector < pair < HcalDetId, const float *> >HPDNoiseLibraryReader::getBiasedNoisyHcalDetIds(int timeSliceId) {

    vector < pair < HcalDetId, const float *> >result;

    // decide which phi are noisy by using noise rates 
    // and random numbers (to be called for each event)
    // at least one Phi is always noisy.
    getBiasedNoisyPhis();
    for (int i = 0; i < int (theNoisyPhi.size()); ++i) {
        int iphi = theNoisyPhi[i];
        HPDNoiseData *data;

        data = getNoiseData(iphi);
        for (unsigned int i = 0; i < data->size(); ++i) {
	    float* data_ = const_cast<float*>(data->getDataFrame(i).getFrame());
	    shuffleData(timeSliceId, data_);
	    const float* _data_ =const_cast<const float*>(data_);
            result.emplace_back(data->getDataFrame(i).id(), _data_);
        }
    }
    return result;
}

vector < pair < HcalDetId, const float *> >HPDNoiseLibraryReader::getBiasedNoisyHcalDetIds() {

    vector < pair < HcalDetId, const float *> >result;

    // decide which phi are noisy by using noise rates 
    // and random numbers (to be called for each event)
    // at least one Phi is always noisy.
    getBiasedNoisyPhis();
    for (int i = 0; i < int (theNoisyPhi.size()); ++i) {
        int iphi = theNoisyPhi[i];
        HPDNoiseData *data;

        data = getNoiseData(iphi);
        for (unsigned int i = 0; i < data->size(); ++i) {
            result.emplace_back(data->getDataFrame(i).id(), data->getDataFrame(i).getFrame());
        }
    }
    return result;
}

double HPDNoiseLibraryReader::getIonFeedbackNoise(HcalDetId id, double energy, double bias) {

    // constants for simulation/parameterization
    double pe2Charge = 0.333333;    // fC/p.e.
    double GeVperfC = 0.177;    // in unit GeV/fC and this will be updated when it start reading from DB.
    double PedSigma = 0.8;
    double noise = 0.;          // fC

    int iphi = (id.ieta() > 0) ? (id.iphi()) : (id.iphi() + 72);
    double rateInTail = theIonFeedbackFirstPeakRate[iphi - 1];
    double rateInSecondTail = theIonFeedbackSecondPeakRate[iphi - 1];

    if (bias != 0.) {
        rateInTail = rateInTail * bias;
        rateInSecondTail = rateInSecondTail * bias;
    } else {
        edm::LogError("HPDNoise") << "HPDNoise: ion feedback error (biased or unbiased selection)." << bias << " failed";
        throw cms::Exception("Unknown", "biase selection ")
        << "Usage of " << bias << " fails\n";
    }
    double Charge = energy / GeVperfC;

    // three gauss fit is applied to data to get ion feedback distribution
    // the script is at neutralino: /home/tyetkin/work/hpd_noise/PlotFromPelin.C
    // a new fit woth double-sigmoids is under way.
    // parameters (in fC)
    // first gaussian
    // double p0 = 9.53192e+05;
    // double p1 = -3.13653e-01;
    // double p2 = 2.78350e+00;

    // second gaussian
    // double p3 = 2.41611e+03;
    double p4 = 2.06117e+01;
    double p5 = 1.09239e+01;

    // third gaussian
    // double p6 = 3.42793e+01;
    double p7 = 5.45548e+01;
    double p8 = 1.59696e+01;

    if (Charge > 3 * PedSigma) {    // 3 sigma away from pedestal mean
        int npe = int (Charge / pe2Charge);
        double a = 0.;
        double b = 0.;

        for (int j = 0; j < npe; ++j) {
            double probability = theRandFlat->shoot();

            if (probability < rateInTail) { // total tail
                if (probability < rateInSecondTail) {   // second tail
                    Rannor(a, b);
                    noise += b * p8 + p7;
                } else {
                    Rannor(a, b);
                    noise += b * p5 + p4;
                }
            }
        }
        // add pedestal 
        if (noise > 0.)
            noise += theRandGaussQ->fire(0, 2 * PedSigma);
    }
    return (noise * GeVperfC);  // returns noise in GeV.

}

bool HPDNoiseLibraryReader::IsNoiseApplicable(int iphi) {

    bool isAccepted = false;
    vector < int >::iterator phi_iter;

    phi_iter = find(theNoisyPhi.begin(), theNoisyPhi.end(), iphi);
    if (phi_iter != theNoisyPhi.end()) {
        isAccepted = true;
    }
    return isAccepted;
}
void HPDNoiseLibraryReader::shuffleData(int timeSliceId, float* &data)
{
   if(timeSliceId == -1 || (timeSliceId>=10)) return;
   //make a local copy of input data
   float Data[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
   for(int i=0;i<10;++i){
       Data[i] = data[i];
   }
   int ts_max = -1;
   float max = -999.;
   for(int i=0;i<10;++i){
       if(Data[i]>max){
           max = data[i];
	   ts_max = i;
       }
   }
   if((ts_max == -1)){//couldn't find ts_max, return the same value.
       return;
   }else{
       // always shift the noise to the right by putting zeroes to the previous slices.
       // the noise is pedestal subtracted. 0 value is acceptable.
       int k = -1;
       for(int i=0;i<10;++i){
	   data[i] = 0.;
	   int newIdx = timeSliceId+k;
	   float dd = 0.;
	   if(newIdx < 10){
	       data[newIdx] = Data[ts_max+k];
	       dd = Data[ts_max+k];
	       i = newIdx;
	   }
	   data[i] = dd;
	   ++k;
       }
													   
   }
}

//I couldn't find Rannor in CLHEP/Random. For now, use it from ROOT (copy/paste) by little modification.
void HPDNoiseLibraryReader::Rannor(double &a, double &b) {
    double r,
      x,
      y,
      z;

    y = theRandFlat->shoot();
    z = theRandFlat->shoot();
    x = z * 6.28318530717958623;
    r = TMath::Sqrt(-2 * TMath::Log(y));
    a = r * TMath::Sin(x);
    b = r * TMath::Cos(x);
}
string HPDNoiseLibraryReader::itos(int i) {
    stringstream s;

    s << i;
    return s.str();
}

void HPDNoiseLibraryReader::clearPhi() {
    theNoisyPhi.clear();
}

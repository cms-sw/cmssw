
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseLibraryReader.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1F.h>

class HPDNoiseLibraryReaderTest : public edm::EDAnalyzer {
   public:
      explicit HPDNoiseLibraryReaderTest(const edm::ParameterSet&);
      ~HPDNoiseLibraryReaderTest();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      TFile* theFile;
      HPDNoiseLibraryReader* reader;
      bool UseBiasedNoise;
};

HPDNoiseLibraryReaderTest::HPDNoiseLibraryReaderTest(const edm::ParameterSet& iConfig)
{
    reader = new HPDNoiseLibraryReader(iConfig);
    UseBiasedNoise = iConfig.getUntrackedParameter<bool> ("UseBiasedHPDNoise", false);
    
}


HPDNoiseLibraryReaderTest::~HPDNoiseLibraryReaderTest()
{
    delete reader;
} 


void HPDNoiseLibraryReaderTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   vector<pair<HcalDetId, const float*> >NoisyHcalDetIds;
   if(UseBiasedNoise){
       NoisyHcalDetIds = reader->getBiasedNoisyHcalDetIds();
   }else{
       NoisyHcalDetIds = reader->getNoisyHcalDetIds();
   }
   //iterate over vector
   vector< pair<HcalDetId, const float* > >::const_iterator itNoise;
   for(itNoise=NoisyHcalDetIds.begin(); itNoise!=NoisyHcalDetIds.end();++itNoise){
       const float* noise = (*itNoise).second;
       cout << "DetId: " << (*itNoise).first << " noise (in units of fC) ";
       for(int ts=0;ts<10;++ts){
           cout << " time_sample=" << ts <<" noise= " << noise[ts] << " "; 
       }
    }
    cout << endl;

}


void HPDNoiseLibraryReaderTest::beginJob()
{
}

void HPDNoiseLibraryReaderTest::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HPDNoiseLibraryReaderTest);

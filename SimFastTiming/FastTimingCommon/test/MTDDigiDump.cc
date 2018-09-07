#include <memory>
#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"


class MTDDigiDump : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit MTDDigiDump(const edm::ParameterSet&);
      ~MTDDigiDump();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

  // ----------member data ---------------------------

      edm::EDGetTokenT<BTLDigiCollection> tok_BTL_digi; 
      edm::EDGetTokenT<ETLDigiCollection> tok_ETL_digi; 

};


MTDDigiDump::MTDDigiDump(const edm::ParameterSet& iConfig)

{

  tok_BTL_digi = consumes<BTLDigiCollection>(edm::InputTag("mix","FTLBarrel"));
  tok_ETL_digi = consumes<ETLDigiCollection>(edm::InputTag("mix","FTLEndcap"));

}


MTDDigiDump::~MTDDigiDump() {}


//
// member functions
//

// ------------ method called for each event ------------
void
MTDDigiDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;

  edm::Handle<BTLDigiCollection> h_BTL_digi;
  iEvent.getByToken( tok_BTL_digi, h_BTL_digi );

  edm::Handle<ETLDigiCollection> h_ETL_digi;
  iEvent.getByToken( tok_ETL_digi, h_ETL_digi );

 
  // --- BTL DIGIs:

  if ( h_BTL_digi->size() > 0 ) {

    std::cout << " ----------------------------------------" << std::endl;
    std::cout << " BTL DIGI collection:" << std::endl;
  
    for (const auto& dataFrame: *h_BTL_digi) {

      // --- detector element ID:
      std::cout << "   det ID:  det = " << dataFrame.id().det() 
		<< "  subdet = "  << dataFrame.id().mtdSubDetector() 
		<< "  side = "    << dataFrame.id().mtdSide() 
		<< "  rod = "     << dataFrame.id().mtdRR() 
		<< "  mod = "     << dataFrame.id().module() 
		<< "  type = "    << dataFrame.id().modType() 
		<< "  crystal = " << dataFrame.id().crystal() 
		<< std::endl;


      // --- loop over the dataFrame samples
      for (int isample = 0; isample<dataFrame.size(); ++isample){

	const auto& sample = dataFrame.sample(isample);

	std::cout << "       sample " << isample << ":"; 
	if ( sample.data()==0 && sample.toa()==0 ) {
	  std::cout << std::endl;
	  continue;
	}
	std::cout << "  amplitude = " << sample.data() 
		  << "  time1 = " <<  sample.toa() 
		  << "  time2 = " <<  sample.toa2() << std::endl;

      } // isaple loop

    } // digi loop

  } // if ( h_BTL_digi->size() > 0 )



  // --- ETL DIGIs:

  if ( h_ETL_digi->size() > 0 ) {

    std::cout << " ----------------------------------------" << std::endl;
    std::cout << " ETL DIGI collection:" << std::endl;
  
    for (const auto& dataFrame: *h_ETL_digi) {

      // --- detector element ID:
      std::cout << "   det ID:  det = " << dataFrame.id().det() 
		<< "  subdet = " << dataFrame.id().mtdSubDetector()
		<< "  side = "   << dataFrame.id().mtdSide() 
		<< "  ring = "   << dataFrame.id().mtdRR() 
		<< "  mod = "    << dataFrame.id().module() 
		<< "  type = "   << dataFrame.id().modType() 
		<< std::endl;

      
      // --- loop over the dataFrame samples
      for (int isample = 0; isample<dataFrame.size(); ++isample){

	const auto& sample = dataFrame.sample(isample);
	
	std::cout << "       sample " << isample << ":"; 
	if ( sample.data()==0 && sample.toa()==0 ) {
	  std::cout << std::endl;
	  continue;
	} 
	std::cout << "  amplitude = " << sample.data() << "  time = " <<  sample.toa() << std::endl;

      } // isample loop

    } // digi loop

  } // if ( h_ETL_digi->size() > 0 )
 
}


// ------------ method called once each job just before starting event loop  ------------
void 
MTDDigiDump::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MTDDigiDump::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MTDDigiDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDDigiDump);

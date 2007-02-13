/**_________________________________________________________________
   class:   BTagValidator.cc
   package: Validation/RecoB
   

 author: Victor Bazterra, UIC
         Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BTagValidator.cc,v 1.1 2007/02/11 03:32:53 yumiceva Exp $

________________________________________________________________**/

#include "Validation/RecoB/interface/BTagValidator.h"


// root include files
#include "TClass.h"
#include "TDirectory.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TString.h"

#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"

#include "RecoBTag/Analysis/interface/TrackCountingTagPlotter.h"
#include "RecoBTag/Analysis/interface/TrackProbabilityTagPlotter.h"


//
// constructors and destructor
//
BTagValidator::BTagValidator(const edm::ParameterSet& iConfig) :
  algorithm_( iConfig.getParameter<std::string>( "algorithm" ) ),
  rootFile_( iConfig.getParameter<std::string>( "rootfile" ) ),
  DQMFile_( iConfig.getParameter<std::string>( "DQMFile" ) ),
  histogramList_( iConfig.getParameter<vstring>( "histogramList" ) ) {
	
	// change assert to CMS catch exeptions
	//assert( !algorithm_.empty() ) ;   
	//assert( !rootFile_.empty() ) ;     
	//assert( !DQMFile_.empty() ) ;   
	//assert( !histogramList_.empty() ) ;   

	if (algorithm_ == "TrackCounting") 
		petBase_ = new BTagPABase<reco::TrackCountingTagInfoCollection, TrackCountingTagPlotter>(iConfig);
	else if (algorithm_ == "TrackProbability")
		petBase_ = new BTagPABase<reco::TrackCountingTagInfoCollection, TrackProbabilityTagPlotter>(iConfig);
	else {
		cout << "BTagPerformanceAnalyzer: Unknown algorithm "<< algorithm_ <<endl;
		cout << " Choose between JetTag, TrackCounting, TrackProbability\n";
		exit(1);
	}
	
}


BTagValidator::~BTagValidator() {
 
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
BTagValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  petBase_->analyze(iEvent, iSetup);
}


void 
BTagValidator::endJob() {

	
	petBase_->endJob();

	// Validation section
	TObject * tObject ;

	// DQM element
	MonitorElement * monitorElement ;

	TFile * file = new TFile ( TString ( rootFile_ ) ) ;
   
	// get hold of back-end interface
	DaqMonitorBEInterface * dbe = edm::Service<DaqMonitorBEInterface>().operator->();
	
	dbe->setCurrentFolder( algorithm_ );
   
	for (std::size_t i=0; i<histogramList_.size(); ++i) {
		
		tObject = gDirectory->Get( TString( histogramList_[i] ) ) ;

		if ( tObject->IsA()->InheritsFrom( "TH1" ) ) {
			
			TH1 * histogram = (TH1*) tObject ;  
			std::cout << "Histogram 1D " << i << ":" << std::endl ;
			std::cout << "  name : " << histogram->GetName() << std::endl ;
			std::cout << "  title: " << histogram->GetTitle() << std::endl ;
			std::cout << "  nbins: " << histogram->GetXaxis()->GetNbins() << std::endl ;
			std::cout << "  xmin : " << histogram->GetXaxis()->GetXmin() << std::endl ;
			std::cout << "  xmax : " << histogram->GetXaxis()->GetXmax() << std::endl ;
			
			monitorElement = dbe->book1D (
				std::string( histogram->GetName() ),
				std::string( histogram->GetTitle() ),
				histogram->GetXaxis()->GetNbins(),
				histogram->GetXaxis()->GetXmin(),
				histogram->GetXaxis()->GetXmax()
				);

			for(Int_t x=0; x<histogram->GetXaxis()->GetNbins(); x++) {
				monitorElement->setBinContent ( x, histogram->GetBinContent( x ) ) ; 
				monitorElement->setBinError ( x, histogram->GetBinError( x ) ) ;
			}  
		}
		else if ( tObject->IsA()->InheritsFrom( "TH2" ) ) {
			
			TH2 * histogram = (TH2*) tObject ;  
			std::cout << "Histogram 2D " << i << ":" << std::endl ;
			std::cout << "  name : " << histogram->GetName() << std::endl ;
			std::cout << "  title: " << histogram->GetTitle() << std::endl ;
			std::cout << "  nbins: " << histogram->GetXaxis()->GetNbins() << std::endl ;
			std::cout << "  xmin : " << histogram->GetXaxis()->GetXmin() << std::endl ;
			std::cout << "  xmax : " << histogram->GetXaxis()->GetXmax() << std::endl ;
			std::cout << "  ymin : " << histogram->GetYaxis()->GetXmin() << std::endl ;
			std::cout << "  ymax : " << histogram->GetYaxis()->GetXmax() << std::endl ;
               
			monitorElement = dbe->book2D (
				std::string( histogram->GetName() ),
				std::string( histogram->GetTitle() ),
				histogram->GetXaxis()->GetNbins(),
				histogram->GetXaxis()->GetXmin(),
				histogram->GetXaxis()->GetXmax(),
				histogram->GetYaxis()->GetNbins(),
				histogram->GetYaxis()->GetXmin(),
				histogram->GetYaxis()->GetXmax()	  
				);

			for(Int_t x=0; x<histogram->GetXaxis()->GetNbins(); x++)
				for(Int_t y=0; y<histogram->GetYaxis()->GetNbins(); y++) {
					monitorElement->setBinContent ( x, y, histogram->GetBinContent( x, y ) ) ;                 
					monitorElement->setBinError ( x, y, histogram->GetBinError( x, y ) ) ;                 
				}
		}
		else if ( tObject->IsA()->InheritsFrom( "TH3" ) ) {
			
			TH3 * histogram = (TH3*) tObject ;  
			std::cout << "Histogram 3D " << i << ":" << std::endl ;
			std::cout << "  name : " << histogram->GetName() << std::endl ;
			std::cout << "  title: " << histogram->GetTitle() << std::endl ;
			std::cout << "  nbins: " << histogram->GetXaxis()->GetNbins() << std::endl ;
			std::cout << "  xmin : " << histogram->GetXaxis()->GetXmin() << std::endl ;
			std::cout << "  xmax : " << histogram->GetXaxis()->GetXmax() << std::endl ;
			std::cout << "  ymin : " << histogram->GetYaxis()->GetXmin() << std::endl ;
			std::cout << "  ymax : " << histogram->GetYaxis()->GetXmax() << std::endl ;
			std::cout << "  zmin : " << histogram->GetZaxis()->GetXmin() << std::endl ;
			std::cout << "  zmax : " << histogram->GetZaxis()->GetXmax() << std::endl ;
			
			monitorElement = dbe->book3D (
				std::string( histogram->GetName() ),
				std::string( histogram->GetTitle() ),
				histogram->GetXaxis()->GetNbins(),
				histogram->GetXaxis()->GetXmin(),
				histogram->GetXaxis()->GetXmax(),
				histogram->GetYaxis()->GetNbins(),
				histogram->GetYaxis()->GetXmin(),
				histogram->GetYaxis()->GetXmax(),	  
				histogram->GetZaxis()->GetNbins(),
				histogram->GetZaxis()->GetXmin(),
				histogram->GetZaxis()->GetXmax()	  
				);

			for(Int_t x=0; x<histogram->GetXaxis()->GetNbins(); x++)
				for(Int_t y=0; y<histogram->GetYaxis()->GetNbins(); y++)
					for(Int_t z=0; z<histogram->GetZaxis()->GetNbins(); z++) {
						monitorElement->setBinContent ( x, y, z, histogram->GetBinContent( x, y, z ) ) ;                 
						monitorElement->setBinError ( x, y, z, histogram->GetBinError( x, y, z ) ) ;                 
					}
		}
	}

	dbe->showDirStructure() ;
	dbe->save(DQMFile_) ;  
	file->Close() ;
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagValidator);

/**_________________________________________________________________
   class:   BTagValidator.cc
   package: Validation/RecoB
   

 author: Victor Bazterra, UIC
         Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BTagValidator.cc,v 1.16 2010/07/20 02:58:37 wmtan Exp $

________________________________________________________________**/

#include "Validation/RecoB/interface/BTagValidator.h"

// root include files
#include "TClass.h"
#include "TDirectory.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TString.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "Validation/RecoB/interface/HistoCompare.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// constructors and destructor
//
BTagValidator::BTagValidator(const edm::ParameterSet& iConfig) {


	algorithm_ = iConfig.getParameter<std::string>( "algorithm" );
	rootFile_ = iConfig.getParameter<std::string>( "rootfile" );
	DQMFile_ = iConfig.getParameter<std::string>( "DQMFile" );
	TString tversion(edm::getReleaseVersion());
	tversion = tversion.Remove(0,1);
	tversion = tversion.Remove(tversion.Length()-1,tversion.Length());
	DQMFile_  = std::string(tversion)+"_"+DQMFile_;
	histogramList_ = iConfig.getParameter<vstring>( "histogramList" );
	referenceFilename_ = iConfig.getParameter<std::string>( "referenceFilename" );
	doCompare_ = iConfig.getParameter<bool>( "compareHistograms");
}


BTagValidator::~BTagValidator() {

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
BTagValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ }


void 
BTagValidator::endJob() 
{
	// Validation section
	TObject * tObject ;

	// DQM element
	MonitorElement* monElement ;
	MonitorElement* monElementRes;

	TFile *	file = new TFile ( TString ( rootFile_ ) ) ;

	// comparison
	HistoCompare hcompare;

	if (doCompare_) hcompare.SetReferenceFilename(TString(referenceFilename_) );
	
	file->cd();

	// get hold of back-end interface
	DQMStore * dbe = edm::Service<DQMStore>().operator->();
	
	dbe->setCurrentFolder( "RecoBV/"+algorithm_ );
   
	for (std::size_t i=0; i<histogramList_.size(); ++i) {
		
		tObject = gDirectory->Get( TString( histogramList_[i] ) ) ;

		if ( tObject == 0 ) 
			throw cms::Exception("BTagValidator") << "Histogram " << histogramList_[i] << " was not produced in the analysis. ";	
								
		if ( tObject->IsA()->InheritsFrom( "TH1" ) ) {
			
			TH1 * histogram = (TH1*) tObject ;  
			
			TH1* hresiduals = 0;
			
			if (doCompare_) 
				hresiduals = hcompare.Compare(histogram, "/DQMData/"+TString(algorithm_)+"/"+TString(histogram->GetName()) );
			
			file->cd();
			
			monElement = dbe->book1D (
				std::string( histogram->GetName() ),
				std::string( histogram->GetTitle() ),
				histogram->GetXaxis()->GetNbins(),
				histogram->GetXaxis()->GetXmin(),
				histogram->GetXaxis()->GetXmax()
				);

			monElement->setAxisTitle( std::string ( histogram->GetXaxis()->GetTitle()) , 1);
			monElement->setAxisTitle( std::string ( histogram->GetYaxis()->GetTitle()) , 2);


			for(Int_t x=0; x<histogram->GetXaxis()->GetNbins(); x++) {
			  monElement->setBinContent ( x, histogram->GetBinContent( x ) ) ;
			  monElement->setBinError ( x, histogram->GetBinError( x ) ) ;
                        }

			if (doCompare_ && hresiduals!= 0 ) {
			  monElementRes = dbe->book1D (
						       std::string( hresiduals->GetName() ),
						       std::string( hresiduals->GetTitle() ),
						       hresiduals->GetXaxis()->GetNbins(),
						       hresiduals->GetXaxis()->GetXmin(),
						       hresiduals->GetXaxis()->GetXmax()
						       );

			
			  for(Int_t x=0; x<hresiduals->GetXaxis()->GetNbins(); x++) {
			    monElementRes->setBinContent ( x, hresiduals->GetBinContent( x ) ) ; 
			    monElementRes->setBinError ( x, hresiduals->GetBinError( x ) ) ;
			  }  
			}
		}
		else if ( tObject->IsA()->InheritsFrom( "TH2" ) ) {
			
			TH2 * histogram = (TH2*) tObject ;  
               
			monElement = dbe->book2D (
				std::string( histogram->GetName() ),
				std::string( histogram->GetTitle() ),
				histogram->GetXaxis()->GetNbins(),
				histogram->GetXaxis()->GetXmin(),
				histogram->GetXaxis()->GetXmax(),
				histogram->GetYaxis()->GetNbins(),
				histogram->GetYaxis()->GetXmin(),
				histogram->GetYaxis()->GetXmax()	  
				);

			monElement->setAxisTitle( std::string ( histogram->GetXaxis()->GetTitle()) , 1);
			monElement->setAxisTitle( std::string ( histogram->GetYaxis()->GetTitle()) , 2);
			
			for(Int_t x=0; x<histogram->GetXaxis()->GetNbins(); x++)
				for(Int_t y=0; y<histogram->GetYaxis()->GetNbins(); y++) {
					monElement->setBinContent ( x, y, histogram->GetBinContent( x, y ) ) ;                 
					monElement->setBinError ( x, y, histogram->GetBinError( x, y ) ) ;                 
				}
		}
		else if ( tObject->IsA()->InheritsFrom( "TH3" ) ) {
			
			TH3 * histogram = (TH3*) tObject ;  
			
			monElement = dbe->book3D (
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

			monElement->setAxisTitle( std::string ( histogram->GetXaxis()->GetTitle()) , 1);
			monElement->setAxisTitle( std::string ( histogram->GetYaxis()->GetTitle()) , 2);
			
			for(Int_t x=0; x<histogram->GetXaxis()->GetNbins(); x++)
				for(Int_t y=0; y<histogram->GetYaxis()->GetNbins(); y++)
					for(Int_t z=0; z<histogram->GetZaxis()->GetNbins(); z++) {
						monElement->setBinContent ( x, y, z, histogram->GetBinContent( x, y, z ) ) ;                 
						monElement->setBinError ( x, y, z, histogram->GetBinError( x, y, z ) ) ;                 
					}
		}
	}

	dbe->showDirStructure() ;
	dbe->save(DQMFile_) ;  
	file->Close() ;
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagValidator);

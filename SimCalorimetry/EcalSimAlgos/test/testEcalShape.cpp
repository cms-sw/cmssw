#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>
#include<iomanip>

#include "TROOT.h"
#include "TStyle.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TF1.h"

int main() 
{
   edm::MessageDrop::instance()->debugEnabled = false;

   const EcalSimParameterMap parameterMap ;

   bool useDBShape = false; // for the purpose of testing the EcalShape class, doesn't need to (should not) fetch a shape from the DB but just to assume the testbeam one
   const APDShape theAPDShape(useDBShape) ;
   const EBShape theEBShape(useDBShape) ;
   const EEShape theEEShape(useDBShape) ;

   const int nsamp = 500; // phase I hardcoded ECAL shapes arrays
   const int tconv = 10;  // kNBinsPerNSec = 10

   const unsigned int histsiz = nsamp*tconv;

   for( unsigned int i ( 0 ) ; i != 3 ; ++i )
   {

      const DetId id ( 0 == i || 2 == i ?
		       (DetId) EBDetId(1,1) :
		       (DetId) EEDetId(1,50,1) ) ;

      const EcalShapeBase* theShape ( 0 == i ? 
				      (EcalShapeBase*) &theEBShape : 
				      ( 1 == i ? 
					(EcalShapeBase*) &theEEShape :
					(EcalShapeBase*) &theAPDShape  ) ) ;
      
      const double ToM ( theShape->timeOfMax()  ) ;
      const double T0  ( theShape->timeOfThr()  ) ;
      const double rT  ( theShape->timeToRise() ) ;


      // standard display of the implemented shape function
      const int csize = 500;
      TCanvas * showShape = new TCanvas("showShape","showShape",2*csize,csize);
/*

//  const std::vector<double>& nt = theShape.getTimeTable();
//  const std::vector<double>& ntd = theShape.getDerivTable();

  TH1F* shape1 = new TH1F("shape1","Tabulated Ecal MGPA shape",histsiz,0.,(float)(histsiz));
  TH1F* deriv1 = new TH1F("deriv1","Tabulated Ecal MGPA derivative",histsiz,0.,(float)(histsiz));


  
  std::cout << "interpolated ECAL pulse shape and its derivative \n" << std::endl;
  for ( unsigned int i = 0; i < histsiz; ++i ) 
  {
     const double time ( (i-0.5)*0.1 - T0 ) ;
     const double myShape ( theShape( time ) ) ;
     const double myDeriv ( theShape.derivative( time ) ) ;
     shape1->Fill((float)(i+0.5),(float)myShape );
     deriv1->Fill((float)(i+0.5),(float)myDeriv );
     std::cout << " time (ns) = " << std::fixed << std::setw(6) << std::setprecision(2) << time + T0 + 0.1
	       << " shape = " << std::setw(11) << std::setprecision(8) << myShape 
	       << " derivative = " << std::setw(11) << std::setprecision(8) << myDeriv << std::endl;
  }

  showShape->Divide(2,1);
  showShape->cd(1);
  shape1->Draw();
  showShape->cd(2);
  deriv1->Draw();
  showShape->SaveAs("EcalShape.jpg");
  showShape->Clear("");

  delete shape1;
  delete deriv1;
*/

      const std::string name ( 0 == i ? "Barrel" : 
			       ( 1 == i ? "Endcap" :
				 "APD" ) ) ;

      std::cout << "\n ********************* "
		<< name 
		<< "************************" ; 

      std::cout << "\n Maximum time from tabulated values = " 
		<< std::fixed    << std::setw(6)   
		<< std::setprecision(2) << ToM << std::endl ;

      std::cout << "\n Tzero from tabulated values        = " 
		<< std::fixed    << std::setw(6)   
		<< std::setprecision(2) << T0 << std::endl ;

      std::cout << "\n Rising time from tabulated values  = " 
		<< std::fixed    << std::setw(6)   
		<< std::setprecision(2) << rT << std::endl;

      // signal used with the nominal parameters and no jitter

      std::cout << "\n computed ECAL " << name 
		<< " pulse shape and its derivative (LHC timePhaseShift = 1) \n" << std::endl;
      const double tzero = rT - ( parameterMap.simParameters( id ).binOfMaximum() - 1. )*25. ;
      double x = tzero ;

      const std::string title1 ( "Computed Ecal " + name + " MGPA shape" ) ;
      const std::string title2 ( "Computed Ecal " + name + " MGPA derivative" ) ;

      TH1F* shape2 = new TH1F( "shape2", title1.c_str(), nsamp, 0., (float) nsamp ) ;
      TH1F* deriv2 = new TH1F( "deriv2", title2.c_str(), nsamp, 0., (float) nsamp ) ;
      double y = 0.;
      double dy = 0.;

      for( unsigned int i ( 0 ) ; i != histsiz ; ++i ) 
      {
	 y  = (*theShape)(x);
	 dy = theShape->derivative(x);
	 shape2->Fill((float)(x-tzero),(float)y);
	 deriv2->Fill((float)(x-tzero),(float)dy);
	 std::cout << " time (ns) = "  << std::fixed    << std::setw(6)         << std::setprecision(2) << x-tzero 
		   << " shape = "      << std::setw(11) << std::setprecision(5) << y
		   << " derivative = " << std::setw(11) << std::setprecision(5) << dy << std::endl;
	 x = x+1./(double)tconv;
      }

      for( unsigned int iSample ( 0 ) ; iSample != 10 ; ++iSample ) 
      {
	 std::cout << (*theShape)(tzero + iSample*25.0) << std::endl; 
      }

      showShape->Divide(2,1);
      showShape->cd(1);
      gPad->SetGrid();
      shape2->GetXaxis()->SetNdivisions(10,kFALSE);
      shape2->Draw();

      showShape->cd(2);
      gPad->SetGrid();
      deriv2->GetXaxis()->SetNdivisions(10,kFALSE);
      deriv2->Draw();

      const std::string fname ( name + "EcalShapeUsed.jpg" ) ;
      showShape->SaveAs( fname.c_str() );

      delete shape2;
      delete deriv2;
      delete showShape;

   }

   return 0;
} 

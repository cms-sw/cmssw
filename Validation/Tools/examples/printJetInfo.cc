// -*- C++ -*-

// CMS includes
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "PhysicsTools/FWLite/interface/EventContainer.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h" 

// Root includes
#include "TString.h"
#include "TROOT.h"

using namespace std;

///////////////////////////
// ///////////////////// //
// // Main Subroutine // //
// ///////////////////// //
///////////////////////////

int main (int argc, char* argv[]) 
{
   ////////////////////////////////
   // ////////////////////////// //
   // // Command Line Options // //
   // ////////////////////////// //
   ////////////////////////////////


   // Tell people what this analysis code does and setup default options.
   optutl::CommandLineParser parser ("");

   ////////////////////////////////////////////////
   // Change any defaults or add any new command //
   //      line options you would like here.     //
   ////////////////////////////////////////////////

   // Parse the command line arguments
   parser.parseArguments (argc, argv);

   //////////////////////////////////
   // //////////////////////////// //
   // // Create Event Container // //
   // //////////////////////////// //
   //////////////////////////////////

   // This object 'event' is used both to get all information from the
   // event as well as to store histograms, etc.
   fwlite::EventContainer eventCont (parser);

   ////////////////////////////////////////
   // ////////////////////////////////// //
   // //         Begin Run            // //
   // // (e.g., book histograms, etc) // //
   // ////////////////////////////////// //
   ////////////////////////////////////////

   // Setup a style
   gROOT->SetStyle ("Plain");

   // Book those histograms!

   //////////////////////
   // //////////////// //
   // // Event Loop // //
   // //////////////// //
   //////////////////////

   edm::Handle< std::vector< reco::CaloJet> > jetHandle;
   edm::InputTag jetLabel ("sisCone5CaloJets");

   for (eventCont.toBegin(); ! eventCont.atEnd(); ++eventCont) 
   {

      cout << "run " << eventCont.eventAuxiliary().run() << " event " 
           << eventCont.eventAuxiliary().event() << endl;
      cout << " index     Et         eta      phi" << endl;


      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////
      eventCont.getByLabel (jetLabel, jetHandle);
      assert (jetHandle.isValid());
      const std::vector< reco::CaloJet > &jetVec( *jetHandle.product() );
      int index = 0;
      for (std::vector< reco::CaloJet >::const_iterator iter = jetVec.begin(); 
           jetVec.end() != iter;
           ++iter, ++index)
      {
         cout << "   " << setw(2) << index << ") ";
         cout << setw(8) << Form ("%8.4f", iter->et() ) << "  " 
              << setw(8) << Form ("%8.4f", iter->eta()) << "  " 
              << setw(8) << Form ("%8.4f", iter->phi()) << "  " << endl;              
      }
      
   } // for eventCont

      
   ////////////////////////
   // ////////////////// //
   // // Clean Up Job // //
   // ////////////////// //
   ////////////////////////

   // Histograms will be automatically written to the root file
   // specificed by command line options.

   // All done!  Bye bye.
   return 0;
}

// -*- C++ -*-
//
// Package:    TtDecaySelection
// Class:      TtDecaySelection
// 
/**\class TtDecaySelection TtDecaySelection.cc AnalysisDataFormats/TopObjects/src/TtDecaySelection.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr 10 18:17:49 CEST 2007
// $Id: TtDecaySelection.h,v 1.1 2007/05/02 15:10:44 lowette Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include<string>
#include<vector>

using namespace std;


//
// class declaration
//


class TtDecaySelection : public edm::EDFilter {
   public:
      explicit TtDecaySelection(const edm::ParameterSet&);
      ~TtDecaySelection();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      
      
   private:
      int decay_;	        // decay exists of two digis number,
				// first: #nr of leptons in decay
				// second = 0 if you allow all leptons
				//	    1 if you allow only electrons
                                //          2 if you allow only muons
                                //          3 if you allow only taus
                                //          4 if you allow no electrons
				//	    5 if you allow no muons
                                //          6 if you allow no taus
				// decay = -1 means no selection at all


};

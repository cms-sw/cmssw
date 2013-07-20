// -*- C++ -*-
//
// Package:     HelpfulWatchers
// Class  :     BeginOfTrackCounter
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Tue Nov 29 12:26:42 EST 2005
// $Id: BeginOfTrackCounter.cc,v 1.1 2005/11/29 18:42:56 chrjones Exp $
//

// system include files

// user include files
#include "SimG4Core/HelpfulWatchers/src/BeginOfTrackCounter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// constants, enums and typedefs
//
using namespace simwatcher;
//
// static data member definitions
//

//
// constructors and destructor
//
BeginOfTrackCounter::BeginOfTrackCounter(const edm::ParameterSet& iPSet) :
   m_count(0),
   m_label(iPSet.getUntrackedParameter<std::string>("instanceLabel","nBeginOfTracks"))
{
   produces<int>(m_label);
}


//
// member functions
//

void
BeginOfTrackCounter::produce(edm::Event& e, const edm::EventSetup&)
{
   std::auto_ptr<int> product(new int(m_count));
   e.put(product,m_label);
   m_count = 0;
}

void
BeginOfTrackCounter::update(const BeginOfTrack*){
   ++m_count;
}

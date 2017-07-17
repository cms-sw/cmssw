#include "SimG4Core/CustomPhysics/interface/RHStopDump.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

RHStopDump::RHStopDump(edm::ParameterSet const & parameters) 
  : mStream (parameters.getParameter<std::string>("stoppedFile").c_str()),
    mProducer (parameters.getUntrackedParameter<std::string>("producer", "g4SimHits"))
{}

 void RHStopDump::analyze(const edm::Event& fEvent, const edm::EventSetup&) {
   edm::Handle<std::vector<std::string> > names;
   fEvent.getByLabel (mProducer, "StoppedParticlesName", names);
   edm::Handle<std::vector<float> > xs;
   fEvent.getByLabel (mProducer, "StoppedParticlesX", xs);
   edm::Handle<std::vector<float> > ys;
   fEvent.getByLabel (mProducer, "StoppedParticlesY", ys);
   edm::Handle<std::vector<float> > zs;
   fEvent.getByLabel (mProducer, "StoppedParticlesZ", zs);
   edm::Handle<std::vector<float> > ts;
   fEvent.getByLabel (mProducer, "StoppedParticlesTime", ts);
   edm::Handle<std::vector<int> > ids;
   fEvent.getByLabel (mProducer, "StoppedParticlesPdgId", ids);
   edm::Handle<std::vector<float> > masses;
   fEvent.getByLabel (mProducer, "StoppedParticlesMass", masses);
   edm::Handle<std::vector<float> > charges;
   fEvent.getByLabel (mProducer, "StoppedParticlesCharge", charges);

   if (names->size() != xs->size() || xs->size() != ys->size() || ys->size() != zs->size()) {
     edm::LogError ("RHStopDump") << "mismatch array sizes name/x/y/z:"
				  << names->size() << '/' << xs->size() << '/' << ys->size() << '/' << zs->size()
				  << std::endl;
   }
   else {
     for (size_t i = 0; i < names->size(); ++i) {
       mStream << (*names)[i] << ' ' << (*xs)[i] << ' ' << (*ys)[i] << ' ' << (*zs)[i] << ' ' << (*ts)[i] << std::endl;
       mStream << (*ids)[i] << ' ' << (*masses)[i] << ' ' << (*charges)[i] << std::endl;
     }
   }
 }

#include "FWCore/Utilities/interface/EDMException.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"

int PdtEntry::pdgId() const {
  if (pdgId_ == 0)
    throw cms::Exception("ConfigError") << "PdtEntry::pdgId was not set.\n"
                                        << "please, call PdtEntry::setup(const EventSetup & es)";
  return pdgId_;
}

std::string const &PdtEntry::name() const {
  if (name_.empty())
    throw cms::Exception("ConfigError") << "PdtEntry::name was not set."
                                        << "please, call PdtEntry::setup(const EventSetup & es)";
  return name_;
}

HepPDT::ParticleData const &PdtEntry::data() const {
  if (data_ == nullptr)
    throw cms::Exception("ConfigError") << "PdtEntry::name was not set."
                                        << "please, call PdtEntry::setup(const EventSetup & es)";
  return *data_;
}

void PdtEntry::setup(HepPDT::ParticleDataTable const &pdt) {
  HepPDT::ParticleData const *p = nullptr;
  if (pdgId_ == 0) {
    p = pdt.particle(name_);
    if (p == nullptr)
      throw cms::Exception("ConfigError") << "PDT has no entry for " << name_ << "."
                                          << "PdtEntry can't be set.";
    pdgId_ = p->pid();
  } else {
    p = pdt.particle(pdgId_);
    if (p == nullptr)
      throw cms::Exception("ConfigError") << "PDT has no entry for " << pdgId_ << "."
                                          << "PdtEntry can't be set.";
    name_ = p->name();
  }
  data_ = p;
}
namespace edm {
  namespace pdtentry {
    PdtEntry getPdtEntry(Entry const &e, char const *name) {
      if (e.typeCode() == 'I')
        return PdtEntry(e.getInt32());
      else if (e.typeCode() == 'S')
        return PdtEntry(e.getString());
      else if (e.typeCode() == 'Z')
        return PdtEntry(e.getString());
      else
        throw Exception(errors::Configuration, "EntryError")
            << "can not convert representation of " << name << " to value of type PdtEntry. "
            << "Please, provide a parameter either of type int32 or string.";
    }

    std::vector<PdtEntry> getPdtEntryVector(Entry const &e, char const *name) {
      std::vector<PdtEntry> ret;
      if (e.typeCode() == 'i') {
        std::vector<int> v(e.getVInt32());
        for (std::vector<int>::const_iterator i = v.begin(); i != v.end(); ++i)
          ret.push_back(PdtEntry(*i));
        return ret;
      } else if (e.typeCode() == 's') {
        std::vector<std::string> v(e.getVString());
        for (std::vector<std::string>::const_iterator i = v.begin(); i != v.end(); ++i)
          ret.push_back(PdtEntry(*i));
        return ret;
      } else if (e.typeCode() == 'z') {
        std::vector<std::string> v(e.getVString());
        for (std::vector<std::string>::const_iterator i = v.begin(); i != v.end(); ++i)
          ret.push_back(PdtEntry(*i));
        return ret;
      } else
        throw Exception(errors::Configuration, "EntryError")
            << "can not convert representation of " << name << " to value of type PdtEntry. "
            << "Please, provide a parameter either of type int32 or string.";
    }
  }  // namespace pdtentry
}  // namespace edm

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/types.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

static
bool dddGetStringRaw(const DDFilteredView & view, const std::string & name, std::string & value) {
  std::vector<const DDsvalues_type *> result;
  view.specificsV(result);
  for (std::vector<const DDsvalues_type *>::iterator it = result.begin(); it != result.end(); ++it)   {
    DDValue parameter(name);
    if (DDfetch(*it, parameter)) {
      if (parameter.strings().size() == 1) {
        value = parameter.strings().front();
        return true;
      } else {
        throw cms::Exception("Configuration")<< " ERROR: multiple " << name << " tags encountered";
        return false;
      }
    }
  }
  return false;
}

static inline
double dddGetDouble(const std::string & s, const DDFilteredView & view) {
  std::string value;
  if (dddGetStringRaw(view, s, value))
    return double(::atof(value.c_str()));
  else
    return NAN;
}

static inline
std::string dddGetString(const std::string & s, const DDFilteredView & view) {
  std::string value;
  if (dddGetStringRaw(view, s, value))
    return value;
  else
    return std::string();
}

static inline
std::ostream & operator<<(std::ostream & out, const math::XYZVector & v) {
  out << std::fixed << std::setprecision(3);
  return out << "(" << v.rho() << ", " << v.z() << ", " << v.phi() << ")";
}

class ListGroups : public edm::EDAnalyzer
{
public:
  ListGroups(const edm::ParameterSet &);
  virtual ~ListGroups();

private:
  void analyze(const edm::Event &, const edm::EventSetup &);

  edm::ESWatcher<IdealGeometryRecord> m_geometryWatcher;
};

ListGroups::ListGroups(const edm::ParameterSet &) {
}

ListGroups::~ListGroups() {
}

void
ListGroups::analyze(const edm::Event &, const edm::EventSetup & setup) {
  if (m_geometryWatcher.check(setup)) {
    edm::ESTransientHandle<DDCompactView> cv;
    setup.get<IdealGeometryRecord>().get(cv);
    DDFilteredView fv(*cv);

    DDSpecificsFilter filter;
    filter.setCriteria(DDValue("TrackingMaterialGroup", ""), DDSpecificsFilter::not_equals);
    fv.addFilter(filter);

    while (fv.next()) {
      // print the group name and full hierarchy of all items
      std::cout << dddGetString("TrackingMaterialGroup", fv) << '\t';

      // start from 2 to skip the leading /OCMS[0]/CMSE[1] part
      const DDGeoHistory & history = fv.geoHistory();
      std::cout << '/';
      for (unsigned int h = 2; h < history.size(); ++h)
        std::cout << '/' << history[h].logicalPart().name().name() << '[' << history[h].copyno() << ']';

      // DD3Vector and DDTranslation are the same type as math::XYZVector
      math::XYZVector position = fv.translation() / 10.;  // mm -> cm
      std::cout << "\t" << position << std::endl;
    };
    std::cout << std::endl;
  }
}

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ListGroups);

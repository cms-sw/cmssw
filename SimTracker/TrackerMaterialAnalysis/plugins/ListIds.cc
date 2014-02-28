#include <string>
#include <vector>
#include <iostream>

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
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

static
bool dddGetStringRaw(const DDFilteredView & view, const std::string & name, std::string & value) {
  DDValue parameter(name);
  std::vector<const DDsvalues_type *> result;
  view.specificsV(result);
  for (std::vector<const DDsvalues_type *>::iterator it = result.begin(); it != result.end(); ++it) {
    if (DDfetch(*it,parameter)) {
      if (parameter.strings().size() == 1) {
        value = parameter.strings().front();
        return true;
      } else {
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
  return out << "(" << v.rho() << ", " << v.z() << ", " << v.phi() << ")";
}

class ListIds : public edm::EDAnalyzer
{
public:
  ListIds(const edm::ParameterSet &);
  virtual ~ListIds();

private:
  void analyze(const edm::Event &, const edm::EventSetup &);

  edm::ESWatcher<IdealGeometryRecord>       m_idealGeometryWatcher;
  edm::ESWatcher<TrackerDigiGeometryRecord> m_trackerGeometryWatcher;
};

ListIds::ListIds(const edm::ParameterSet &) {
}

ListIds::~ListIds() {
}

void
ListIds::analyze(const edm::Event &, const edm::EventSetup & setup) {
  if (m_idealGeometryWatcher.check(setup)) {
    std::cout << "______________________________ DDD ______________________________" << std::endl;
    edm::ESTransientHandle<DDCompactView> hDdd;
    setup.get<IdealGeometryRecord>().get(hDdd);

    std::string attribute = "TkDDDStructure";
    CmsTrackerStringToEnum theCmsTrackerStringToEnum;
    DDSpecificsFilter filter;
    filter.setCriteria(DDValue(attribute, "any", 0), DDSpecificsFilter::not_equals);
    DDFilteredView fv(*hDdd);
    fv.addFilter(filter);
    if (theCmsTrackerStringToEnum.type(dddGetString(attribute, fv)) != GeometricDet::Tracker) {
      fv.firstChild();
      if (theCmsTrackerStringToEnum.type(dddGetString(attribute, fv)) != GeometricDet::Tracker)
        throw cms::Exception("Configuration") << "The first child of the DDFilteredView is not what is expected \n"
                                              << dddGetString(attribute, fv);
    }

    std::cout << std::fixed << std::setprecision(3);
    do {
      // print the full hierarchy of all Silicon items
      if (fv.logicalPart().material().name() == "materials:Silicon") {

        // start from 2 to skip the leading /OCMS[0]/CMSE[1] part
        const DDGeoHistory & history = fv.geoHistory();
        std::cout << '/';
        for (unsigned int h = 2; h < history.size(); ++h) {
          std::cout << '/' << history[h].logicalPart().name().name() << '[' << history[h].copyno() << ']';
        }

        // DD3Vector and DDTranslation are the same type as math::XYZVector
        math::XYZVector position = fv.translation() / 10.;  // mm -> cm
        std::cout << "\t" << position << std::endl;
      }
    } while (fv.next());
    std::cout << std::endl;
  }

  if (m_trackerGeometryWatcher.check(setup)) {
    std::cout << "______________________________ std::vector<GeomDet*> from TrackerGeometry::dets() ______________________________" << std::endl;
    edm::ESTransientHandle<TrackerGeometry> hGeo;
    setup.get<TrackerDigiGeometryRecord>().get( hGeo );

    std::cout << std::fixed << std::setprecision(3);
    const std::vector<GeomDet*> & dets = hGeo->dets();
    for (unsigned int i = 0; i < dets.size(); ++i) {
      const GeomDet & det = *dets[i];

      // Surface::PositionType is a typedef for Point3DBase<float,GlobalTag> a.k.a. GlobalPoint
      const Surface::PositionType & p = det.position();
      math::XYZVector position(p.x(), p.y(), p.z());

      std::cout << det.subDetector() << '\t'
                << det.geographicalId().det() << '\t'
                << det.geographicalId().subdetId() << '\t'
                << det.geographicalId().rawId() << "\t"
                << position;
      const std::vector<const GeomDet*> & parts = det.components();
      if (parts.size()) {
        std::cout << "\t[" << parts[0]->geographicalId().rawId();
        for (unsigned int j = 1; j < parts.size(); ++j)
          std::cout << '\t' << parts[j]->geographicalId().rawId();
        std::cout << ']';
      }
      std::cout << std::endl;
    }
  }

}

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ListIds);

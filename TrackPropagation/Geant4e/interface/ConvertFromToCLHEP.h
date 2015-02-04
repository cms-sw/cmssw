#ifndef TrackPropagation_ConvertFromToCLHEP_h
#define TrackPropagation_ConvertFromToCLHEP_h

//CLHEP
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Normal3D.h" 
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//CMS
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

//Geant4
#include "G4ErrorFreeTrajState.hh"

/** Utilities to convert among CLHEP and CMS points and vectors
 */

namespace TrackPropagation
{
/**
 Convert a CMS GlobalPoint to a CLHEP HepGeom::Point3D<double>
 CMS uses cm while Geant4 uses mm. This is taken into account in the
 conversion.
 */

inline HepGeom::Point3D<double> globalPointToHepPoint3D(const GlobalPoint& r)
{
	return HepGeom::Point3D<double>(r.x() * cm, r.y() * cm, r.z() * cm);
}

/** Convert a CLHEP HepGeom::Point3D<double>  to a CMS GlobalPoint
 CMS uses cms while Geant4 uses mm. This is taken into account in the
 conversion.
 */
inline GlobalPoint hepPoint3DToGlobalPoint(const HepGeom::Point3D<double> & r)
{
	return GlobalPoint(r.x() / cm, r.y() / cm, r.z() / cm);
}

/** Convert a G4double representing a scaler measure ( e.g. Track length ) to the
 CMS convention, which is using mm
 */
inline double g4doubleToCmsDouble(const G4double & d)
{
	return d / cm;
}

/** Convert a CMS GlobalVector to a CLHEP HepGeom::Normal3D<double>
 CMS uses GeV while G4 uses MeV
 */
inline HepGeom::Normal3D<double> globalVectorToHepNormal3D(
		const GlobalVector& p)
{
	return HepGeom::Normal3D<double>(p.x(), p.y(), p.z());
}

/** Convert a CLHEP HepGeom::Normal3D<double>  to a CMS GlobalVector
 CMS uses GeV while G4 uses MeV
 */
inline GlobalVector hepNormal3DToGlobalVector(
		const HepGeom::Normal3D<double> & p)
{
	return GlobalVector(p.x(), p.y(), p.z());
}

/** Convert a CMS GlobalVector to a CLHEP CLHEP::Hep3Vector
 */
inline CLHEP::Hep3Vector globalVectorToHep3Vector(const GlobalVector& p)
{
	return CLHEP::Hep3Vector(p.x(), p.y(), p.z());
}

/** Convert a CLHEP CLHEP::Hep3Vector to a CMS GlobalVector
 */
inline GlobalVector hep3VectorToGlobalVector(const CLHEP::Hep3Vector& p)
{
	return GlobalVector(p.x(), p.y(), p.z());
}

/** Convert a CMS GlobalPoint to a CLHEP CLHEP::Hep3Vector
 CMS uses cm while Geant4 uses mm. This is taken into account in the
 conversion.
 */
inline CLHEP::Hep3Vector globalPointToHep3Vector(const GlobalPoint& r)
{
	return CLHEP::Hep3Vector(r.x() * cm, r.y() * cm, r.z() * cm);
}

/** Convert a CLHEP CLHEP::Hep3Vector to a CMS GlobalPoint
 CMS uses cm while Geant4 uses mm. This is taken into account in the
 conversion.
 */
inline GlobalPoint hep3VectorToGlobalPoint(const CLHEP::Hep3Vector& v)
{
	return GlobalPoint(v.x() / cm, v.y() / cm, v.z() / cm);
}

/** Convert a CMS TkRotation<float> to a CLHEP CLHEP::HepRotation=G4RotationMatrix
 */
inline CLHEP::HepRotation tkRotationFToHepRotation(const TkRotation<float>& tkr)
{
	return CLHEP::HepRotation(CLHEP::Hep3Vector(tkr.xx(), tkr.yx(), tkr.zx()),
			CLHEP::Hep3Vector(tkr.xy(), tkr.yy(), tkr.zy()),
			CLHEP::Hep3Vector(tkr.xz(), tkr.yz(), tkr.zz()));
}

/** Convert a CLHEP CLHEP::Hep3Vector to a CMS GlobalPoint
 */
inline TkRotation<float> hepRotationToTkRotationF(const CLHEP::HepRotation& r)
{
	return TkRotation<float>(r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(),
			r.zx(), r.zy(), r.zz());
}

/** Convert a G4 Trajectory Error Matrix to the CMS Algebraic Sym Matrix
 CMS uses q/p as first parameter, G4 uses 1/p
 */

inline AlgebraicSymMatrix55 g4ErrorTrajErrToAlgebraicSymMatrix55(
		const G4ErrorTrajErr& e, const int q)
{
	assert(q != 0);
	//From DataFormats/CLHEP/interface/Migration.h
	//typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> > AlgebraicSymMatrix55;
	AlgebraicSymMatrix55 m55;
	for (unsigned int i = 0; i < 5; i++)
		for (unsigned int j = 0; j < 5; j++)
		{
			m55(i, j) = e(i + 1, j + 1);
			if (i == 0)
				m55(i, j) = double(q) * m55(i, j);
			if (j == 0)
				m55(i, j) = double(q) * m55(i, j);

		}
	return m55;
}

/** Convert a CMS Algebraic Sym Matrix (for curv error) to a G4 Trajectory Error Matrix
 */
inline G4ErrorTrajErr algebraicSymMatrix55ToG4ErrorTrajErr(
		const AlgebraicSymMatrix55& e, const int q)
{
	assert(q != 0);
	G4ErrorTrajErr g4err(5, 1);
	for (unsigned int i = 0; i < 5; i++)
		for (unsigned int j = 0; j < 5; j++)
		{
			g4err(i + 1, j + 1) = e(i, j);

			if (i == 0)
				g4err(i + 1, j + 1) = g4err(i + 1, j + 1) * double(q);
			if (j == 0)
				g4err(i + 1, j + 1) = g4err(i + 1, j + 1) * double(q);

		}
	return g4err;
}

}

#endif

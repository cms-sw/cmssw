#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraintT.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

VertexKinematicConstraintT::VertexKinematicConstraintT() {}

VertexKinematicConstraintT::~VertexKinematicConstraintT() {}

void VertexKinematicConstraintT::init(const std::vector<KinematicState>& states,
                                      const GlobalPoint& ipoint,
                                      const GlobalVector& fieldValue) {
  int num = states.size();
  if (num != 2)
    throw VertexException("VertexKinematicConstraintT !=2 states passed");
  double mfz = fieldValue.z();

  int j = 0;
  for (std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++) {
    mom[j] = i->globalMomentum();
    dpos[j] = ipoint - i->globalPosition();
    a_i[j] = -i->particleCharge() * mfz;

    double pvx = mom[j].x() - a_i[j] * dpos[j].y();
    double pvy = mom[j].y() + a_i[j] * dpos[j].x();
    double pvt2 = pvx * pvx + pvy * pvy;
    novera[j] = (dpos[j].x() * mom[j].x() + dpos[j].y() * mom[j].y());
    n[j] = a_i[j] * novera[j];
    m[j] = (pvx * mom[j].x() + pvy * mom[j].y());
    k[j] = -mom[j].z() / (mom[j].perp2() * pvt2);
    delta[j] = std::atan2(n[j], m[j]);

    ++j;
  }
}

void VertexKinematicConstraintT::fillValue() const {
  //it is 2 equations per track
  for (int j = 0; j != 2; ++j) {
    if (a_i[j] != 0) {
      //vector of values
      super::vl(j * 2) = dpos[j].y() * mom[j].x() - dpos[j].x() * mom[j].y() -
                         a_i[j] * (dpos[j].x() * dpos[j].x() + dpos[j].y() * dpos[j].y()) * 0.5;
      super::vl(j * 2 + 1) = dpos[j].z() - mom[j].z() * delta[j] / a_i[j];
    } else {
      //neutral particle
      double pt2Inverse = 1. / mom[j].perp2();
      super::vl(j * 2) = dpos[j].y() * mom[j].x() - dpos[j].x() * mom[j].y();
      super::vl(j * 2 + 1) =
          dpos[j].z() - mom[j].z() * ((dpos[j].x() * mom[j].x() + dpos[j].y() * mom[j].y()) * pt2Inverse);
    }
  }
}

void VertexKinematicConstraintT::fillParametersDerivative() const {
  ROOT::Math::SMatrix<double, 2, 7> el_part_d;
  for (int j = 0; j != 2; ++j) {
    if (a_i[j] != 0) {
      //charged particle

      //D Jacobian matrix
      el_part_d(0, 0) = mom[j].y() + a_i[j] * dpos[j].x();
      el_part_d(0, 1) = -mom[j].x() + a_i[j] * dpos[j].y();
      el_part_d(1, 0) = -k[j] * (m[j] * mom[j].x() - n[j] * mom[j].y());
      el_part_d(1, 1) = -k[j] * (m[j] * mom[j].y() + n[j] * mom[j].x());
      el_part_d(1, 2) = -1.;
      el_part_d(0, 3) = dpos[j].y();
      el_part_d(0, 4) = -dpos[j].x();
      el_part_d(1, 3) = k[j] * (m[j] * dpos[j].x() - novera[j] * (2 * mom[j].x() - a_i[j] * dpos[j].y()));
      el_part_d(1, 4) = k[j] * (m[j] * dpos[j].y() - novera[j] * (2 * mom[j].y() + a_i[j] * dpos[j].x()));
      el_part_d(1, 5) = -delta[j] / a_i[j];
      super::jac_d().Place_at(el_part_d, j * 2, j * 7);
    } else {
      //neutral particle
      double pt2Inverse = 1. / mom[j].perp2();
      el_part_d(0, 0) = mom[j].y();
      el_part_d(0, 1) = -mom[j].x();
      el_part_d(1, 0) = mom[j].x() * (mom[j].z() * pt2Inverse);
      el_part_d(1, 1) = mom[j].y() * (mom[j].z() * pt2Inverse);
      el_part_d(1, 2) = -1.;
      el_part_d(0, 3) = dpos[j].y();
      el_part_d(0, 4) = -dpos[j].x();
      el_part_d(1, 3) = 2 * (dpos[j].x() * mom[j].x() + dpos[j].y() * mom[j].y()) * pt2Inverse * mom[j].x() *
                            (mom[j].z() * pt2Inverse) -
                        dpos[j].x() * (mom[j].z() * pt2Inverse);
      el_part_d(1, 4) = 2 * (dpos[j].x() * mom[j].x() + dpos[j].y() * mom[j].y()) * pt2Inverse * mom[j].y() *
                            (mom[j].z() * pt2Inverse) -
                        dpos[j].x() * (mom[j].z() * pt2Inverse);
      el_part_d(1, 5) = -(dpos[j].x() * mom[j].x() + dpos[j].y() * mom[j].y()) * pt2Inverse;
      super::jac_d().Place_at(el_part_d, j * 2, j * 7);
    }
  }
}

void VertexKinematicConstraintT::fillPositionDerivative() const {
  ROOT::Math::SMatrix<double, 2, 3> el_part_e;
  for (int j = 0; j != 2; ++j) {
    if (a_i[j] != 0) {
      //charged particle

      //E jacobian matrix
      el_part_e(0, 0) = -(mom[j].y() + a_i[j] * dpos[j].x());
      el_part_e(0, 1) = mom[j].x() - a_i[j] * dpos[j].y();
      el_part_e(1, 0) = k[j] * (m[j] * mom[j].x() - n[j] * mom[j].y());
      el_part_e(1, 1) = k[j] * (m[j] * mom[j].y() + n[j] * mom[j].x());
      el_part_e(1, 2) = 1;
      super::jac_e().Place_at(el_part_e, 2 * j, 0);
    } else {
      //neutral particle
      double pt2Inverse = 1. / mom[j].perp2();
      el_part_e(0, 0) = -mom[j].y();
      el_part_e(0, 1) = mom[j].x();
      el_part_e(1, 0) = -mom[j].x() * mom[j].z() * pt2Inverse;
      el_part_e(1, 1) = -mom[j].y() * mom[j].z() * pt2Inverse;
      el_part_e(1, 2) = 1;
      super::jac_e().Place_at(el_part_e, 2 * j, 0);
    }
  }
}

int VertexKinematicConstraintT::numberOfEquations() const { return 2; }

//
// ********************************************************************
// * DISCLAIMER                                                       *
// *                                                                  *
// * The following disclaimer summarizes all the specific disclaimers *
// * of contributors to this software. The specific disclaimers,which *
// * govern, are listed with their locations in:                      *
// *   http://cern.ch/geant4/license                                  *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.                                                             *
// *                                                                  *
// * This  code  implementation is the  intellectual property  of the *
// * GEANT4 collaboration.                                            *
// * By copying,  distributing  or modifying the Program (or any work *
// * based  on  the Program)  you indicate  your  acceptance of  this *
// * statement, and all its terms.                                    *
// ********************************************************************
//
//
//
// Hadronic Process: Reaction Dynamics
// original by H.P. Wellisch
// modified by J.L. Chuma, TRIUMF, 19-Nov-1996
// Last modified: 27-Mar-1997
// modified by H.P. Wellisch, 24-Apr-97
// H.P. Wellisch, 25.Apr-97: Side of current and target particle taken into account
// H.P. Wellisch, 29.Apr-97: Bug fix in NuclearReaction. (pseudo1 was without energy)
// J.L. Chuma, 30-Apr-97:  Changed return value for GenerateXandPt.  It seems possible
//                         that GenerateXandPt could eliminate some secondaries, but
//                         still finish its calculations, thus we would not want to
//                         then use TwoCluster to again calculate the momenta if vecLen
//                         was less than 6.
// J.L. Chuma, 10-Jun-97:  Modified NuclearReaction.  Was not creating ReactionProduct's
//                         with the new operator, thus they would be meaningless when
//                         going out of scope.
// J.L. Chuma, 20-Jun-97:  Modified GenerateXandPt and TwoCluster to fix units problems
// J.L. Chuma, 23-Jun-97:  Modified ProduceStrangeParticlePairs to fix units problems
// J.L. Chuma, 26-Jun-97:  Modified ProduceStrangeParticlePairs to fix array indices
//                         which were sometimes going out of bounds
// J.L. Chuma, 04-Jul-97:  Many minor modifications to GenerateXandPt and TwoCluster
// J.L. Chuma, 06-Aug-97:  Added original incident particle, before Fermi motion and
//                         evaporation effects are included, needed for self absorption
//                         and corrections for single particle spectra (shower particles)
// logging stopped 1997
// J. Allison, 17-Jun-99:  Replaced a min function to get correct behaviour on DEC.

#include "SimG4Core/CustomPhysics/interface/FullModelReactionDynamics.h"
#include "G4AntiProton.hh"
#include "G4AntiNeutron.hh"
#include "Randomize.hh"
#include <iostream>
#include "G4ParticleTable.hh"
#include "G4Poisson.hh"

using namespace CLHEP;

G4bool FullModelReactionDynamics::GenerateXandPt(
    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
    G4int &vecLen,
    G4ReactionProduct &modifiedOriginal,      // Fermi motion & evap. effects included
    const G4HadProjectile *originalIncident,  // the original incident particle
    G4ReactionProduct &currentParticle,
    G4ReactionProduct &targetParticle,
    const G4Nucleus &targetNucleus,
    G4bool &incidentHasChanged,
    G4bool &targetHasChanged,
    G4bool leadFlag,
    G4ReactionProduct &leadingStrangeParticle) {
  //
  // derived from original FORTRAN code GENXPT by H. Fesefeldt (11-Oct-1987)
  //
  //  Generation of X- and PT- values for incident, target, and all secondary particles
  //  A simple single variable description E D3S/DP3= F(Q) with
  //  Q^2 = (M*X)^2 + PT^2 is used. Final state kinematic is produced
  //  by an FF-type iterative cascade method
  //
  //  internal units are GeV
  //
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);

  // Protection in case no secondary has been created; cascades down to two-body.
  if (vecLen == 0)
    return false;

  G4ParticleDefinition *aPiMinus = G4PionMinus::PionMinus();
  G4ParticleDefinition *aProton = G4Proton::Proton();
  G4ParticleDefinition *aNeutron = G4Neutron::Neutron();
  G4ParticleDefinition *aPiPlus = G4PionPlus::PionPlus();
  G4ParticleDefinition *aPiZero = G4PionZero::PionZero();
  G4ParticleDefinition *aKaonPlus = G4KaonPlus::KaonPlus();
  G4ParticleDefinition *aKaonMinus = G4KaonMinus::KaonMinus();
  G4ParticleDefinition *aKaonZeroS = G4KaonZeroShort::KaonZeroShort();
  G4ParticleDefinition *aKaonZeroL = G4KaonZeroLong::KaonZeroLong();

  G4int i, l;
  G4bool veryForward = false;

  const G4double ekOriginal = modifiedOriginal.GetKineticEnergy() / GeV;
  const G4double etOriginal = modifiedOriginal.GetTotalEnergy() / GeV;
  const G4double mOriginal = modifiedOriginal.GetMass() / GeV;
  const G4double pOriginal = modifiedOriginal.GetMomentum().mag() / GeV;
  G4double targetMass = targetParticle.GetDefinition()->GetPDGMass() / GeV;
  G4double centerofmassEnergy =
      std::sqrt(mOriginal * mOriginal + targetMass * targetMass + 2.0 * targetMass * etOriginal);  // GeV
  G4double currentMass = currentParticle.GetMass() / GeV;
  targetMass = targetParticle.GetMass() / GeV;
  //
  //  randomize the order of the secondary particles
  //  note that the current and target particles are not affected
  //
  for (i = 0; i < vecLen; ++i) {
    G4int itemp = G4int(G4UniformRand() * vecLen);
    G4ReactionProduct pTemp = *vec[itemp];
    *vec[itemp] = *vec[i];
    *vec[i] = pTemp;
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  }

  if (currentMass == 0.0 && targetMass == 0.0)  // annihilation
  {
    // no kinetic energy in target .....
    G4double ek = currentParticle.GetKineticEnergy();
    G4ThreeVector m = currentParticle.GetMomentum();
    currentParticle = *vec[0];
    targetParticle = *vec[1];
    for (i = 0; i < (vecLen - 2); ++i)
      *vec[i] = *vec[i + 2];
    G4ReactionProduct *temp = vec[vecLen - 1];
    delete temp;
    temp = vec[vecLen - 2];
    delete temp;
    vecLen -= 2;
    currentMass = currentParticle.GetMass() / GeV;
    targetMass = targetParticle.GetMass() / GeV;
    incidentHasChanged = true;
    targetHasChanged = true;
    currentParticle.SetKineticEnergy(ek);
    currentParticle.SetMomentum(m);
    veryForward = true;
  }
  const G4double atomicWeight = targetNucleus.GetN_asInt();
  const G4double atomicNumber = targetNucleus.GetZ_asInt();
  const G4double protonMass = aProton->GetPDGMass() / MeV;
  if ((originalIncident->GetDefinition() == aKaonMinus || originalIncident->GetDefinition() == aKaonZeroL ||
       originalIncident->GetDefinition() == aKaonZeroS || originalIncident->GetDefinition() == aKaonPlus) &&
      G4UniformRand() >= 0.7) {
    G4ReactionProduct temp = currentParticle;
    currentParticle = targetParticle;
    targetParticle = temp;
    incidentHasChanged = true;
    targetHasChanged = true;
    currentMass = currentParticle.GetMass() / GeV;
    targetMass = targetParticle.GetMass() / GeV;
  }
  const G4double afc = std::min(0.75,
                                0.312 + 0.200 * std::log(std::log(centerofmassEnergy * centerofmassEnergy)) +
                                    std::pow(centerofmassEnergy * centerofmassEnergy, 1.5) / 6000.0);

  G4double freeEnergy = centerofmassEnergy - currentMass - targetMass;

  if (freeEnergy < 0) {
    G4cout << "Free energy < 0!" << G4endl;
    G4cout << "E_CMS = " << centerofmassEnergy << " GeV" << G4endl;
    G4cout << "m_curr = " << currentMass << " GeV" << G4endl;
    G4cout << "m_orig = " << mOriginal << " GeV" << G4endl;
    G4cout << "m_targ = " << targetMass << " GeV" << G4endl;
    G4cout << "E_free = " << freeEnergy << " GeV" << G4endl;
  }

  G4double forwardEnergy = freeEnergy / 2.;
  G4int forwardCount = 1;  // number of particles in forward hemisphere

  G4double backwardEnergy = freeEnergy / 2.;
  G4int backwardCount = 1;  // number of particles in backward hemisphere
  if (veryForward) {
    if (currentParticle.GetSide() == -1) {
      forwardEnergy += currentMass;
      forwardCount--;
      backwardEnergy -= currentMass;
      backwardCount++;
    }
    if (targetParticle.GetSide() != -1) {
      backwardEnergy += targetMass;
      backwardCount--;
      forwardEnergy -= targetMass;
      forwardCount++;
    }
  }
  for (i = 0; i < vecLen; ++i) {
    if (vec[i]->GetSide() == -1) {
      ++backwardCount;
      backwardEnergy -= vec[i]->GetMass() / GeV;
    } else {
      ++forwardCount;
      forwardEnergy -= vec[i]->GetMass() / GeV;
    }
  }
  //
  //  add particles from intranuclear cascade
  //  nuclearExcitationCount = number of new secondaries produced by nuclear excitation
  //  extraCount = number of nucleons within these new secondaries
  //
  G4double xtarg;
  if (centerofmassEnergy < (2.0 + G4UniformRand()))
    xtarg = afc * (std::pow(atomicWeight, 0.33) - 1.0) * (2.0 * backwardCount + vecLen + 2) / 2.0;
  else
    xtarg = afc * (std::pow(atomicWeight, 0.33) - 1.0) * (2.0 * backwardCount);
  if (xtarg <= 0.0)
    xtarg = 0.01;
  G4int nuclearExcitationCount = G4Poisson(xtarg);
  if (atomicWeight < 1.0001)
    nuclearExcitationCount = 0;
  G4int extraNucleonCount = 0;
  if (nuclearExcitationCount > 0) {
    const G4double nucsup[] = {1.00, 0.7, 0.5, 0.4, 0.35, 0.3};
    const G4double psup[] = {3., 6., 20., 50., 100., 1000.};
    G4int momentumBin = 0;
    while ((momentumBin < 6) && (modifiedOriginal.GetTotalMomentum() / GeV > psup[momentumBin]))
      ++momentumBin;
    momentumBin = std::min(5, momentumBin);
    //
    //  NOTE: in GENXPT, these new particles were given negative codes
    //        here I use  NewlyAdded = true  instead
    //
    for (i = 0; i < nuclearExcitationCount; ++i) {
      G4ReactionProduct *pVec = new G4ReactionProduct();
      if (G4UniformRand() < nucsup[momentumBin]) {
        if (G4UniformRand() > 1.0 - atomicNumber / atomicWeight)
          pVec->SetDefinition(aProton);
        else
          pVec->SetDefinition(aNeutron);
        pVec->SetSide(-2);  // -2 means backside nucleon
        ++extraNucleonCount;
        backwardEnergy += pVec->GetMass() / GeV;
      } else {
        G4double ran = G4UniformRand();
        if (ran < 0.3181)
          pVec->SetDefinition(aPiPlus);
        else if (ran < 0.6819)
          pVec->SetDefinition(aPiZero);
        else
          pVec->SetDefinition(aPiMinus);
        pVec->SetSide(-1);  // backside particle, but not a nucleon
      }
      pVec->SetNewlyAdded(true);  // true is the same as IPA(i)<0
      vec.SetElement(vecLen++, pVec);
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
      backwardEnergy -= pVec->GetMass() / GeV;
      ++backwardCount;
    }
  }
  //
  //  assume conservation of kinetic energy in forward & backward hemispheres
  //
  G4int is, iskip;
  while (forwardEnergy <= 0.0)  // must eliminate a particle from the forward side
  {
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    iskip = G4int(G4UniformRand() * forwardCount) + 1;  // 1 <= iskip <= forwardCount
    is = 0;
    G4int forwardParticlesLeft = 0;
    for (i = (vecLen - 1); i >= 0; --i) {
      if (vec[i]->GetSide() == 1 && vec[i]->GetMayBeKilled()) {
        forwardParticlesLeft = 1;
        if (++is == iskip) {
          forwardEnergy += vec[i]->GetMass() / GeV;
          for (G4int j = i; j < (vecLen - 1); j++)
            *vec[j] = *vec[j + 1];  // shift up
          --forwardCount;
          G4ReactionProduct *temp = vec[vecLen - 1];
          delete temp;
          if (--vecLen == 0)
            return false;  // all the secondaries have been eliminated
          break;           // --+
        }                  //   |
      }                    //   |
    }                      // break goes down to here
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    if (forwardParticlesLeft == 0) {
      forwardEnergy += currentParticle.GetMass() / GeV;
      currentParticle.SetDefinitionAndUpdateE(targetParticle.GetDefinition());
      targetParticle.SetDefinitionAndUpdateE(vec[0]->GetDefinition());
      // above two lines modified 20-oct-97: were just simple equalities
      --forwardCount;
      for (G4int j = 0; j < (vecLen - 1); ++j)
        *vec[j] = *vec[j + 1];
      G4ReactionProduct *temp = vec[vecLen - 1];
      delete temp;
      if (--vecLen == 0)
        return false;  // all the secondaries have been eliminated
      break;
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  while (backwardEnergy <= 0.0)  // must eliminate a particle from the backward side
  {
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    iskip = G4int(G4UniformRand() * backwardCount) + 1;  // 1 <= iskip <= backwardCount
    is = 0;
    G4int backwardParticlesLeft = 0;
    for (i = (vecLen - 1); i >= 0; --i) {
      if (vec[i]->GetSide() < 0 && vec[i]->GetMayBeKilled()) {
        backwardParticlesLeft = 1;
        if (++is == iskip)  // eliminate the i'th particle
        {
          if (vec[i]->GetSide() == -2) {
            --extraNucleonCount;
            //extraNucleonMass -= vec[i]->GetMass() / GeV;
            backwardEnergy -= vec[i]->GetTotalEnergy() / GeV;
          }
          backwardEnergy += vec[i]->GetTotalEnergy() / GeV;
          for (G4int j = i; j < (vecLen - 1); ++j)
            *vec[j] = *vec[j + 1];  // shift up
          --backwardCount;
          G4ReactionProduct *temp = vec[vecLen - 1];
          delete temp;
          if (--vecLen == 0)
            return false;  // all the secondaries have been eliminated
          break;
        }
      }
    }
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    if (backwardParticlesLeft == 0) {
      backwardEnergy += targetParticle.GetMass() / GeV;
      targetParticle = *vec[0];
      --backwardCount;
      for (G4int j = 0; j < (vecLen - 1); ++j)
        *vec[j] = *vec[j + 1];
      G4ReactionProduct *temp = vec[vecLen - 1];
      delete temp;
      if (--vecLen == 0)
        return false;  // all the secondaries have been eliminated
      break;
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  //
  //  define initial state vectors for Lorentz transformations
  //  the pseudoParticles have non-standard masses, hence the "pseudo"
  //
  G4ReactionProduct pseudoParticle[10];
  for (i = 0; i < 10; ++i)
    pseudoParticle[i].SetZero();

  pseudoParticle[0].SetMass(mOriginal * GeV);
  pseudoParticle[0].SetMomentum(0.0, 0.0, pOriginal * GeV);
  pseudoParticle[0].SetTotalEnergy(std::sqrt(pOriginal * pOriginal + mOriginal * mOriginal) * GeV);

  pseudoParticle[1].SetMass(protonMass * MeV);  // this could be targetMass
  pseudoParticle[1].SetTotalEnergy(protonMass * MeV);

  pseudoParticle[3].SetMass(protonMass * (1 + extraNucleonCount) * MeV);
  pseudoParticle[3].SetTotalEnergy(protonMass * (1 + extraNucleonCount) * MeV);

  pseudoParticle[8].SetMomentum(1.0 * GeV, 0.0, 0.0);

  pseudoParticle[2] = pseudoParticle[0] + pseudoParticle[1];
  pseudoParticle[3] = pseudoParticle[3] + pseudoParticle[0];

  pseudoParticle[0].Lorentz(pseudoParticle[0], pseudoParticle[2]);
  pseudoParticle[1].Lorentz(pseudoParticle[1], pseudoParticle[2]);

  G4double dndl[20];
  //
  //  main loop for 4-momentum generation
  //  see Pitha-report (Aachen) for a detailed description of the method
  //
  G4double aspar, pt, et, x, pp, pp1, rthnve, phinve, rmb, wgt;
  G4int innerCounter, outerCounter;
  G4bool eliminateThisParticle, resetEnergies, constantCrossSection;

  G4double forwardKinetic = 0.0, backwardKinetic = 0.0;
  //
  // process the secondary particles in reverse order
  // the incident particle is Done after the secondaries
  // nucleons, including the target, in the backward hemisphere are also Done later
  //
  G4double binl[20] = {0.,  0.1,  0.2,  0.3,  0.4,  0.5, 0.6, 0.7,  0.8,  0.9,
                       1.0, 1.11, 1.25, 1.43, 1.67, 2.0, 2.5, 3.33, 5.00, 10.00};
  G4int backwardNucleonCount = 0;  // number of nucleons in backward hemisphere
  G4double totalEnergy, kineticEnergy, vecMass;

  for (i = (vecLen - 1); i >= 0; --i) {
    G4double phi = G4UniformRand() * twopi;
    if (vec[i]->GetNewlyAdded())  // added from intranuclear cascade
    {
      if (vec[i]->GetSide() == -2)  //  is a nucleon
      {
        if (backwardNucleonCount < 18) {
          if (vec[i]->GetDefinition() == G4PionMinus::PionMinus() ||
              vec[i]->GetDefinition() == G4PionPlus::PionPlus() || vec[i]->GetDefinition() == G4PionZero::PionZero()) {
            for (G4int i = 0; i < vecLen; i++)
              delete vec[i];
            vecLen = 0;
            G4ExceptionDescription ed;
            ed << "FullModelReactionDynamics::GenerateXandPt : a pion has been counted as a backward nucleon";
            G4Exception("FullModelReactionDynamics::GenerateXandPt", "had064", FatalException, ed);
          }
          vec[i]->SetSide(-3);
          ++backwardNucleonCount;
          continue;
        }
      }
    }
    //
    //  set pt and phi values, they are changed somewhat in the iteration loop
    //  set mass parameter for lambda fragmentation model
    //
    vecMass = vec[i]->GetMass() / GeV;
    G4double ran = -std::log(1.0 - G4UniformRand()) / 3.5;
    if (vec[i]->GetSide() == -2)  // backward nucleon
    {
      if (vec[i]->GetDefinition() == aKaonMinus || vec[i]->GetDefinition() == aKaonZeroL ||
          vec[i]->GetDefinition() == aKaonZeroS || vec[i]->GetDefinition() == aKaonPlus ||
          vec[i]->GetDefinition() == aPiMinus || vec[i]->GetDefinition() == aPiZero ||
          vec[i]->GetDefinition() == aPiPlus) {
        aspar = 0.75;
        pt = std::sqrt(std::pow(ran, 1.7));
      } else {         // vec[i] must be a proton, neutron,
        aspar = 0.20;  //  lambda, sigma, xsi, or ion
        pt = std::sqrt(std::pow(ran, 1.2));
      }
    } else {  // not a backward nucleon
      if (vec[i]->GetDefinition() == aPiMinus || vec[i]->GetDefinition() == aPiZero ||
          vec[i]->GetDefinition() == aPiPlus) {
        aspar = 0.75;
        pt = std::sqrt(std::pow(ran, 1.7));
      } else if (vec[i]->GetDefinition() == aKaonMinus || vec[i]->GetDefinition() == aKaonZeroL ||
                 vec[i]->GetDefinition() == aKaonZeroS || vec[i]->GetDefinition() == aKaonPlus) {
        aspar = 0.70;
        pt = std::sqrt(std::pow(ran, 1.7));
      } else {         // vec[i] must be a proton, neutron,
        aspar = 0.65;  //  lambda, sigma, xsi, or ion
        pt = std::sqrt(std::pow(ran, 1.5));
      }
    }
    pt = std::max(0.001, pt);
    vec[i]->SetMomentum(pt * std::cos(phi) * GeV, pt * std::sin(phi) * GeV);
    for (G4int j = 0; j < 20; ++j)
      binl[j] = j / (19. * pt);
    if (vec[i]->GetSide() > 0)
      et = pseudoParticle[0].GetTotalEnergy() / GeV;
    else
      et = pseudoParticle[1].GetTotalEnergy() / GeV;
    dndl[0] = 0.0;
    //
    //   start of outer iteration loop
    //
    outerCounter = 0;
    eliminateThisParticle = true;
    resetEnergies = true;
    while (++outerCounter < 3) {
      for (l = 1; l < 20; ++l) {
        x = (binl[l] + binl[l - 1]) / 2.;
        pt = std::max(0.001, pt);
        if (x > 1.0 / pt)
          dndl[l] += dndl[l - 1];  //  changed from just  =  on 02 April 98
        else
          dndl[l] = et * aspar / std::sqrt(std::pow((1. + aspar * x * aspar * x), 3)) * (binl[l] - binl[l - 1]) /
                        std::sqrt(pt * x * et * pt * x * et + pt * pt + vecMass * vecMass) +
                    dndl[l - 1];
      }
      innerCounter = 0;
      vec[i]->SetMomentum(pt * std::cos(phi) * GeV, pt * std::sin(phi) * GeV);
      //
      //   start of inner iteration loop
      //
      while (++innerCounter < 7) {
        ran = G4UniformRand() * dndl[19];
        l = 1;
        while ((ran >= dndl[l]) && (l < 20))
          l++;
        l = std::min(19, l);
        x = std::min(1.0, pt * (binl[l - 1] + G4UniformRand() * (binl[l] - binl[l - 1]) / 2.));
        if (vec[i]->GetSide() < 0)
          x *= -1.;
        vec[i]->SetMomentum(x * et * GeV);  // set the z-momentum
        totalEnergy = std::sqrt(x * et * x * et + pt * pt + vecMass * vecMass);
        vec[i]->SetTotalEnergy(totalEnergy * GeV);
        kineticEnergy = vec[i]->GetKineticEnergy() / GeV;
        if (vec[i]->GetSide() > 0)  // forward side
        {
          if ((forwardKinetic + kineticEnergy) < 0.95 * forwardEnergy) {
            pseudoParticle[4] = pseudoParticle[4] + (*vec[i]);
            forwardKinetic += kineticEnergy;
            pseudoParticle[6] = pseudoParticle[4] + pseudoParticle[5];
            pseudoParticle[6].SetMomentum(0.0);  // set the z-momentum
            phi = pseudoParticle[6].Angle(pseudoParticle[8]);
            if (pseudoParticle[6].GetMomentum().y() / MeV < 0.0)
              phi = twopi - phi;
            phi += pi + normal() * pi / 12.0;
            if (phi > twopi)
              phi -= twopi;
            if (phi < 0.0)
              phi = twopi - phi;
            outerCounter = 2;               // leave outer loop
            eliminateThisParticle = false;  // don't eliminate this particle
            resetEnergies = false;
            break;  // leave inner loop
          }
          if (innerCounter > 5)
            break;                        // leave inner loop
          if (backwardEnergy >= vecMass)  // switch sides
          {
            vec[i]->SetSide(-1);
            forwardEnergy += vecMass;
            backwardEnergy -= vecMass;
            ++backwardCount;
          }
        } else {  // backward side
          G4double xxx = 0.95 + 0.05 * extraNucleonCount / 20.0;
          if ((backwardKinetic + kineticEnergy) < xxx * backwardEnergy) {
            pseudoParticle[5] = pseudoParticle[5] + (*vec[i]);
            backwardKinetic += kineticEnergy;
            pseudoParticle[6] = pseudoParticle[4] + pseudoParticle[5];
            pseudoParticle[6].SetMomentum(0.0);  // set the z-momentum
            phi = pseudoParticle[6].Angle(pseudoParticle[8]);
            if (pseudoParticle[6].GetMomentum().y() / MeV < 0.0)
              phi = twopi - phi;
            phi += pi + normal() * pi / 12.0;
            if (phi > twopi)
              phi -= twopi;
            if (phi < 0.0)
              phi = twopi - phi;
            outerCounter = 2;               // leave outer loop
            eliminateThisParticle = false;  // don't eliminate this particle
            resetEnergies = false;
            break;  // leave inner loop
          }
          if (innerCounter > 5)
            break;                       // leave inner loop
          if (forwardEnergy >= vecMass)  // switch sides
          {
            vec[i]->SetSide(1);
            forwardEnergy -= vecMass;
            backwardEnergy += vecMass;
            backwardCount--;
          }
        }
        G4ThreeVector momentum = vec[i]->GetMomentum();
        vec[i]->SetMomentum(momentum.x() * 0.9, momentum.y() * 0.9);
        pt *= 0.9;
        dndl[19] *= 0.9;
      }  // closes inner loop
      if (resetEnergies) {
        //   if we get to here, the inner loop has been Done 6 Times
        //   reset the kinetic energies of previously Done particles, if they are lighter
        //    than protons and in the forward hemisphere
        //   then continue with outer loop
        //
        forwardKinetic = 0.0;
        backwardKinetic = 0.0;
        pseudoParticle[4].SetZero();
        pseudoParticle[5].SetZero();
        for (l = i + 1; l < vecLen; ++l) {
          if (vec[l]->GetSide() > 0 || vec[l]->GetDefinition() == aKaonMinus || vec[l]->GetDefinition() == aKaonZeroL ||
              vec[l]->GetDefinition() == aKaonZeroS || vec[l]->GetDefinition() == aKaonPlus ||
              vec[l]->GetDefinition() == aPiMinus || vec[l]->GetDefinition() == aPiZero ||
              vec[l]->GetDefinition() == aPiPlus) {
            G4double tempMass = vec[l]->GetMass() / MeV;
            totalEnergy = 0.95 * vec[l]->GetTotalEnergy() / MeV + 0.05 * tempMass;
            totalEnergy = std::max(tempMass, totalEnergy);
            vec[l]->SetTotalEnergy(totalEnergy * MeV);
            pp = std::sqrt(std::abs(totalEnergy * totalEnergy - tempMass * tempMass));
            pp1 = vec[l]->GetMomentum().mag() / MeV;
            if (pp1 < 1.0e-6 * GeV) {
              G4double rthnve = pi * G4UniformRand();
              G4double phinve = twopi * G4UniformRand();
              G4double srth = std::sin(rthnve);
              vec[l]->SetMomentum(
                  pp * srth * std::cos(phinve) * MeV, pp * srth * std::sin(phinve) * MeV, pp * std::cos(rthnve) * MeV);
            } else {
              vec[l]->SetMomentum(vec[l]->GetMomentum() * (pp / pp1));
            }
            G4double px = vec[l]->GetMomentum().x() / MeV;
            G4double py = vec[l]->GetMomentum().y() / MeV;
            pt = std::max(1.0, std::sqrt(px * px + py * py)) / GeV;
            if (vec[l]->GetSide() > 0) {
              forwardKinetic += vec[l]->GetKineticEnergy() / GeV;
              pseudoParticle[4] = pseudoParticle[4] + (*vec[l]);
            } else {
              backwardKinetic += vec[l]->GetKineticEnergy() / GeV;
              pseudoParticle[5] = pseudoParticle[5] + (*vec[l]);
            }
          }
        }
      }
    }  // closes outer loop

    if (eliminateThisParticle && vec[i]->GetMayBeKilled())  // not enough energy, eliminate this particle
    {
      if (vec[i]->GetSide() > 0) {
        --forwardCount;
        forwardEnergy += vecMass;
      } else {
        if (vec[i]->GetSide() == -2) {
          --extraNucleonCount;
          backwardEnergy -= vecMass;
        }
        --backwardCount;
        backwardEnergy += vecMass;
      }
      for (G4int j = i; j < (vecLen - 1); ++j)
        *vec[j] = *vec[j + 1];  // shift up
      G4ReactionProduct *temp = vec[vecLen - 1];
      delete temp;
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
      if (--vecLen == 0)
        return false;  // all the secondaries have been eliminated
      pseudoParticle[6] = pseudoParticle[4] + pseudoParticle[5];
      pseudoParticle[6].SetMomentum(0.0);  // set z-momentum
    }
  }  // closes main for loop

  //
  //  for the incident particle:  it was placed in the forward hemisphere
  //   set pt and phi values, they are changed somewhat in the iteration loop
  //   set mass parameter for lambda fragmentation model
  //
  G4double phi = G4UniformRand() * twopi;
  G4double ran = -std::log(1.0 - G4UniformRand());
  if (currentParticle.GetDefinition() == aPiMinus || currentParticle.GetDefinition() == aPiZero ||
      currentParticle.GetDefinition() == aPiPlus) {
    aspar = 0.60;
    pt = std::sqrt(std::pow(ran / 6.0, 1.7));
  } else if (currentParticle.GetDefinition() == aKaonMinus || currentParticle.GetDefinition() == aKaonZeroL ||
             currentParticle.GetDefinition() == aKaonZeroS || currentParticle.GetDefinition() == aKaonPlus) {
    aspar = 0.50;
    pt = std::sqrt(std::pow(ran / 5.0, 1.4));
  } else {
    aspar = 0.40;
    pt = std::sqrt(std::pow(ran / 4.0, 1.2));
  }
  for (G4int j = 0; j < 20; ++j)
    binl[j] = j / (19. * pt);
  currentParticle.SetMomentum(pt * std::cos(phi) * GeV, pt * std::sin(phi) * GeV);
  et = pseudoParticle[0].GetTotalEnergy() / GeV;
  dndl[0] = 0.0;
  vecMass = currentParticle.GetMass() / GeV;
  for (l = 1; l < 20; ++l) {
    x = (binl[l] + binl[l - 1]) / 2.;
    if (x > 1.0 / pt)
      dndl[l] += dndl[l - 1];  //  changed from just  =   on 02 April 98
    else
      dndl[l] = aspar / std::sqrt(std::pow((1. + sqr(aspar * x)), 3)) * (binl[l] - binl[l - 1]) * et /
                    std::sqrt(pt * x * et * pt * x * et + pt * pt + vecMass * vecMass) +
                dndl[l - 1];
  }
  ran = G4UniformRand() * dndl[19];
  l = 1;
  while ((ran > dndl[l]) && (l < 20))
    l++;
  l = std::min(19, l);
  x = std::min(1.0, pt * (binl[l - 1] + G4UniformRand() * (binl[l] - binl[l - 1]) / 2.));
  currentParticle.SetMomentum(x * et * GeV);  // set the z-momentum
  if (forwardEnergy < forwardKinetic)
    totalEnergy = vecMass + 0.04 * std::fabs(normal());
  else
    totalEnergy = vecMass + forwardEnergy - forwardKinetic;
  currentParticle.SetTotalEnergy(totalEnergy * GeV);
  pp = std::sqrt(std::abs(totalEnergy * totalEnergy - vecMass * vecMass)) * GeV;
  pp1 = currentParticle.GetMomentum().mag() / MeV;
  if (pp1 < 1.0e-6 * GeV) {
    G4double rthnve = pi * G4UniformRand();
    G4double phinve = twopi * G4UniformRand();
    G4double srth = std::sin(rthnve);
    currentParticle.SetMomentum(
        pp * srth * std::cos(phinve) * MeV, pp * srth * std::sin(phinve) * MeV, pp * std::cos(rthnve) * MeV);
  } else {
    currentParticle.SetMomentum(currentParticle.GetMomentum() * (pp / pp1));
  }
  pseudoParticle[4] = pseudoParticle[4] + currentParticle;
  //
  // this finishes the current particle
  // now for the target particle
  //
  if (backwardNucleonCount < 18) {
    targetParticle.SetSide(-3);
    ++backwardNucleonCount;
  } else {
    //  set pt and phi values, they are changed somewhat in the iteration loop
    //  set mass parameter for lambda fragmentation model
    //
    vecMass = targetParticle.GetMass() / GeV;
    ran = -std::log(1.0 - G4UniformRand());
    aspar = 0.40;
    pt = std::max(0.001, std::sqrt(std::pow(ran / 4.0, 1.2)));
    targetParticle.SetMomentum(pt * std::cos(phi) * GeV, pt * std::sin(phi) * GeV);
    for (G4int j = 0; j < 20; ++j)
      binl[j] = (j - 1.) / (19. * pt);
    et = pseudoParticle[1].GetTotalEnergy() / GeV;
    dndl[0] = 0.0;
    outerCounter = 0;
    resetEnergies = true;
    while (++outerCounter < 3)  // start of outer iteration loop
    {
      for (l = 1; l < 20; ++l) {
        x = (binl[l] + binl[l - 1]) / 2.;
        if (x > 1.0 / pt)
          dndl[l] += dndl[l - 1];  // changed from just  =  on 02 April 98
        else
          dndl[l] = aspar / std::sqrt(std::pow((1. + aspar * x * aspar * x), 3)) * (binl[l] - binl[l - 1]) * et /
                        std::sqrt(pt * x * et * pt * x * et + pt * pt + vecMass * vecMass) +
                    dndl[l - 1];
      }
      innerCounter = 0;
      while (++innerCounter < 7)  // start of inner iteration loop
      {
        l = 1;
        ran = G4UniformRand() * dndl[19];
        while ((ran >= dndl[l]) && (l < 20))
          l++;
        l = std::min(19, l);
        x = std::min(1.0, pt * (binl[l - 1] + G4UniformRand() * (binl[l] - binl[l - 1]) / 2.));
        if (targetParticle.GetSide() < 0)
          x *= -1.;
        targetParticle.SetMomentum(x * et * GeV);  // set the z-momentum
        totalEnergy = std::sqrt(x * et * x * et + pt * pt + vecMass * vecMass);
        targetParticle.SetTotalEnergy(totalEnergy * GeV);
        if (targetParticle.GetSide() < 0) {
          G4double xxx = 0.95 + 0.05 * extraNucleonCount / 20.0;
          if ((backwardKinetic + totalEnergy - vecMass) < xxx * backwardEnergy) {
            pseudoParticle[5] = pseudoParticle[5] + targetParticle;
            backwardKinetic += totalEnergy - vecMass;
            pseudoParticle[6] = pseudoParticle[4] + pseudoParticle[5];
            pseudoParticle[6].SetMomentum(0.0);  // set z-momentum
            outerCounter = 2;                    // leave outer loop
            resetEnergies = false;
            break;  // leave inner loop
          }
          if (innerCounter > 5)
            break;                       // leave inner loop
          if (forwardEnergy >= vecMass)  // switch sides
          {
            targetParticle.SetSide(1);
            forwardEnergy -= vecMass;
            backwardEnergy += vecMass;
            --backwardCount;
          }
          G4ThreeVector momentum = targetParticle.GetMomentum();
          targetParticle.SetMomentum(momentum.x() * 0.9, momentum.y() * 0.9);
          pt *= 0.9;
          dndl[19] *= 0.9;
        } else  // target has gone to forward side
        {
          if (forwardEnergy < forwardKinetic)
            totalEnergy = vecMass + 0.04 * std::fabs(normal());
          else
            totalEnergy = vecMass + forwardEnergy - forwardKinetic;
          targetParticle.SetTotalEnergy(totalEnergy * GeV);
          pp = std::sqrt(std::abs(totalEnergy * totalEnergy - vecMass * vecMass)) * GeV;
          pp1 = targetParticle.GetMomentum().mag() / MeV;
          if (pp1 < 1.0e-6 * GeV) {
            G4double rthnve = pi * G4UniformRand();
            G4double phinve = twopi * G4UniformRand();
            G4double srth = std::sin(rthnve);
            targetParticle.SetMomentum(
                pp * srth * std::cos(phinve) * MeV, pp * srth * std::sin(phinve) * MeV, pp * std::cos(rthnve) * MeV);
          } else
            targetParticle.SetMomentum(targetParticle.GetMomentum() * (pp / pp1));

          pseudoParticle[4] = pseudoParticle[4] + targetParticle;
          outerCounter = 2;  // leave outer loop
          //eliminateThisParticle = false;       // don't eliminate this particle
          resetEnergies = false;
          break;  // leave inner loop
        }
      }  // closes inner loop
      if (resetEnergies) {
        //   if we get to here, the inner loop has been Done 6 Times
        //   reset the kinetic energies of previously Done particles, if they are lighter
        //    than protons and in the forward hemisphere
        //   then continue with outer loop

        forwardKinetic = backwardKinetic = 0.0;
        pseudoParticle[4].SetZero();
        pseudoParticle[5].SetZero();
        for (l = 0; l < vecLen; ++l)  // changed from l=1  on 02 April 98
        {
          if (vec[l]->GetSide() > 0 || vec[l]->GetDefinition() == aKaonMinus || vec[l]->GetDefinition() == aKaonZeroL ||
              vec[l]->GetDefinition() == aKaonZeroS || vec[l]->GetDefinition() == aKaonPlus ||
              vec[l]->GetDefinition() == aPiMinus || vec[l]->GetDefinition() == aPiZero ||
              vec[l]->GetDefinition() == aPiPlus) {
            G4double tempMass = vec[l]->GetMass() / GeV;
            totalEnergy = std::max(tempMass, 0.95 * vec[l]->GetTotalEnergy() / GeV + 0.05 * tempMass);
            vec[l]->SetTotalEnergy(totalEnergy * GeV);
            pp = std::sqrt(std::abs(totalEnergy * totalEnergy - tempMass * tempMass)) * GeV;
            pp1 = vec[l]->GetMomentum().mag() / MeV;
            if (pp1 < 1.0e-6 * GeV) {
              G4double rthnve = pi * G4UniformRand();
              G4double phinve = twopi * G4UniformRand();
              G4double srth = std::sin(rthnve);
              vec[l]->SetMomentum(
                  pp * srth * std::cos(phinve) * MeV, pp * srth * std::sin(phinve) * MeV, pp * std::cos(rthnve) * MeV);
            } else
              vec[l]->SetMomentum(vec[l]->GetMomentum() * (pp / pp1));

            pt = std::max(0.001 * GeV,
                          std::sqrt(sqr(vec[l]->GetMomentum().x() / MeV) + sqr(vec[l]->GetMomentum().y() / MeV))) /
                 GeV;
            if (vec[l]->GetSide() > 0) {
              forwardKinetic += vec[l]->GetKineticEnergy() / GeV;
              pseudoParticle[4] = pseudoParticle[4] + (*vec[l]);
            } else {
              backwardKinetic += vec[l]->GetKineticEnergy() / GeV;
              pseudoParticle[5] = pseudoParticle[5] + (*vec[l]);
            }
          }
        }
      }
    }  // closes outer loop
  }
  //
  //  this finishes the target particle
  // backward nucleons produced with a cluster model
  //
  pseudoParticle[6].Lorentz(pseudoParticle[3], pseudoParticle[2]);
  pseudoParticle[6] = pseudoParticle[6] - pseudoParticle[4];
  pseudoParticle[6] = pseudoParticle[6] - pseudoParticle[5];
  if (backwardNucleonCount == 1)  // target particle is the only backward nucleon
  {
    G4double ekin = std::min(backwardEnergy - backwardKinetic, centerofmassEnergy / 2.0 - protonMass / GeV);
    if (ekin < 0.04)
      ekin = 0.04 * std::fabs(normal());
    vecMass = targetParticle.GetMass() / GeV;
    totalEnergy = ekin + vecMass;
    targetParticle.SetTotalEnergy(totalEnergy * GeV);
    pp = std::sqrt(std::abs(totalEnergy * totalEnergy - vecMass * vecMass)) * GeV;
    pp1 = pseudoParticle[6].GetMomentum().mag() / MeV;
    if (pp1 < 1.0e-6 * GeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      G4double srth = std::sin(rthnve);
      targetParticle.SetMomentum(
          pp * srth * std::cos(phinve) * MeV, pp * srth * std::sin(phinve) * MeV, pp * std::cos(rthnve) * MeV);
    } else {
      targetParticle.SetMomentum(pseudoParticle[6].GetMomentum() * (pp / pp1));
    }
    pseudoParticle[5] = pseudoParticle[5] + targetParticle;
  } else  // more than one backward nucleon
  {
    const G4double cpar[] = {0.6, 0.6, 0.35, 0.15, 0.10};
    const G4double gpar[] = {2.6, 2.6, 1.80, 1.30, 1.20};
    // Replaced the following min function to get correct behaviour on DEC.
    G4int tempCount = std::max(1, std::min(5, backwardNucleonCount)) - 1;
    //G4cout << "backwardNucleonCount " << backwardNucleonCount << G4endl;
    //G4cout << "tempCount " << tempCount << G4endl;
    G4double rmb0 = 0.0;
    if (targetParticle.GetSide() == -3)
      rmb0 += targetParticle.GetMass() / GeV;
    for (i = 0; i < vecLen; ++i) {
      if (vec[i]->GetSide() == -3)
        rmb0 += vec[i]->GetMass() / GeV;
    }
    rmb = rmb0 + std::pow(-std::log(1.0 - G4UniformRand()), cpar[tempCount]) / gpar[tempCount];
    totalEnergy = pseudoParticle[6].GetTotalEnergy() / GeV;
    vecMass = std::min(rmb, totalEnergy);
    pseudoParticle[6].SetMass(vecMass * GeV);
    pp = std::sqrt(std::abs(totalEnergy * totalEnergy - vecMass * vecMass)) * GeV;
    pp1 = pseudoParticle[6].GetMomentum().mag() / MeV;
    if (pp1 < 1.0e-6 * GeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      G4double srth = std::sin(rthnve);
      pseudoParticle[6].SetMomentum(
          -pp * srth * std::cos(phinve) * MeV, -pp * srth * std::sin(phinve) * MeV, -pp * std::cos(rthnve) * MeV);
    } else
      pseudoParticle[6].SetMomentum(pseudoParticle[6].GetMomentum() * (-pp / pp1));

    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> tempV;  // tempV contains the backward nucleons
    tempV.Initialize(backwardNucleonCount);
    G4int tempLen = 0;
    if (targetParticle.GetSide() == -3)
      tempV.SetElement(tempLen++, &targetParticle);
    for (i = 0; i < vecLen; ++i) {
      if (vec[i]->GetSide() == -3)
        tempV.SetElement(tempLen++, vec[i]);
    }
    if (tempLen != backwardNucleonCount) {
      G4ExceptionDescription ed;
      ed << "tempLen is not the same as backwardNucleonCount" << G4endl;
      ed << "tempLen = " << tempLen << ", backwardNucleonCount = " << backwardNucleonCount << G4endl;
      ed << "targetParticle side = " << targetParticle.GetSide() << G4endl;
      ed << "currentParticle side = " << currentParticle.GetSide() << G4endl;
      for (i = 0; i < vecLen; ++i)
        ed << "particle #" << i << " side = " << vec[i]->GetSide() << G4endl;
      G4Exception("FullModelReactionDynamics::GenerateXandPt", "had064", FatalException, ed);
    }
    constantCrossSection = true;
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    if (tempLen >= 2) {
      GenerateNBodyEvent(pseudoParticle[6].GetMass(), constantCrossSection, tempV, tempLen);
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
      if (targetParticle.GetSide() == -3) {
        targetParticle.Lorentz(targetParticle, pseudoParticle[6]);
        // tempV contains the real stuff
        pseudoParticle[5] = pseudoParticle[5] + targetParticle;
      }
      for (i = 0; i < vecLen; ++i) {
        if (vec[i]->GetSide() == -3) {
          vec[i]->Lorentz(*vec[i], pseudoParticle[6]);
          pseudoParticle[5] = pseudoParticle[5] + (*vec[i]);
        }
      }
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    }
  }
  //
  //  Lorentz transformation in lab system
  //
  if (vecLen == 0)
    return false;  // all the secondaries have been eliminated
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);

  G4int numberofFinalStateNucleons =
      currentParticle.GetDefinition()->GetBaryonNumber() + targetParticle.GetDefinition()->GetBaryonNumber();
  currentParticle.Lorentz(currentParticle, pseudoParticle[1]);
  targetParticle.Lorentz(targetParticle, pseudoParticle[1]);

  for (i = 0; i < vecLen; ++i) {
    numberofFinalStateNucleons += vec[i]->GetDefinition()->GetBaryonNumber();
    vec[i]->Lorentz(*vec[i], pseudoParticle[1]);
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  if (veryForward)
    numberofFinalStateNucleons++;
  numberofFinalStateNucleons = std::max(1, numberofFinalStateNucleons);
  //
  // leadFlag will be true
  //  iff original particle is at least as heavy as K+ and not a proton or neutron AND
  //   if
  //    incident particle is at least as heavy as K+ and it is not a proton or neutron
  //     leadFlag is set to the incident particle
  //   or
  //    target particle is at least as heavy as K+ and it is not a proton or neutron
  //     leadFlag is set to the target particle
  //
  G4bool leadingStrangeParticleHasChanged = true;
  if (leadFlag) {
    if (currentParticle.GetDefinition() == leadingStrangeParticle.GetDefinition())
      leadingStrangeParticleHasChanged = false;
    if (leadingStrangeParticleHasChanged && (targetParticle.GetDefinition() == leadingStrangeParticle.GetDefinition()))
      leadingStrangeParticleHasChanged = false;
    if (leadingStrangeParticleHasChanged) {
      for (i = 0; i < vecLen; i++) {
        if (vec[i]->GetDefinition() == leadingStrangeParticle.GetDefinition()) {
          leadingStrangeParticleHasChanged = false;
          break;
        }
      }
    }
    if (leadingStrangeParticleHasChanged) {
      G4bool leadTest =
          (leadingStrangeParticle.GetDefinition() == aKaonMinus ||
           leadingStrangeParticle.GetDefinition() == aKaonZeroL ||
           leadingStrangeParticle.GetDefinition() == aKaonZeroS ||
           leadingStrangeParticle.GetDefinition() == aKaonPlus || leadingStrangeParticle.GetDefinition() == aPiMinus ||
           leadingStrangeParticle.GetDefinition() == aPiZero || leadingStrangeParticle.GetDefinition() == aPiPlus);
      G4bool targetTest = false;

      // following modified by JLC 22-Oct-97

      if ((leadTest && targetTest) || !(leadTest || targetTest))  // both true or both false
      {
        targetParticle.SetDefinitionAndUpdateE(leadingStrangeParticle.GetDefinition());
        targetHasChanged = true;
        // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
      } else {
        currentParticle.SetDefinitionAndUpdateE(leadingStrangeParticle.GetDefinition());
        incidentHasChanged = false;
        // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
      }
    }
  }  // end of if( leadFlag )

  pseudoParticle[3].SetMomentum(0.0, 0.0, pOriginal * GeV);
  pseudoParticle[3].SetMass(mOriginal * GeV);
  pseudoParticle[3].SetTotalEnergy(std::sqrt(pOriginal * pOriginal + mOriginal * mOriginal) * GeV);

  const G4ParticleDefinition *aOrgDef = modifiedOriginal.GetDefinition();
  G4int diff = 0;
  if (aOrgDef == G4Proton::Proton() || aOrgDef == G4Neutron::Neutron())
    diff = 1;
  if (numberofFinalStateNucleons == 1)
    diff = 0;
  pseudoParticle[4].SetMomentum(0.0, 0.0, 0.0);
  pseudoParticle[4].SetMass(protonMass * (numberofFinalStateNucleons - diff) * MeV);
  pseudoParticle[4].SetTotalEnergy(protonMass * (numberofFinalStateNucleons - diff) * MeV);

  G4double theoreticalKinetic = pseudoParticle[3].GetTotalEnergy() / MeV + pseudoParticle[4].GetTotalEnergy() / MeV -
                                currentParticle.GetMass() / MeV - targetParticle.GetMass() / MeV;

  G4double simulatedKinetic = currentParticle.GetKineticEnergy() / MeV + targetParticle.GetKineticEnergy() / MeV;

  pseudoParticle[5] = pseudoParticle[3] + pseudoParticle[4];
  pseudoParticle[3].Lorentz(pseudoParticle[3], pseudoParticle[5]);
  pseudoParticle[4].Lorentz(pseudoParticle[4], pseudoParticle[5]);

  pseudoParticle[7].SetZero();
  pseudoParticle[7] = pseudoParticle[7] + currentParticle;
  pseudoParticle[7] = pseudoParticle[7] + targetParticle;

  for (i = 0; i < vecLen; ++i) {
    pseudoParticle[7] = pseudoParticle[7] + *vec[i];
    simulatedKinetic += vec[i]->GetKineticEnergy() / MeV;
    theoreticalKinetic -= vec[i]->GetMass() / MeV;
  }
  if (vecLen <= 16 && vecLen > 0) {
    // must create a new set of ReactionProducts here because GenerateNBody will
    // modify the momenta for the particles, and we don't want to do this
    //
    G4ReactionProduct tempR[130];
    tempR[0] = currentParticle;
    tempR[1] = targetParticle;
    for (i = 0; i < vecLen; ++i)
      tempR[i + 2] = *vec[i];
    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> tempV;
    tempV.Initialize(vecLen + 2);
    G4int tempLen = 0;
    for (i = 0; i < vecLen + 2; ++i)
      tempV.SetElement(tempLen++, &tempR[i]);
    constantCrossSection = true;

    wgt = GenerateNBodyEvent(pseudoParticle[3].GetTotalEnergy() / MeV + pseudoParticle[4].GetTotalEnergy() / MeV,
                             constantCrossSection,
                             tempV,
                             tempLen);
    if (wgt > -.5) {
      theoreticalKinetic = 0.0;
      for (i = 0; i < tempLen; ++i) {
        pseudoParticle[6].Lorentz(*tempV[i], pseudoParticle[4]);
        theoreticalKinetic += pseudoParticle[6].GetKineticEnergy() / MeV;
      }
    }
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  }
  //
  //  Make sure, that the kinetic energies are correct
  //
  if (simulatedKinetic != 0.0) {
    wgt = (theoreticalKinetic) / simulatedKinetic;
    theoreticalKinetic = currentParticle.GetKineticEnergy() / MeV * wgt;
    simulatedKinetic = theoreticalKinetic;
    currentParticle.SetKineticEnergy(theoreticalKinetic * MeV);
    pp = currentParticle.GetTotalMomentum() / MeV;
    pp1 = currentParticle.GetMomentum().mag() / MeV;
    if (pp1 < 1.0e-6 * GeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      currentParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                  pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                  pp * std::cos(rthnve) * MeV);
    } else {
      currentParticle.SetMomentum(currentParticle.GetMomentum() * (pp / pp1));
    }
    theoreticalKinetic = targetParticle.GetKineticEnergy() / MeV * wgt;
    targetParticle.SetKineticEnergy(theoreticalKinetic * MeV);
    simulatedKinetic += theoreticalKinetic;
    pp = targetParticle.GetTotalMomentum() / MeV;
    pp1 = targetParticle.GetMomentum().mag() / MeV;
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    if (pp1 < 1.0e-6 * GeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      targetParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                 pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                 pp * std::cos(rthnve) * MeV);
    } else {
      targetParticle.SetMomentum(targetParticle.GetMomentum() * (pp / pp1));
    }
    for (i = 0; i < vecLen; ++i) {
      theoreticalKinetic = vec[i]->GetKineticEnergy() / MeV * wgt;
      simulatedKinetic += theoreticalKinetic;
      vec[i]->SetKineticEnergy(theoreticalKinetic * MeV);
      pp = vec[i]->GetTotalMomentum() / MeV;
      pp1 = vec[i]->GetMomentum().mag() / MeV;
      if (pp1 < 1.0e-6 * GeV) {
        rthnve = pi * G4UniformRand();
        phinve = twopi * G4UniformRand();
        vec[i]->SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                            pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                            pp * std::cos(rthnve) * MeV);
      } else
        vec[i]->SetMomentum(vec[i]->GetMomentum() * (pp / pp1));
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  Rotate(numberofFinalStateNucleons,
         pseudoParticle[3].GetMomentum(),
         modifiedOriginal,
         originalIncident,
         targetNucleus,
         currentParticle,
         targetParticle,
         vec,
         vecLen);
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  //
  // add black track particles
  // the total number of particles produced is restricted to 198
  // this may have influence on very high energies
  //
  if (atomicWeight >= 1.5) {
    // npnb is number of proton/neutron black track particles
    // ndta is the number of deuterons, tritons, and alphas produced
    // epnb is the kinetic energy available for proton/neutron black track particles
    // edta is the kinetic energy available for deuteron/triton/alpha particles
    //
    G4double epnb, edta;
    G4int npnb = 0;
    G4int ndta = 0;

    epnb = targetNucleus.GetPNBlackTrackEnergy();   // was enp1 in fortran code
    edta = targetNucleus.GetDTABlackTrackEnergy();  // was enp3 in fortran code
    const G4double pnCutOff = 0.001;
    const G4double dtaCutOff = 0.001;
    const G4double kineticMinimum = 1.e-6;
    const G4double kineticFactor = -0.010;
    G4double sprob = 0.0;  // sprob = probability of self-absorption in heavy molecules
    const G4double ekIncident = originalIncident->GetKineticEnergy() / GeV;
    if (ekIncident >= 5.0)
      sprob = std::min(1.0, 0.6 * std::log(ekIncident - 4.0));
    if (epnb >= pnCutOff) {
      npnb = G4Poisson((1.5 + 1.25 * numberofFinalStateNucleons) * epnb / (epnb + edta));
      if (numberofFinalStateNucleons + npnb > atomicWeight)
        npnb = G4int(atomicWeight + 0.00001 - numberofFinalStateNucleons);
      npnb = std::min(npnb, 127 - vecLen);
    }
    if (edta >= dtaCutOff) {
      ndta = G4Poisson((1.5 + 1.25 * numberofFinalStateNucleons) * edta / (epnb + edta));
      ndta = std::min(ndta, 127 - vecLen);
    }
    G4double spall = numberofFinalStateNucleons;
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);

    AddBlackTrackParticles(epnb,
                           npnb,
                           edta,
                           ndta,
                           sprob,
                           kineticMinimum,
                           kineticFactor,
                           modifiedOriginal,
                           spall,
                           targetNucleus,
                           vec,
                           vecLen);

    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  }
  //
  //  calculate time delay for nuclear reactions
  //
  if ((atomicWeight >= 1.5) && (atomicWeight <= 230.0) && (ekOriginal <= 0.2))
    currentParticle.SetTOF(1.0 - 500.0 * std::exp(-ekOriginal / 0.04) * std::log(G4UniformRand()));
  else
    currentParticle.SetTOF(1.0);
  return true;
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
}

void FullModelReactionDynamics::SuppressChargedPions(G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                                                     G4int &vecLen,
                                                     const G4ReactionProduct &modifiedOriginal,
                                                     G4ReactionProduct &currentParticle,
                                                     G4ReactionProduct &targetParticle,
                                                     const G4Nucleus &targetNucleus,
                                                     G4bool &incidentHasChanged,
                                                     G4bool &targetHasChanged) {
  // this code was originally in the fortran code TWOCLU
  //
  // suppress charged pions, for various reasons
  //
  const G4double atomicWeight = targetNucleus.GetN_asInt();
  const G4double atomicNumber = targetNucleus.GetZ_asInt();
  const G4double pOriginal = modifiedOriginal.GetTotalMomentum() / GeV;

  G4ParticleDefinition *aProton = G4Proton::Proton();
  G4ParticleDefinition *anAntiProton = G4AntiProton::AntiProton();
  G4ParticleDefinition *aNeutron = G4Neutron::Neutron();
  G4ParticleDefinition *anAntiNeutron = G4AntiNeutron::AntiNeutron();
  G4ParticleDefinition *aPiMinus = G4PionMinus::PionMinus();
  G4ParticleDefinition *aPiPlus = G4PionPlus::PionPlus();

  const G4bool antiTest =
      modifiedOriginal.GetDefinition() != anAntiProton && modifiedOriginal.GetDefinition() != anAntiNeutron;
  if (antiTest && (currentParticle.GetDefinition() == aPiPlus || currentParticle.GetDefinition() == aPiMinus) &&
      (G4UniformRand() <= (10.0 - pOriginal) / 6.0) && (G4UniformRand() <= atomicWeight / 300.0)) {
    if (G4UniformRand() > atomicNumber / atomicWeight)
      currentParticle.SetDefinitionAndUpdateE(aNeutron);
    else
      currentParticle.SetDefinitionAndUpdateE(aProton);
    incidentHasChanged = true;
  }
  for (G4int i = 0; i < vecLen; ++i) {
    if (antiTest && (vec[i]->GetDefinition() == aPiPlus || vec[i]->GetDefinition() == aPiMinus) &&
        (G4UniformRand() <= (10.0 - pOriginal) / 6.0) && (G4UniformRand() <= atomicWeight / 300.0)) {
      if (G4UniformRand() > atomicNumber / atomicWeight)
        vec[i]->SetDefinitionAndUpdateE(aNeutron);
      else
        vec[i]->SetDefinitionAndUpdateE(aProton);
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
}

G4bool FullModelReactionDynamics::TwoCluster(
    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
    G4int &vecLen,
    G4ReactionProduct &modifiedOriginal,      // Fermi motion & evap. effects included
    const G4HadProjectile *originalIncident,  // the original incident particle
    G4ReactionProduct &currentParticle,
    G4ReactionProduct &targetParticle,
    const G4Nucleus &targetNucleus,
    G4bool &incidentHasChanged,
    G4bool &targetHasChanged,
    G4bool leadFlag,
    G4ReactionProduct &leadingStrangeParticle) {
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  // derived from original FORTRAN code TWOCLU by H. Fesefeldt (11-Oct-1987)
  //
  //  Generation of X- and PT- values for incident, target, and all secondary particles
  //
  //  A simple two cluster model is used.
  //  This should be sufficient for low energy interactions.
  //

  // + debugging
  // raise(SIGSEGV);
  // - debugging

  G4int i;
  G4ParticleDefinition *aPiMinus = G4PionMinus::PionMinus();
  G4ParticleDefinition *aProton = G4Proton::Proton();
  G4ParticleDefinition *aNeutron = G4Neutron::Neutron();
  G4ParticleDefinition *aPiPlus = G4PionPlus::PionPlus();
  G4ParticleDefinition *aPiZero = G4PionZero::PionZero();
  const G4double protonMass = aProton->GetPDGMass() / MeV;
  const G4double ekOriginal = modifiedOriginal.GetKineticEnergy() / GeV;
  const G4double etOriginal = modifiedOriginal.GetTotalEnergy() / GeV;
  const G4double mOriginal = modifiedOriginal.GetMass() / GeV;
  const G4double pOriginal = modifiedOriginal.GetMomentum().mag() / GeV;
  G4double targetMass = targetParticle.GetDefinition()->GetPDGMass() / GeV;
  G4double centerofmassEnergy =
      std::sqrt(mOriginal * mOriginal + targetMass * targetMass + 2.0 * targetMass * etOriginal);  // GeV
  G4double currentMass = currentParticle.GetMass() / GeV;
  targetMass = targetParticle.GetMass() / GeV;

  if (currentMass == 0.0 && targetMass == 0.0) {
    G4double ek = currentParticle.GetKineticEnergy();
    G4ThreeVector m = currentParticle.GetMomentum();
    currentParticle = *vec[0];
    targetParticle = *vec[1];
    for (i = 0; i < (vecLen - 2); ++i)
      *vec[i] = *vec[i + 2];
    if (vecLen < 2) {
      for (G4int i = 0; i < vecLen; i++)
        delete vec[i];
      vecLen = 0;
      G4ExceptionDescription ed;
      ed << "Negative number of particles";
      G4Exception("FullModelReactionDynamics::TwoCluster", "had064", FatalException, ed);
    }
    delete vec[vecLen - 1];
    delete vec[vecLen - 2];
    vecLen -= 2;
    incidentHasChanged = true;
    targetHasChanged = true;
    currentParticle.SetKineticEnergy(ek);
    currentParticle.SetMomentum(m);
  }
  const G4double atomicWeight = targetNucleus.GetN_asInt();
  const G4double atomicNumber = targetNucleus.GetZ_asInt();
  //
  // particles have been distributed in forward and backward hemispheres
  // in center of mass system of the hadron nucleon interaction
  //
  // incident is always in forward hemisphere
  G4int forwardCount = 1;  // number of particles in forward hemisphere
  currentParticle.SetSide(1);
  G4double forwardMass = currentParticle.GetMass() / GeV;
  //G4double cMass = forwardMass;

  // target is always in backward hemisphere
  G4int backwardCount = 1;         // number of particles in backward hemisphere
  G4int backwardNucleonCount = 1;  // number of nucleons in backward hemisphere
  targetParticle.SetSide(-1);
  G4double backwardMass = targetParticle.GetMass() / GeV;
  //G4double bMass = backwardMass;

  for (i = 0; i < vecLen; ++i) {
    if (vec[i]->GetSide() < 0)
      vec[i]->SetSide(-1);  // added by JLC, 2Jul97
    // to take care of the case where vec has been preprocessed by GenerateXandPt
    // and some of them have been set to -2 or -3
    if (vec[i]->GetSide() == -1) {
      ++backwardCount;
      backwardMass += vec[i]->GetMass() / GeV;
    } else {
      ++forwardCount;
      forwardMass += vec[i]->GetMass() / GeV;
    }
  }
  //
  // nucleons and some pions from intranuclear cascade
  //
  G4double term1 = std::log(centerofmassEnergy * centerofmassEnergy);
  if (term1 < 0)
    term1 = 0.0001;  // making sure xtarg<0;
  const G4double afc = 0.312 + 0.2 * std::log(term1);
  G4double xtarg;
  if (centerofmassEnergy < 2.0 + G4UniformRand())  // added +2 below, JLC 4Jul97
    xtarg = afc * (std::pow(atomicWeight, 0.33) - 1.0) * (2 * backwardCount + vecLen + 2) / 2.0;
  else
    xtarg = afc * (std::pow(atomicWeight, 0.33) - 1.0) * (2 * backwardCount);
  if (xtarg <= 0.0)
    xtarg = 0.01;
  G4int nuclearExcitationCount = G4Poisson(xtarg);
  if (atomicWeight < 1.0001)
    nuclearExcitationCount = 0;
  G4int extraNucleonCount = 0;
  if (nuclearExcitationCount > 0) {
    G4int momentumBin = std::min(4, G4int(pOriginal / 3.0));
    const G4double nucsup[] = {1.0, 0.8, 0.6, 0.5, 0.4};
    //
    //  NOTE: in TWOCLU, these new particles were given negative codes
    //        here we use  NewlyAdded = true  instead
    //
    for (i = 0; i < nuclearExcitationCount; ++i) {
      G4ReactionProduct *pVec = new G4ReactionProduct();
      if (G4UniformRand() < nucsup[momentumBin])  // add proton or neutron
      {
        if (G4UniformRand() > 1.0 - atomicNumber / atomicWeight)
          // HPW: looks like a gheisha bug
          pVec->SetDefinition(aProton);
        else
          pVec->SetDefinition(aNeutron);
        ++backwardNucleonCount;
        ++extraNucleonCount;
        //extraNucleonMass += pVec->GetMass() / GeV;
      } else {  // add a pion
        G4double ran = G4UniformRand();
        if (ran < 0.3181)
          pVec->SetDefinition(aPiPlus);
        else if (ran < 0.6819)
          pVec->SetDefinition(aPiZero);
        else
          pVec->SetDefinition(aPiMinus);
      }
      pVec->SetSide(-2);  // backside particle
      pVec->SetNewlyAdded(true);
      vec.SetElement(vecLen++, pVec);
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  G4double eAvailable = centerofmassEnergy - (forwardMass + backwardMass);
  G4bool secondaryDeleted;
  G4double pMass;
  while (eAvailable <= 0.0)  // must eliminate a particle
  {
    secondaryDeleted = false;
    for (i = (vecLen - 1); i >= 0; --i) {
      if (vec[i]->GetSide() == 1 && vec[i]->GetMayBeKilled()) {
        pMass = vec[i]->GetMass() / GeV;
        for (G4int j = i; j < (vecLen - 1); ++j)
          *vec[j] = *vec[j + 1];  // shift up
        --forwardCount;
        //forwardEnergy += pMass;
        forwardMass -= pMass;
        secondaryDeleted = true;
        break;
      } else if (vec[i]->GetSide() == -1 && vec[i]->GetMayBeKilled()) {
        pMass = vec[i]->GetMass() / GeV;
        for (G4int j = i; j < (vecLen - 1); ++j)
          *vec[j] = *vec[j + 1];  // shift up
        --backwardCount;
        //backwardEnergy += pMass;
        backwardMass -= pMass;
        secondaryDeleted = true;
        break;
      }
    }  // breaks go down to here
    if (secondaryDeleted) {
      G4ReactionProduct *temp = vec[vecLen - 1];
      delete temp;
      --vecLen;
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    } else {
      if (vecLen == 0) {
        return false;  // all secondaries have been eliminated
      }
      if (targetParticle.GetSide() == -1) {
        pMass = targetParticle.GetMass() / GeV;
        targetParticle = *vec[0];
        for (G4int j = 0; j < (vecLen - 1); ++j)
          *vec[j] = *vec[j + 1];  // shift up
        --backwardCount;
        //backwardEnergy += pMass;
        backwardMass -= pMass;
        secondaryDeleted = true;
      } else if (targetParticle.GetSide() == 1) {
        pMass = targetParticle.GetMass() / GeV;
        targetParticle = *vec[0];
        for (G4int j = 0; j < (vecLen - 1); ++j)
          *vec[j] = *vec[j + 1];  // shift up
        --forwardCount;
        //forwardEnergy += pMass;
        forwardMass -= pMass;
        secondaryDeleted = true;
      }
      if (secondaryDeleted) {
        G4ReactionProduct *temp = vec[vecLen - 1];
        delete temp;
        --vecLen;
        // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
      } else {
        if (currentParticle.GetSide() == -1) {
          pMass = currentParticle.GetMass() / GeV;
          currentParticle = *vec[0];
          for (G4int j = 0; j < (vecLen - 1); ++j)
            *vec[j] = *vec[j + 1];  // shift up
          --backwardCount;
          //backwardEnergy += pMass;
          backwardMass -= pMass;
          secondaryDeleted = true;
        } else if (currentParticle.GetSide() == 1) {
          pMass = currentParticle.GetMass() / GeV;
          currentParticle = *vec[0];
          for (G4int j = 0; j < (vecLen - 1); ++j)
            *vec[j] = *vec[j + 1];  // shift up
          --forwardCount;           //This line can cause infinite loop
          //forwardEnergy += pMass;
          forwardMass -= pMass;
          secondaryDeleted = true;
        }
        if (secondaryDeleted) {
          G4ReactionProduct *temp = vec[vecLen - 1];
          delete temp;
          --vecLen;
          // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
        } else
          break;
      }
    }
    eAvailable = centerofmassEnergy - (forwardMass + backwardMass);
  }
  //
  // This is the start of the TwoCluster function
  //  Choose masses for the 3 clusters:
  //   forward cluster
  //   backward meson cluster
  //   backward nucleon cluster
  //
  G4double rmc = 0.0, rmd = 0.0;  // rme = 0.0;
  const G4double cpar[] = {0.6, 0.6, 0.35, 0.15, 0.10};
  const G4double gpar[] = {2.6, 2.6, 1.8, 1.30, 1.20};

  if (forwardCount == 0)
    return false;

  if (forwardCount == 1)
    rmc = forwardMass;
  else {
    G4int ntc = std::max(1, std::min(5, forwardCount)) - 1;  // check if offset by 1 @@
    rmc = forwardMass + std::pow(-std::log(1.0 - G4UniformRand()), cpar[ntc]) / gpar[ntc];
  }
  if (backwardCount == 1)
    rmd = backwardMass;
  else {
    G4int ntc = std::max(1, std::min(5, backwardCount)) - 1;  // check, if offfset by 1 @@
    rmd = backwardMass + std::pow(-std::log(1.0 - G4UniformRand()), cpar[ntc]) / gpar[ntc];
  }
  while (rmc + rmd > centerofmassEnergy) {
    if ((rmc <= forwardMass) && (rmd <= backwardMass)) {
      G4double temp = 0.999 * centerofmassEnergy / (rmc + rmd);
      rmc *= temp;
      rmd *= temp;
    } else {
      rmc = 0.1 * forwardMass + 0.9 * rmc;
      rmd = 0.1 * backwardMass + 0.9 * rmd;
    }
  }
  //
  //  Set beam, target of first interaction in centre of mass system
  //
  G4ReactionProduct pseudoParticle[8];
  for (i = 0; i < 8; ++i)
    pseudoParticle[i].SetZero();

  pseudoParticle[1].SetMass(mOriginal * GeV);
  pseudoParticle[1].SetTotalEnergy(etOriginal * GeV);
  pseudoParticle[1].SetMomentum(0.0, 0.0, pOriginal * GeV);

  pseudoParticle[2].SetMass(protonMass * MeV);
  pseudoParticle[2].SetTotalEnergy(protonMass * MeV);
  pseudoParticle[2].SetMomentum(0.0, 0.0, 0.0);
  //
  //  transform into centre of mass system
  //
  pseudoParticle[0] = pseudoParticle[1] + pseudoParticle[2];
  pseudoParticle[1].Lorentz(pseudoParticle[1], pseudoParticle[0]);
  pseudoParticle[2].Lorentz(pseudoParticle[2], pseudoParticle[0]);

  const G4double pfMin = 0.0001;
  G4double pf = (centerofmassEnergy * centerofmassEnergy + rmd * rmd - rmc * rmc);
  pf *= pf;
  pf -= 4 * centerofmassEnergy * centerofmassEnergy * rmd * rmd;
  pf = std::sqrt(std::max(pf, pfMin)) / (2.0 * centerofmassEnergy);
  //
  //  set final state masses and energies in centre of mass system
  //
  pseudoParticle[3].SetMass(rmc * GeV);
  pseudoParticle[3].SetTotalEnergy(std::sqrt(pf * pf + rmc * rmc) * GeV);

  pseudoParticle[4].SetMass(rmd * GeV);
  pseudoParticle[4].SetTotalEnergy(std::sqrt(pf * pf + rmd * rmd) * GeV);
  //
  // set |T| and |TMIN|
  //
  const G4double bMin = 0.01;
  const G4double b1 = 4.0;
  const G4double b2 = 1.6;
  G4double t = std::log(1.0 - G4UniformRand()) / std::max(bMin, b1 + b2 * std::log(pOriginal));
  G4double t1 = pseudoParticle[1].GetTotalEnergy() / GeV - pseudoParticle[3].GetTotalEnergy() / GeV;
  G4double pin = pseudoParticle[1].GetMomentum().mag() / GeV;
  G4double tacmin = t1 * t1 - (pin - pf) * (pin - pf);
  //
  // calculate (std::sin(teta/2.)^2 and std::cos(teta), set azimuth angle phi
  //
  const G4double smallValue = 1.0e-10;
  G4double dumnve = 4.0 * pin * pf;
  if (dumnve == 0.0)
    dumnve = smallValue;
  G4double ctet = std::max(-1.0, std::min(1.0, 1.0 + 2.0 * (t - tacmin) / dumnve));
  dumnve = std::max(0.0, 1.0 - ctet * ctet);
  G4double stet = std::sqrt(dumnve);
  G4double phi = G4UniformRand() * twopi;
  //
  // calculate final state momenta in centre of mass system
  //
  pseudoParticle[3].SetMomentum(pf * stet * std::sin(phi) * GeV, pf * stet * std::cos(phi) * GeV, pf * ctet * GeV);
  pseudoParticle[4].SetMomentum(pseudoParticle[3].GetMomentum() * (-1.0));
  //
  // simulate backward nucleon cluster in lab. system and transform in cms
  //
  G4double pp, pp1, rthnve, phinve;
  if (nuclearExcitationCount > 0) {
    const G4double ga = 1.2;
    G4double ekit1 = 0.04;
    G4double ekit2 = 0.6;
    if (ekOriginal <= 5.0) {
      ekit1 *= ekOriginal * ekOriginal / 25.0;
      ekit2 *= ekOriginal * ekOriginal / 25.0;
    }
    const G4double a = (1.0 - ga) / (std::pow(ekit2, (1.0 - ga)) - std::pow(ekit1, (1.0 - ga)));
    for (i = 0; i < vecLen; ++i) {
      if (vec[i]->GetSide() == -2) {
        G4double kineticE =
            std::pow((G4UniformRand() * (1.0 - ga) / a + std::pow(ekit1, (1.0 - ga))), (1.0 / (1.0 - ga)));
        vec[i]->SetKineticEnergy(kineticE * GeV);
        G4double vMass = vec[i]->GetMass() / MeV;
        G4double totalE = kineticE + vMass;
        pp = std::sqrt(std::abs(totalE * totalE - vMass * vMass));
        G4double cost = std::min(1.0, std::max(-1.0, std::log(2.23 * G4UniformRand() + 0.383) / 0.96));
        G4double sint = std::sqrt(std::max(0.0, (1.0 - cost * cost)));
        phi = twopi * G4UniformRand();
        vec[i]->SetMomentum(pp * sint * std::sin(phi) * MeV, pp * sint * std::cos(phi) * MeV, pp * cost * MeV);
        vec[i]->Lorentz(*vec[i], pseudoParticle[0]);
      }
    }
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  }
  //
  // fragmentation of forward cluster and backward meson cluster
  //
  currentParticle.SetMomentum(pseudoParticle[3].GetMomentum());
  currentParticle.SetTotalEnergy(pseudoParticle[3].GetTotalEnergy());

  targetParticle.SetMomentum(pseudoParticle[4].GetMomentum());
  targetParticle.SetTotalEnergy(pseudoParticle[4].GetTotalEnergy());

  pseudoParticle[5].SetMomentum(pseudoParticle[3].GetMomentum() * (-1.0));
  pseudoParticle[5].SetMass(pseudoParticle[3].GetMass());
  pseudoParticle[5].SetTotalEnergy(pseudoParticle[3].GetTotalEnergy());

  pseudoParticle[6].SetMomentum(pseudoParticle[4].GetMomentum() * (-1.0));
  pseudoParticle[6].SetMass(pseudoParticle[4].GetMass());
  pseudoParticle[6].SetTotalEnergy(pseudoParticle[4].GetTotalEnergy());

  G4double wgt;
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  if (forwardCount > 1)  // tempV will contain the forward particles
  {
    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> tempV;
    tempV.Initialize(forwardCount);
    G4bool constantCrossSection = true;
    G4int tempLen = 0;
    if (currentParticle.GetSide() == 1)
      tempV.SetElement(tempLen++, &currentParticle);
    if (targetParticle.GetSide() == 1)
      tempV.SetElement(tempLen++, &targetParticle);
    for (i = 0; i < vecLen; ++i) {
      if (vec[i]->GetSide() == 1) {
        if (tempLen < 18)
          tempV.SetElement(tempLen++, vec[i]);
        else {
          vec[i]->SetSide(-1);
          continue;
        }
      }
    }
    if (tempLen >= 2) {
      GenerateNBodyEvent(pseudoParticle[3].GetMass() / MeV, constantCrossSection, tempV, tempLen);
      if (currentParticle.GetSide() == 1)
        currentParticle.Lorentz(currentParticle, pseudoParticle[5]);
      if (targetParticle.GetSide() == 1)
        targetParticle.Lorentz(targetParticle, pseudoParticle[5]);
      for (i = 0; i < vecLen; ++i) {
        if (vec[i]->GetSide() == 1)
          vec[i]->Lorentz(*vec[i], pseudoParticle[5]);
      }
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  if (backwardCount > 1)  //  tempV will contain the backward particles,
  {                       //  but not those created from the intranuclear cascade
    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> tempV;
    tempV.Initialize(backwardCount);
    G4bool constantCrossSection = true;
    G4int tempLen = 0;
    if (currentParticle.GetSide() == -1)
      tempV.SetElement(tempLen++, &currentParticle);
    if (targetParticle.GetSide() == -1)
      tempV.SetElement(tempLen++, &targetParticle);
    for (i = 0; i < vecLen; ++i) {
      if (vec[i]->GetSide() == -1) {
        if (tempLen < 18)
          tempV.SetElement(tempLen++, vec[i]);
        else {
          vec[i]->SetSide(-2);
          vec[i]->SetKineticEnergy(0.0);
          vec[i]->SetMomentum(0.0, 0.0, 0.0);
          continue;
        }
      }
    }
    if (tempLen >= 2) {
      GenerateNBodyEvent(pseudoParticle[4].GetMass() / MeV, constantCrossSection, tempV, tempLen);
      if (currentParticle.GetSide() == -1)
        currentParticle.Lorentz(currentParticle, pseudoParticle[6]);
      if (targetParticle.GetSide() == -1)
        targetParticle.Lorentz(targetParticle, pseudoParticle[6]);
      for (i = 0; i < vecLen; ++i) {
        if (vec[i]->GetSide() == -1)
          vec[i]->Lorentz(*vec[i], pseudoParticle[6]);
      }
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  //
  // Lorentz transformation in lab system
  //
  G4int numberofFinalStateNucleons =
      currentParticle.GetDefinition()->GetBaryonNumber() + targetParticle.GetDefinition()->GetBaryonNumber();
  currentParticle.Lorentz(currentParticle, pseudoParticle[2]);
  targetParticle.Lorentz(targetParticle, pseudoParticle[2]);

  for (i = 0; i < vecLen; ++i) {
    numberofFinalStateNucleons += vec[i]->GetDefinition()->GetBaryonNumber();
    vec[i]->Lorentz(*vec[i], pseudoParticle[2]);
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  numberofFinalStateNucleons = std::max(1, numberofFinalStateNucleons);
  //
  // sometimes the leading strange particle is lost, set it back
  //
  G4bool dum = true;
  if (leadFlag) {
    // leadFlag will be true
    //  iff original particle is at least as heavy as K+ and not a proton or neutron AND
    //   if
    //    incident particle is at least as heavy as K+ and it is not a proton or neutron
    //     leadFlag is set to the incident particle
    //   or
    //    target particle is at least as heavy as K+ and it is not a proton or neutron
    //     leadFlag is set to the target particle
    //
    if (currentParticle.GetDefinition() == leadingStrangeParticle.GetDefinition())
      dum = false;
    else if (targetParticle.GetDefinition() == leadingStrangeParticle.GetDefinition())
      dum = false;
    else {
      for (i = 0; i < vecLen; ++i) {
        if (vec[i]->GetDefinition() == leadingStrangeParticle.GetDefinition()) {
          dum = false;
          break;
        }
      }
    }
    if (dum) {
      G4double leadMass = leadingStrangeParticle.GetMass() / MeV;
      G4double ekin;
      if (((leadMass < protonMass) && (targetParticle.GetMass() / MeV < protonMass)) ||
          ((leadMass >= protonMass) && (targetParticle.GetMass() / MeV >= protonMass))) {
        ekin = targetParticle.GetKineticEnergy() / GeV;
        pp1 = targetParticle.GetMomentum().mag() / MeV;  // old momentum
        targetParticle.SetDefinition(leadingStrangeParticle.GetDefinition());
        targetParticle.SetKineticEnergy(ekin * GeV);
        pp = targetParticle.GetTotalMomentum() / MeV;  // new momentum
        if (pp1 < 1.0e-3) {
          rthnve = pi * G4UniformRand();
          phinve = twopi * G4UniformRand();
          targetParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                     pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                     pp * std::cos(rthnve) * MeV);
        } else
          targetParticle.SetMomentum(targetParticle.GetMomentum() * (pp / pp1));

        targetHasChanged = true;
      } else {
        ekin = currentParticle.GetKineticEnergy() / GeV;
        pp1 = currentParticle.GetMomentum().mag() / MeV;
        currentParticle.SetDefinition(leadingStrangeParticle.GetDefinition());
        currentParticle.SetKineticEnergy(ekin * GeV);
        pp = currentParticle.GetTotalMomentum() / MeV;
        if (pp1 < 1.0e-3) {
          rthnve = pi * G4UniformRand();
          phinve = twopi * G4UniformRand();
          currentParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                      pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                      pp * std::cos(rthnve) * MeV);
        } else
          currentParticle.SetMomentum(currentParticle.GetMomentum() * (pp / pp1));

        incidentHasChanged = true;
      }
    }
  }  // end of if( leadFlag )
  //
  //  for various reasons, the energy balance is not sufficient,
  //  check that,  energy balance, angle of final system, etc.
  //
  pseudoParticle[4].SetMass(mOriginal * GeV);
  pseudoParticle[4].SetTotalEnergy(etOriginal * GeV);
  pseudoParticle[4].SetMomentum(0.0, 0.0, pOriginal * GeV);

  const G4ParticleDefinition *aOrgDef = modifiedOriginal.GetDefinition();
  G4int diff = 0;
  if (aOrgDef == G4Proton::Proton() || aOrgDef == G4Neutron::Neutron())
    diff = 1;
  if (numberofFinalStateNucleons == 1)
    diff = 0;
  pseudoParticle[5].SetMomentum(0.0, 0.0, 0.0);
  pseudoParticle[5].SetMass(protonMass * (numberofFinalStateNucleons - diff) * MeV);
  pseudoParticle[5].SetTotalEnergy(protonMass * (numberofFinalStateNucleons - diff) * MeV);

  //    G4double ekin0 = pseudoParticle[4].GetKineticEnergy()/GeV;
  G4double theoreticalKinetic = pseudoParticle[4].GetTotalEnergy() / GeV + pseudoParticle[5].GetTotalEnergy() / GeV;

  pseudoParticle[6] = pseudoParticle[4] + pseudoParticle[5];
  pseudoParticle[4].Lorentz(pseudoParticle[4], pseudoParticle[6]);
  pseudoParticle[5].Lorentz(pseudoParticle[5], pseudoParticle[6]);

  if (vecLen < 16) {
    G4ReactionProduct tempR[130];
    //G4ReactionProduct *tempR = new G4ReactionProduct [vecLen+2];
    tempR[0] = currentParticle;
    tempR[1] = targetParticle;
    for (i = 0; i < vecLen; ++i)
      tempR[i + 2] = *vec[i];

    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> tempV;
    tempV.Initialize(vecLen + 2);
    G4bool constantCrossSection = true;
    G4int tempLen = 0;
    for (i = 0; i < vecLen + 2; ++i)
      tempV.SetElement(tempLen++, &tempR[i]);

    if (tempLen >= 2) {
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
      GenerateNBodyEvent(pseudoParticle[4].GetTotalEnergy() / MeV + pseudoParticle[5].GetTotalEnergy() / MeV,
                         constantCrossSection,
                         tempV,
                         tempLen);
      theoreticalKinetic = 0.0;
      for (i = 0; i < vecLen + 2; ++i) {
        pseudoParticle[7].SetMomentum(tempV[i]->GetMomentum());
        pseudoParticle[7].SetMass(tempV[i]->GetMass());
        pseudoParticle[7].SetTotalEnergy(tempV[i]->GetTotalEnergy());
        pseudoParticle[7].Lorentz(pseudoParticle[7], pseudoParticle[5]);
        theoreticalKinetic += pseudoParticle[7].GetKineticEnergy() / GeV;
      }
    }
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    //delete [] tempR;
  } else {
    theoreticalKinetic -= (currentParticle.GetMass() / GeV + targetParticle.GetMass() / GeV);
    for (i = 0; i < vecLen; ++i)
      theoreticalKinetic -= vec[i]->GetMass() / GeV;
  }
  G4double simulatedKinetic = currentParticle.GetKineticEnergy() / GeV + targetParticle.GetKineticEnergy() / GeV;
  for (i = 0; i < vecLen; ++i)
    simulatedKinetic += vec[i]->GetKineticEnergy() / GeV;
  //
  // make sure that kinetic energies are correct
  // the backward nucleon cluster is not produced within proper kinematics!!!
  //

  if (simulatedKinetic != 0.0) {
    wgt = (theoreticalKinetic) / simulatedKinetic;
    currentParticle.SetKineticEnergy(wgt * currentParticle.GetKineticEnergy());
    pp = currentParticle.GetTotalMomentum() / MeV;
    pp1 = currentParticle.GetMomentum().mag() / MeV;
    if (pp1 < 0.001 * MeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      currentParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                  pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                  pp * std::cos(rthnve) * MeV);
    } else
      currentParticle.SetMomentum(currentParticle.GetMomentum() * (pp / pp1));

    targetParticle.SetKineticEnergy(wgt * targetParticle.GetKineticEnergy());
    pp = targetParticle.GetTotalMomentum() / MeV;
    pp1 = targetParticle.GetMomentum().mag() / MeV;
    if (pp1 < 0.001 * MeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      targetParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                 pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                 pp * std::cos(rthnve) * MeV);
    } else
      targetParticle.SetMomentum(targetParticle.GetMomentum() * (pp / pp1));

    for (i = 0; i < vecLen; ++i) {
      vec[i]->SetKineticEnergy(wgt * vec[i]->GetKineticEnergy());
      pp = vec[i]->GetTotalMomentum() / MeV;
      pp1 = vec[i]->GetMomentum().mag() / MeV;
      if (pp1 < 0.001) {
        rthnve = pi * G4UniformRand();
        phinve = twopi * G4UniformRand();
        vec[i]->SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                            pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                            pp * std::cos(rthnve) * MeV);
      } else
        vec[i]->SetMomentum(vec[i]->GetMomentum() * (pp / pp1));
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  Rotate(numberofFinalStateNucleons,
         pseudoParticle[4].GetMomentum(),
         modifiedOriginal,
         originalIncident,
         targetNucleus,
         currentParticle,
         targetParticle,
         vec,
         vecLen);
  //
  //  add black track particles
  //  the total number of particles produced is restricted to 198
  //  this may have influence on very high energies
  //
  if (atomicWeight >= 1.5) {
    // npnb is number of proton/neutron black track particles
    // ndta is the number of deuterons, tritons, and alphas produced
    // epnb is the kinetic energy available for proton/neutron black track particles
    // edta is the kinetic energy available for deuteron/triton/alpha particles
    //
    G4double epnb, edta;
    G4int npnb = 0;
    G4int ndta = 0;

    epnb = targetNucleus.GetPNBlackTrackEnergy();   // was enp1 in fortran code
    edta = targetNucleus.GetDTABlackTrackEnergy();  // was enp3 in fortran code
    const G4double pnCutOff = 0.001;                // GeV
    const G4double dtaCutOff = 0.001;               // GeV
    const G4double kineticMinimum = 1.e-6;
    const G4double kineticFactor = -0.005;

    G4double sprob = 0.0;  // sprob = probability of self-absorption in heavy molecules
    const G4double ekIncident = originalIncident->GetKineticEnergy() / GeV;
    if (ekIncident >= 5.0)
      sprob = std::min(1.0, 0.6 * std::log(ekIncident - 4.0));

    if (epnb >= pnCutOff) {
      npnb = G4Poisson((1.5 + 1.25 * numberofFinalStateNucleons) * epnb / (epnb + edta));
      if (numberofFinalStateNucleons + npnb > atomicWeight)
        npnb = G4int(atomicWeight - numberofFinalStateNucleons);
      npnb = std::min(npnb, 127 - vecLen);
    }
    if (edta >= dtaCutOff) {
      ndta = G4Poisson((1.5 + 1.25 * numberofFinalStateNucleons) * edta / (epnb + edta));
      ndta = std::min(ndta, 127 - vecLen);
    }
    G4double spall = numberofFinalStateNucleons;
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);

    AddBlackTrackParticles(epnb,
                           npnb,
                           edta,
                           ndta,
                           sprob,
                           kineticMinimum,
                           kineticFactor,
                           modifiedOriginal,
                           spall,
                           targetNucleus,
                           vec,
                           vecLen);

    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  }
  //if( centerofmassEnergy <= (4.0+G4UniformRand()) )
  //  MomentumCheck( modifiedOriginal, currentParticle, targetParticle, vec, vecLen );
  //
  //  calculate time delay for nuclear reactions
  //
  if ((atomicWeight >= 1.5) && (atomicWeight <= 230.0) && (ekOriginal <= 0.2))
    currentParticle.SetTOF(1.0 - 500.0 * std::exp(-ekOriginal / 0.04) * std::log(G4UniformRand()));
  else
    currentParticle.SetTOF(1.0);

  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  return true;
}

void FullModelReactionDynamics::TwoBody(G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                                        G4int &vecLen,
                                        G4ReactionProduct &modifiedOriginal,
                                        const G4DynamicParticle * /*originalTarget*/,
                                        G4ReactionProduct &currentParticle,
                                        G4ReactionProduct &targetParticle,
                                        const G4Nucleus &targetNucleus,
                                        G4bool & /* targetHasChanged*/) {
  //    G4cout<<"TwoBody called"<<G4endl;
  //
  // derived from original FORTRAN code TWOB by H. Fesefeldt (15-Sep-1987)
  //
  // Generation of momenta for elastic and quasi-elastic 2 body reactions
  //
  // The simple formula ds/d|t| = s0* std::exp(-b*|t|) is used.
  // The b values are parametrizations from experimental data.
  // Not available values are taken from those of similar reactions.
  //
  G4ParticleDefinition *aPiMinus = G4PionMinus::PionMinus();
  G4ParticleDefinition *aPiPlus = G4PionPlus::PionPlus();
  G4ParticleDefinition *aPiZero = G4PionZero::PionZero();
  G4ParticleDefinition *aKaonPlus = G4KaonPlus::KaonPlus();
  G4ParticleDefinition *aKaonMinus = G4KaonMinus::KaonMinus();
  G4ParticleDefinition *aKaonZeroS = G4KaonZeroShort::KaonZeroShort();
  G4ParticleDefinition *aKaonZeroL = G4KaonZeroLong::KaonZeroLong();

  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  static const G4double expxu = 82.;     // upper bound for arg. of exp
  static const G4double expxl = -expxu;  // lower bound for arg. of exp

  const G4double ekOriginal = modifiedOriginal.GetKineticEnergy() / GeV;
  const G4double pOriginal = modifiedOriginal.GetMomentum().mag() / GeV;
  G4double currentMass = currentParticle.GetMass() / GeV;
  G4double targetMass = targetParticle.GetMass() / GeV;
  const G4double atomicWeight = targetNucleus.GetN_asInt();
  //    G4cout<<"Atomic weight is found to be: "<<atomicWeight<<G4endl;
  G4double etCurrent = currentParticle.GetTotalEnergy() / GeV;
  G4double pCurrent = currentParticle.GetTotalMomentum() / GeV;

  G4double cmEnergy =
      std::sqrt(currentMass * currentMass + targetMass * targetMass + 2.0 * targetMass * etCurrent);  // in GeV

  //if( (pOriginal < 0.1) ||
  //    (centerofmassEnergy < 0.01) ) // 2-body scattering not possible
  // Continue with original particle, but spend the nuclear evaporation energy
  //  targetParticle.SetMass( 0.0 );  // flag that the target doesn't exist
  //else                           // Two-body scattering is possible

  if ((pCurrent < 0.1) || (cmEnergy < 0.01))  // 2-body scattering not possible
  {
    targetParticle.SetMass(0.0);  // flag that the target particle doesn't exist
  } else {
    G4double pf = cmEnergy * cmEnergy + targetMass * targetMass - currentMass * currentMass;
    pf = pf * pf - 4 * cmEnergy * cmEnergy * targetMass * targetMass;
    //      G4cout << "pf: " << pf<< G4endl;

    if (pf <= 0.)  // 0.001 )
    {
      for (G4int i = 0; i < vecLen; i++)
        delete vec[i];
      vecLen = 0;
      throw G4HadronicException(__FILE__, __LINE__, "FullModelReactionDynamics::TwoBody: pf is too small ");
    }

    pf = std::sqrt(pf) / (2.0 * cmEnergy);
    //
    // Set beam and target in centre of mass system
    //
    G4ReactionProduct pseudoParticle[3];
    pseudoParticle[0].SetMass(currentMass * GeV);
    pseudoParticle[0].SetTotalEnergy(etCurrent * GeV);
    pseudoParticle[0].SetMomentum(0.0, 0.0, pCurrent * GeV);

    pseudoParticle[1].SetMomentum(0.0, 0.0, 0.0);
    pseudoParticle[1].SetMass(targetMass * GeV);
    pseudoParticle[1].SetKineticEnergy(0.0);
    //
    // Transform into centre of mass system
    //
    pseudoParticle[2] = pseudoParticle[0] + pseudoParticle[1];
    pseudoParticle[0].Lorentz(pseudoParticle[0], pseudoParticle[2]);
    pseudoParticle[1].Lorentz(pseudoParticle[1], pseudoParticle[2]);
    //
    // Set final state masses and energies in centre of mass system
    //
    currentParticle.SetTotalEnergy(std::sqrt(pf * pf + currentMass * currentMass) * GeV);
    targetParticle.SetTotalEnergy(std::sqrt(pf * pf + targetMass * targetMass) * GeV);
    //
    // Set |t| and |tmin|
    //
    const G4double cb = 0.01;
    const G4double b1 = 4.225;
    const G4double b2 = 1.795;
    //
    // Calculate slope b for elastic scattering on proton/neutron
    //
    G4double b = std::max(cb, b1 + b2 * std::log(pOriginal));
    G4double btrang = b * 4.0 * pf * pseudoParticle[0].GetMomentum().mag() / GeV;

    G4double exindt = -1.0;
    exindt += std::exp(std::max(-btrang, expxl));
    //
    // Calculate sqr(std::sin(teta/2.) and std::cos(teta), set azimuth angle phi
    //
    G4double ctet = 1.0 + 2 * std::log(1.0 + G4UniformRand() * exindt) / btrang;
    if (std::fabs(ctet) > 1.0)
      ctet > 0.0 ? ctet = 1.0 : ctet = -1.0;
    G4double stet = std::sqrt((1.0 - ctet) * (1.0 + ctet));
    G4double phi = twopi * G4UniformRand();
    //
    // Calculate final state momenta in centre of mass system
    //
    if (targetParticle.GetDefinition() == aKaonMinus || targetParticle.GetDefinition() == aKaonZeroL ||
        targetParticle.GetDefinition() == aKaonZeroS || targetParticle.GetDefinition() == aKaonPlus ||
        targetParticle.GetDefinition() == aPiMinus || targetParticle.GetDefinition() == aPiZero ||
        targetParticle.GetDefinition() == aPiPlus) {
      currentParticle.SetMomentum(-pf * stet * std::sin(phi) * GeV, -pf * stet * std::cos(phi) * GeV, -pf * ctet * GeV);
    } else {
      currentParticle.SetMomentum(pf * stet * std::sin(phi) * GeV, pf * stet * std::cos(phi) * GeV, pf * ctet * GeV);
    }
    targetParticle.SetMomentum(currentParticle.GetMomentum() * (-1.0));
    //
    // Transform into lab system
    //
    currentParticle.Lorentz(currentParticle, pseudoParticle[1]);
    targetParticle.Lorentz(targetParticle, pseudoParticle[1]);

    /*
      G4cout<<"Check 1"<<G4endl;
      G4cout<<"target E_kin: "<<targetParticle.GetKineticEnergy()<<G4endl;
      G4cout<<"target mass: "<<targetParticle.GetMass()<<G4endl;
      G4cout<<"current E_kin: "<<currentParticle.GetKineticEnergy()<<G4endl;
      G4cout<<"current mass: "<<currentParticle.GetMass()<<G4endl;
      */

    Defs1(modifiedOriginal, currentParticle, targetParticle, vec, vecLen);

    G4double pp, pp1, ekin;
    if (atomicWeight >= 1.5) {
      const G4double cfa = 0.025 * ((atomicWeight - 1.) / 120.) * std::exp(-(atomicWeight - 1.) / 120.);
      pp1 = currentParticle.GetMomentum().mag() / MeV;
      if (pp1 >= 1.0) {
        ekin = currentParticle.GetKineticEnergy() / MeV - cfa * (1.0 + 0.5 * normal()) * GeV;
        ekin = std::max(0.0001 * GeV, ekin);
        currentParticle.SetKineticEnergy(ekin * MeV);
        pp = currentParticle.GetTotalMomentum() / MeV;
        currentParticle.SetMomentum(currentParticle.GetMomentum() * (pp / pp1));
      }
      pp1 = targetParticle.GetMomentum().mag() / MeV;
      if (pp1 >= 1.0) {
        ekin = targetParticle.GetKineticEnergy() / MeV - cfa * (1.0 + normal() / 2.) * GeV;
        ekin = std::max(0.0001 * GeV, ekin);
        targetParticle.SetKineticEnergy(ekin * MeV);
        pp = targetParticle.GetTotalMomentum() / MeV;
        targetParticle.SetMomentum(targetParticle.GetMomentum() * (pp / pp1));
      }
    }
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  if (atomicWeight >= 1.5) {
    // Add black track particles
    //  the procedure is somewhat different than in TwoCluster and GenerateXandPt.
    //  The reason is that we have to also simulate the nuclear reactions
    //  at low energies like a(h,p)b, a(h,p p)b, a(h,n)b etc.
    //
    // npnb is number of proton/neutron black track particles
    // ndta is the number of deuterons, tritons, and alphas produced
    // epnb is the kinetic energy available for proton/neutron black track particles
    // edta is the kinetic energy available for deuteron/triton/alpha particles
    //
    G4double epnb, edta;
    G4int npnb = 0, ndta = 0;

    epnb = targetNucleus.GetPNBlackTrackEnergy();   // was enp1 in fortran code
    edta = targetNucleus.GetDTABlackTrackEnergy();  // was enp3 in fortran code
    const G4double pnCutOff = 0.0001;               // GeV
    const G4double dtaCutOff = 0.0001;              // GeV
    const G4double kineticMinimum = 0.0001;
    const G4double kineticFactor = -0.010;
    G4double sprob = 0.0;  // sprob = probability of self-absorption in heavy molecules
    if (epnb >= pnCutOff) {
      npnb = G4Poisson(epnb / 0.02);
      /*
	G4cout<<"A couple of G4Poisson numbers:"<<G4endl;
	for (int n=0;n!=10;n++) G4cout<<G4Poisson(epnb/0.02)<<", ";
	G4cout<<G4endl;
	*/
      if (npnb > atomicWeight)
        npnb = G4int(atomicWeight);
      if ((epnb > pnCutOff) && (npnb <= 0))
        npnb = 1;
      npnb = std::min(npnb, 127 - vecLen);
    }
    if (edta >= dtaCutOff) {
      ndta = G4int(2.0 * std::log(atomicWeight));
      ndta = std::min(ndta, 127 - vecLen);
    }
    G4double spall = 0.0;
    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);

    /*
      G4cout<<"Check 2"<<G4endl;
      G4cout<<"target E_kin: "<<targetParticle.GetKineticEnergy()<<G4endl;
      G4cout<<"target mass: "<<targetParticle.GetMass()<<G4endl;
      G4cout<<"current E_kin: "<<currentParticle.GetKineticEnergy()<<G4endl;
      G4cout<<"current mass: "<<currentParticle.GetMass()<<G4endl;

      G4cout<<"------------------------------------------------------------------------"<<G4endl;
      G4cout<<"Atomic weight: "<<atomicWeight<<G4endl;
      G4cout<<"number of proton/neutron black track particles: "<<npnb<<G4endl;
      G4cout<<"number of deuterons, tritons, and alphas produced: "<<ndta <<G4endl;
      G4cout<<"kinetic energy available for proton/neutron black track particles: "<<epnb/GeV<<" GeV" <<G4endl;
      G4cout<<"kinetic energy available for deuteron/triton/alpha particles: "<<edta/GeV <<" GeV"<<G4endl;
      G4cout<<"------------------------------------------------------------------------"<<G4endl;
      */

    AddBlackTrackParticles(epnb,
                           npnb,
                           edta,
                           ndta,
                           sprob,
                           kineticMinimum,
                           kineticFactor,
                           modifiedOriginal,
                           spall,
                           targetNucleus,
                           vec,
                           vecLen);

    // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  }
  //
  //  calculate time delay for nuclear reactions
  //
  if ((atomicWeight >= 1.5) && (atomicWeight <= 230.0) && (ekOriginal <= 0.2))
    currentParticle.SetTOF(1.0 - 500.0 * std::exp(-ekOriginal / 0.04) * std::log(G4UniformRand()));
  else
    currentParticle.SetTOF(1.0);
  return;
}

G4double FullModelReactionDynamics::GenerateNBodyEvent(const G4double totalEnergy,  // MeV
                                                       const G4bool constantCrossSection,
                                                       G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                                                       G4int &vecLen) {
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  // derived from original FORTRAN code PHASP by H. Fesefeldt (02-Dec-1986)
  // Returns the weight of the event
  //
  G4int i;
  const G4double expxu = 82.;     // upper bound for arg. of exp
  const G4double expxl = -expxu;  // lower bound for arg. of exp
  if (vecLen < 2) {
    G4cerr << "*** Error in FullModelReactionDynamics::GenerateNBodyEvent" << G4endl;
    G4cerr << "    number of particles < 2" << G4endl;
    G4cerr << "totalEnergy = " << totalEnergy << "MeV, vecLen = " << vecLen << G4endl;
    return -1.0;
  }
  G4double mass[18];    // mass of each particle
  G4double energy[18];  // total energy of each particle
  G4double pcm[3][18];  // pcm is an array with 3 rows and vecLen columns
  G4double totalMass = 0.0;
  //G4double extraMass = 0;
  G4double sm[18];

  for (i = 0; i < vecLen; ++i) {
    mass[i] = vec[i]->GetMass() / GeV;
    //if (vec[i]->GetSide() == -2)
    //  extraMass += vec[i]->GetMass() / GeV;
    vec[i]->SetMomentum(0.0, 0.0, 0.0);
    pcm[0][i] = 0.0;      // x-momentum of i-th particle
    pcm[1][i] = 0.0;      // y-momentum of i-th particle
    pcm[2][i] = 0.0;      // z-momentum of i-th particle
    energy[i] = mass[i];  // total energy of i-th particle
    totalMass += mass[i];
    sm[i] = totalMass;
  }
  G4double totalE = totalEnergy / GeV;
  if (totalMass > totalE) {
    return -1.0;
  }
  G4double kineticEnergy = totalE - totalMass;
  G4double emm[18];
  //G4double *emm = new G4double [vecLen];
  emm[0] = mass[0];
  emm[vecLen - 1] = totalE;
  if (vecLen > 2)  // the random numbers are sorted
  {
    G4double ran[18];
    for (i = 0; i < vecLen; ++i)
      ran[i] = G4UniformRand();
    for (i = 0; i < vecLen - 2; ++i) {
      for (G4int j = vecLen - 2; j > i; --j) {
        if (ran[i] > ran[j]) {
          G4double temp = ran[i];
          ran[i] = ran[j];
          ran[j] = temp;
        }
      }
    }
    for (i = 1; i < vecLen - 1; ++i)
      emm[i] = ran[i - 1] * kineticEnergy + sm[i];
  }
  //   Weight is the sum of logarithms of terms instead of the product of terms
  G4bool lzero = true;
  G4double wtmax = 0.0;
  if (constantCrossSection)  // this is KGENEV=1 in PHASP
  {
    G4double emmax = kineticEnergy + mass[0];
    G4double emmin = 0.0;
    for (i = 1; i < vecLen; ++i) {
      emmin += mass[i - 1];
      emmax += mass[i];
      G4double wtfc = 0.0;
      if (emmax * emmax > 0.0) {
        G4double arg = emmax * emmax +
                       (emmin * emmin - mass[i] * mass[i]) * (emmin * emmin - mass[i] * mass[i]) / (emmax * emmax) -
                       2.0 * (emmin * emmin + mass[i] * mass[i]);
        if (arg > 0.0)
          wtfc = 0.5 * std::sqrt(arg);
      }
      if (wtfc == 0.0) {
        lzero = false;
        break;
      }
      wtmax += std::log(wtfc);
    }
    if (lzero)
      wtmax = -wtmax;
    else
      wtmax = expxu;
  } else {
    //   ffq(n) = pi*(2*pi)^(n-2)/(n-2)!
    const G4double ffq[18] = {0.,
                              3.141592,
                              19.73921,
                              62.01255,
                              129.8788,
                              204.0131,
                              256.3704,
                              268.4705,
                              240.9780,
                              189.2637,
                              132.1308,
                              83.0202,
                              47.4210,
                              24.8295,
                              12.0006,
                              5.3858,
                              2.2560,
                              0.8859};
    wtmax = std::log(std::pow(kineticEnergy, vecLen - 2) * ffq[vecLen - 1] / totalE);
  }
  lzero = true;
  G4double pd[50];
  //G4double *pd = new G4double [vecLen-1];
  for (i = 0; i < vecLen - 1; ++i) {
    pd[i] = 0.0;
    if (emm[i + 1] * emm[i + 1] > 0.0) {
      G4double arg = emm[i + 1] * emm[i + 1] +
                     (emm[i] * emm[i] - mass[i + 1] * mass[i + 1]) * (emm[i] * emm[i] - mass[i + 1] * mass[i + 1]) /
                         (emm[i + 1] * emm[i + 1]) -
                     2.0 * (emm[i] * emm[i] + mass[i + 1] * mass[i + 1]);
      if (arg > 0.0)
        pd[i] = 0.5 * std::sqrt(arg);
    }
    if (pd[i] <= 0.0)  //  changed from  ==  on 02 April 98
      lzero = false;
    else
      wtmax += std::log(pd[i]);
  }
  G4double weight = 0.0;  // weight is returned by GenerateNBodyEvent
  if (lzero)
    weight = std::exp(std::max(std::min(wtmax, expxu), expxl));

  G4double bang, cb, sb, s0, s1, s2, c, s, esys, a, b, gama, beta;
  pcm[0][0] = 0.0;
  pcm[1][0] = pd[0];
  pcm[2][0] = 0.0;
  for (i = 1; i < vecLen; ++i) {
    pcm[0][i] = 0.0;
    pcm[1][i] = -pd[i - 1];
    pcm[2][i] = 0.0;
    bang = twopi * G4UniformRand();
    cb = std::cos(bang);
    sb = std::sin(bang);
    c = 2.0 * G4UniformRand() - 1.0;
    s = std::sqrt(std::fabs(1.0 - c * c));
    if (i < vecLen - 1) {
      esys = std::sqrt(pd[i] * pd[i] + emm[i] * emm[i]);
      beta = pd[i] / esys;
      gama = esys / emm[i];
      for (G4int j = 0; j <= i; ++j) {
        s0 = pcm[0][j];
        s1 = pcm[1][j];
        s2 = pcm[2][j];
        energy[j] = std::sqrt(s0 * s0 + s1 * s1 + s2 * s2 + mass[j] * mass[j]);
        a = s0 * c - s1 * s;  //  rotation
        pcm[1][j] = s0 * s + s1 * c;
        b = pcm[2][j];
        pcm[0][j] = a * cb - b * sb;
        pcm[2][j] = a * sb + b * cb;
        pcm[1][j] = gama * (pcm[1][j] + beta * energy[j]);
      }
    } else {
      for (G4int j = 0; j <= i; ++j) {
        s0 = pcm[0][j];
        s1 = pcm[1][j];
        s2 = pcm[2][j];
        energy[j] = std::sqrt(s0 * s0 + s1 * s1 + s2 * s2 + mass[j] * mass[j]);
        a = s0 * c - s1 * s;  //  rotation
        pcm[1][j] = s0 * s + s1 * c;
        b = pcm[2][j];
        pcm[0][j] = a * cb - b * sb;
        pcm[2][j] = a * sb + b * cb;
      }
    }
  }
  for (i = 0; i < vecLen; ++i) {
    vec[i]->SetMomentum(pcm[0][i] * GeV, pcm[1][i] * GeV, pcm[2][i] * GeV);
    vec[i]->SetTotalEnergy(energy[i] * GeV);
  }
  // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
  return weight;
}

G4double FullModelReactionDynamics::normal() {
  G4double ran = -6.0;
  for (G4int i = 0; i < 12; ++i)
    ran += G4UniformRand();
  return ran;
}

G4int FullModelReactionDynamics::Factorial(G4int n) {  // calculates factorial( n ) = n*(n-1)*(n-2)*...*1
  G4int m = std::min(n, 10);
  G4int result = 1;
  if (m <= 1)
    return result;
  for (G4int i = 2; i <= m; ++i)
    result *= i;
  return result;
}

void FullModelReactionDynamics::Defs1(const G4ReactionProduct &modifiedOriginal,
                                      G4ReactionProduct &currentParticle,
                                      G4ReactionProduct &targetParticle,
                                      G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                                      G4int &vecLen) {
  const G4double pjx = modifiedOriginal.GetMomentum().x() / MeV;
  const G4double pjy = modifiedOriginal.GetMomentum().y() / MeV;
  const G4double pjz = modifiedOriginal.GetMomentum().z() / MeV;
  const G4double p = modifiedOriginal.GetMomentum().mag() / MeV;
  if (pjx * pjx + pjy * pjy > 0.0) {
    G4double cost, sint, ph, cosp, sinp, pix, piy, piz;
    cost = pjz / p;
    sint = 0.5 * (std::sqrt(std::abs((1.0 - cost) * (1.0 + cost))) + std::sqrt(pjx * pjx + pjy * pjy) / p);
    if (pjy < 0.0)
      ph = 3 * halfpi;
    else
      ph = halfpi;
    if (std::abs(pjx) > 0.001 * MeV)
      ph = std::atan2(pjy, pjx);
    cosp = std::cos(ph);
    sinp = std::sin(ph);
    pix = currentParticle.GetMomentum().x() / MeV;
    piy = currentParticle.GetMomentum().y() / MeV;
    piz = currentParticle.GetMomentum().z() / MeV;
    currentParticle.SetMomentum(cost * cosp * pix * MeV - sinp * piy + sint * cosp * piz * MeV,
                                cost * sinp * pix * MeV + cosp * piy + sint * sinp * piz * MeV,
                                -sint * pix * MeV + cost * piz * MeV);
    pix = targetParticle.GetMomentum().x() / MeV;
    piy = targetParticle.GetMomentum().y() / MeV;
    piz = targetParticle.GetMomentum().z() / MeV;
    targetParticle.SetMomentum(cost * cosp * pix * MeV - sinp * piy + sint * cosp * piz * MeV,
                               cost * sinp * pix * MeV + cosp * piy + sint * sinp * piz * MeV,
                               -sint * pix * MeV + cost * piz * MeV);
    for (G4int i = 0; i < vecLen; ++i) {
      pix = vec[i]->GetMomentum().x() / MeV;
      piy = vec[i]->GetMomentum().y() / MeV;
      piz = vec[i]->GetMomentum().z() / MeV;
      vec[i]->SetMomentum(cost * cosp * pix * MeV - sinp * piy + sint * cosp * piz * MeV,
                          cost * sinp * pix * MeV + cosp * piy + sint * sinp * piz * MeV,
                          -sint * pix * MeV + cost * piz * MeV);
    }
  } else {
    if (pjz < 0.0) {
      currentParticle.SetMomentum(-currentParticle.GetMomentum().z());
      targetParticle.SetMomentum(-targetParticle.GetMomentum().z());
      for (G4int i = 0; i < vecLen; ++i)
        vec[i]->SetMomentum(-vec[i]->GetMomentum().z());
    }
  }
}

void FullModelReactionDynamics::Rotate(
    const G4double numberofFinalStateNucleons,
    const G4ThreeVector &temp,
    const G4ReactionProduct &modifiedOriginal,  // Fermi motion & evap. effect included
    const G4HadProjectile *originalIncident,    // original incident particle
    const G4Nucleus &targetNucleus,
    G4ReactionProduct &currentParticle,
    G4ReactionProduct &targetParticle,
    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
    G4int &vecLen) {
  // derived from original FORTRAN code in GENXPT and TWOCLU by H. Fesefeldt
  //
  //   Rotate in direction of z-axis, this does disturb in some way our
  //    inclusive distributions, but it is necessary for momentum conservation
  //
  const G4double atomicWeight = targetNucleus.GetN_asInt();
  const G4double logWeight = std::log(atomicWeight);

  G4ParticleDefinition *aPiMinus = G4PionMinus::PionMinus();
  G4ParticleDefinition *aPiPlus = G4PionPlus::PionPlus();
  G4ParticleDefinition *aPiZero = G4PionZero::PionZero();

  G4int i;
  G4ThreeVector pseudoParticle[4];
  for (i = 0; i < 4; ++i)
    pseudoParticle[i].set(0, 0, 0);
  pseudoParticle[0] = currentParticle.GetMomentum() + targetParticle.GetMomentum();
  for (i = 0; i < vecLen; ++i)
    pseudoParticle[0] = pseudoParticle[0] + (vec[i]->GetMomentum());
  //
  //  Some smearing in transverse direction from Fermi motion
  //
  G4float pp, pp1;
  G4double alekw, p, rthnve, phinve;
  G4double r1, r2, a1, ran1, ran2, xxh, exh, pxTemp, pyTemp, pzTemp;

  r1 = twopi * G4UniformRand();
  r2 = G4UniformRand();
  a1 = std::sqrt(-2.0 * std::log(r2));
  ran1 = a1 * std::sin(r1) * 0.020 * numberofFinalStateNucleons * GeV;
  ran2 = a1 * std::cos(r1) * 0.020 * numberofFinalStateNucleons * GeV;
  G4ThreeVector fermi(ran1, ran2, 0);

  pseudoParticle[0] = pseudoParticle[0] + fermi;  // all particles + fermi
  pseudoParticle[2] = temp;                       // original in cms system
  pseudoParticle[3] = pseudoParticle[0];

  pseudoParticle[1] = pseudoParticle[2].cross(pseudoParticle[3]);
  G4double rotation = 2. * pi * G4UniformRand();
  pseudoParticle[1] = pseudoParticle[1].rotate(rotation, pseudoParticle[3]);
  pseudoParticle[2] = pseudoParticle[3].cross(pseudoParticle[1]);
  for (G4int ii = 1; ii <= 3; ii++) {
    p = pseudoParticle[ii].mag();
    if (p == 0.0)
      pseudoParticle[ii] = G4ThreeVector(0.0, 0.0, 0.0);
    else
      pseudoParticle[ii] = pseudoParticle[ii] * (1. / p);
  }

  pxTemp = pseudoParticle[1].dot(currentParticle.GetMomentum());
  pyTemp = pseudoParticle[2].dot(currentParticle.GetMomentum());
  pzTemp = pseudoParticle[3].dot(currentParticle.GetMomentum());
  currentParticle.SetMomentum(pxTemp, pyTemp, pzTemp);

  pxTemp = pseudoParticle[1].dot(targetParticle.GetMomentum());
  pyTemp = pseudoParticle[2].dot(targetParticle.GetMomentum());
  pzTemp = pseudoParticle[3].dot(targetParticle.GetMomentum());
  targetParticle.SetMomentum(pxTemp, pyTemp, pzTemp);

  for (i = 0; i < vecLen; ++i) {
    pxTemp = pseudoParticle[1].dot(vec[i]->GetMomentum());
    pyTemp = pseudoParticle[2].dot(vec[i]->GetMomentum());
    pzTemp = pseudoParticle[3].dot(vec[i]->GetMomentum());
    vec[i]->SetMomentum(pxTemp, pyTemp, pzTemp);
  }
  //
  //  Rotate in direction of primary particle, subtract binding energies
  //   and make some further corrections if required
  //
  Defs1(modifiedOriginal, currentParticle, targetParticle, vec, vecLen);
  G4double ekin;
  G4double dekin = 0.0;
  G4double ek1 = 0.0;
  G4int npions = 0;
  if (atomicWeight >= 1.5)  // self-absorption in heavy molecules
  {
    // corrections for single particle spectra (shower particles)
    //
    const G4double alem[] = {1.40, 2.30, 2.70, 3.00, 3.40, 4.60, 7.00};
    const G4double val0[] = {0.00, 0.40, 0.48, 0.51, 0.54, 0.60, 0.65};
    alekw = std::log(originalIncident->GetKineticEnergy() / GeV);
    exh = 1.0;
    if (alekw > alem[0])  //   get energy bin
    {
      exh = val0[6];
      for (G4int j = 1; j < 7; ++j) {
        if (alekw < alem[j])  // use linear interpolation/extrapolation
        {
          G4double rcnve = (val0[j] - val0[j - 1]) / (alem[j] - alem[j - 1]);
          exh = rcnve * alekw + val0[j - 1] - rcnve * alem[j - 1];
          break;
        }
      }
      exh = 1.0 - exh;
    }
    const G4double cfa = 0.025 * ((atomicWeight - 1.) / 120.) * std::exp(-(atomicWeight - 1.) / 120.);
    ekin = currentParticle.GetKineticEnergy() / GeV - cfa * (1 + normal() / 2.0);
    ekin = std::max(1.0e-6, ekin);
    xxh = 1.0;
    dekin += ekin * (1.0 - xxh);
    ekin *= xxh;
    currentParticle.SetKineticEnergy(ekin * GeV);
    pp = currentParticle.GetTotalMomentum() / MeV;
    pp1 = currentParticle.GetMomentum().mag() / MeV;
    if (pp1 < 0.001 * MeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      currentParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                  pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                  pp * std::cos(rthnve) * MeV);
    } else
      currentParticle.SetMomentum(currentParticle.GetMomentum() * (pp / pp1));
    ekin = targetParticle.GetKineticEnergy() / GeV - cfa * (1 + normal() / 2.0);
    ekin = std::max(1.0e-6, ekin);
    xxh = 1.0;
    if (((modifiedOriginal.GetDefinition() == aPiPlus) || (modifiedOriginal.GetDefinition() == aPiMinus)) &&
        (targetParticle.GetDefinition() == aPiZero) && (G4UniformRand() < logWeight))
      xxh = exh;
    dekin += ekin * (1.0 - xxh);
    ekin *= xxh;
    if ((targetParticle.GetDefinition() == aPiPlus) || (targetParticle.GetDefinition() == aPiZero) ||
        (targetParticle.GetDefinition() == aPiMinus)) {
      ++npions;
      ek1 += ekin;
    }
    targetParticle.SetKineticEnergy(ekin * GeV);
    pp = targetParticle.GetTotalMomentum() / MeV;
    pp1 = targetParticle.GetMomentum().mag() / MeV;
    if (pp1 < 0.001 * MeV) {
      rthnve = pi * G4UniformRand();
      phinve = twopi * G4UniformRand();
      targetParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                 pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                 pp * std::cos(rthnve) * MeV);
    } else
      targetParticle.SetMomentum(targetParticle.GetMomentum() * (pp / pp1));
    for (i = 0; i < vecLen; ++i) {
      ekin = vec[i]->GetKineticEnergy() / GeV - cfa * (1 + normal() / 2.0);
      ekin = std::max(1.0e-6, ekin);
      xxh = 1.0;
      if (((modifiedOriginal.GetDefinition() == aPiPlus) || (modifiedOriginal.GetDefinition() == aPiMinus)) &&
          (vec[i]->GetDefinition() == aPiZero) && (G4UniformRand() < logWeight))
        xxh = exh;
      dekin += ekin * (1.0 - xxh);
      ekin *= xxh;
      if ((vec[i]->GetDefinition() == aPiPlus) || (vec[i]->GetDefinition() == aPiZero) ||
          (vec[i]->GetDefinition() == aPiMinus)) {
        ++npions;
        ek1 += ekin;
      }
      vec[i]->SetKineticEnergy(ekin * GeV);
      pp = vec[i]->GetTotalMomentum() / MeV;
      pp1 = vec[i]->GetMomentum().mag() / MeV;
      if (pp1 < 0.001 * MeV) {
        rthnve = pi * G4UniformRand();
        phinve = twopi * G4UniformRand();
        vec[i]->SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                            pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                            pp * std::cos(rthnve) * MeV);
      } else
        vec[i]->SetMomentum(vec[i]->GetMomentum() * (pp / pp1));
    }
  }
  if ((ek1 != 0.0) && (npions > 0)) {
    dekin = 1.0 + dekin / ek1;
    //
    //  first do the incident particle
    //
    if ((currentParticle.GetDefinition() == aPiPlus) || (currentParticle.GetDefinition() == aPiZero) ||
        (currentParticle.GetDefinition() == aPiMinus)) {
      currentParticle.SetKineticEnergy(std::max(0.001 * MeV, dekin * currentParticle.GetKineticEnergy()));
      pp = currentParticle.GetTotalMomentum() / MeV;
      pp1 = currentParticle.GetMomentum().mag() / MeV;
      if (pp1 < 0.001) {
        rthnve = pi * G4UniformRand();
        phinve = twopi * G4UniformRand();
        currentParticle.SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                                    pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                                    pp * std::cos(rthnve) * MeV);
      } else
        currentParticle.SetMomentum(currentParticle.GetMomentum() * (pp / pp1));
    }
    for (i = 0; i < vecLen; ++i) {
      if ((vec[i]->GetDefinition() == aPiPlus) || (vec[i]->GetDefinition() == aPiZero) ||
          (vec[i]->GetDefinition() == aPiMinus)) {
        vec[i]->SetKineticEnergy(std::max(0.001 * MeV, dekin * vec[i]->GetKineticEnergy()));
        pp = vec[i]->GetTotalMomentum() / MeV;
        pp1 = vec[i]->GetMomentum().mag() / MeV;
        if (pp1 < 0.001) {
          rthnve = pi * G4UniformRand();
          phinve = twopi * G4UniformRand();
          vec[i]->SetMomentum(pp * std::sin(rthnve) * std::cos(phinve) * MeV,
                              pp * std::sin(rthnve) * std::sin(phinve) * MeV,
                              pp * std::cos(rthnve) * MeV);
        } else
          vec[i]->SetMomentum(vec[i]->GetMomentum() * (pp / pp1));
      }
    }
  }
}

void FullModelReactionDynamics::AddBlackTrackParticles(const G4double epnb,  // GeV
                                                       const G4int npnb,
                                                       const G4double edta,  // GeV
                                                       const G4int ndta,
                                                       const G4double sprob,
                                                       const G4double kineticMinimum,  // GeV
                                                       const G4double kineticFactor,   // GeV
                                                       const G4ReactionProduct &modifiedOriginal,
                                                       G4double spall,
                                                       const G4Nucleus &targetNucleus,
                                                       G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                                                       G4int &vecLen) {
  // derived from original FORTRAN code in GENXPT and TWOCLU by H. Fesefeldt
  //
  // npnb is number of proton/neutron black track particles
  // ndta is the number of deuterons, tritons, and alphas produced
  // epnb is the kinetic energy available for proton/neutron black track particles
  // edta is the kinetic energy available for deuteron/triton/alpha particles
  //

  G4ParticleDefinition *aProton = G4Proton::Proton();
  G4ParticleDefinition *aNeutron = G4Neutron::Neutron();
  G4ParticleDefinition *aDeuteron = G4Deuteron::Deuteron();
  G4ParticleDefinition *aTriton = G4Triton::Triton();
  G4ParticleDefinition *anAlpha = G4Alpha::Alpha();

  const G4double ekOriginal = modifiedOriginal.GetKineticEnergy() / MeV;
  G4double atomicWeight = targetNucleus.GetN_asInt();
  G4double atomicNumber = targetNucleus.GetZ_asInt();

  const G4double ika1 = 3.6;
  const G4double ika2 = 35.56;
  const G4double ika3 = 6.45;
  const G4double sp1 = 1.066;

  G4int i;
  G4double pp;
  // G4double totalQ = 0;
  //G4double kinCreated = 0;
  G4double cfa = 0.025 * ((atomicWeight - 1.0) / 120.0) * std::exp(-(atomicWeight - 1.0) / 120.0);
  if (npnb > 0)  // first add protons and neutrons
  {
    G4double backwardKinetic = 0.0;
    G4int local_npnb = npnb;
    for (i = 0; i < npnb; ++i)
      if (G4UniformRand() < sprob)
        local_npnb--;
    G4double ekin = epnb / local_npnb;

    for (i = 0; i < local_npnb; ++i) {
      G4ReactionProduct *p1 = new G4ReactionProduct();
      if (backwardKinetic > epnb) {
        delete p1;
        break;
      }
      G4double ran = G4UniformRand();
      G4double kinetic = -ekin * std::log(ran) - cfa * (1.0 + 0.5 * normal());
      if (kinetic < 0.0)
        kinetic = -0.010 * std::log(ran);
      backwardKinetic += kinetic;
      if (backwardKinetic > epnb)
        kinetic = std::max(kineticMinimum, epnb - (backwardKinetic - kinetic));
      if (G4UniformRand() > (1.0 - atomicNumber / atomicWeight))
        p1->SetDefinition(aProton);
      else
        p1->SetDefinition(aNeutron);
      vec.SetElement(vecLen, p1);
      ++spall;
      G4double cost = G4UniformRand() * 2.0 - 1.0;
      G4double sint = std::sqrt(std::fabs(1.0 - cost * cost));
      G4double phi = twopi * G4UniformRand();
      vec[vecLen]->SetNewlyAdded(true);
      vec[vecLen]->SetKineticEnergy(kinetic * GeV);
      //kinCreated += kinetic;
      pp = vec[vecLen]->GetTotalMomentum() / MeV;
      vec[vecLen]->SetMomentum(pp * sint * std::sin(phi) * MeV, pp * sint * std::cos(phi) * MeV, pp * cost * MeV);
      vecLen++;
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    }
    if ((atomicWeight >= 10.0) && (ekOriginal <= 2.0 * GeV)) {
      G4double ekw = ekOriginal / GeV;
      G4int ika, kk = 0;
      if (ekw > 1.0)
        ekw *= ekw;
      ekw = std::max(0.1, ekw);
      ika = G4int(ika1 * std::exp((atomicNumber * atomicNumber / atomicWeight - ika2) / ika3) / ekw);
      if (ika > 0) {
        for (i = (vecLen - 1); i >= 0; --i) {
          if ((vec[i]->GetDefinition() == aProton) && vec[i]->GetNewlyAdded()) {
            vec[i]->SetDefinitionAndUpdateE(aNeutron);  // modified 22-Oct-97
            if (++kk > ika)
              break;
          }
        }
      }
    }
  }
  if (ndta > 0)  //  now, try to add deuterons, tritons and alphas
  {
    G4double backwardKinetic = 0.0;
    G4int local_ndta = ndta;
    for (i = 0; i < ndta; ++i)
      if (G4UniformRand() < sprob)
        local_ndta--;
    G4double ekin = edta / local_ndta;

    for (i = 0; i < local_ndta; ++i) {
      G4ReactionProduct *p2 = new G4ReactionProduct();
      if (backwardKinetic > edta) {
        delete p2;
        break;
      }
      G4double ran = G4UniformRand();
      G4double kinetic = -ekin * std::log(ran) - cfa * (1. + 0.5 * normal());
      if (kinetic < 0.0)
        kinetic = kineticFactor * std::log(ran);
      backwardKinetic += kinetic;
      if (backwardKinetic > edta)
        kinetic = edta - (backwardKinetic - kinetic);
      if (kinetic < 0.0)
        kinetic = kineticMinimum;
      G4double cost = 2.0 * G4UniformRand() - 1.0;
      G4double sint = std::sqrt(std::max(0.0, (1.0 - cost * cost)));
      G4double phi = twopi * G4UniformRand();
      ran = G4UniformRand();
      if (ran <= 0.60)
        p2->SetDefinition(aDeuteron);
      else if (ran <= 0.90)
        p2->SetDefinition(aTriton);
      else
        p2->SetDefinition(anAlpha);
      spall += p2->GetMass() / GeV * sp1;
      if (spall > atomicWeight) {
        delete p2;
        break;
      }
      vec.SetElement(vecLen, p2);
      vec[vecLen]->SetNewlyAdded(true);
      vec[vecLen]->SetKineticEnergy(kinetic * GeV);
      //kinCreated += kinetic;
      pp = vec[vecLen]->GetTotalMomentum() / MeV;
      vec[vecLen++]->SetMomentum(pp * sint * std::sin(phi) * MeV, pp * sint * std::cos(phi) * MeV, pp * cost * MeV);
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    }
  }
  // G4double delta = epnb+edta - kinCreated;
}

void FullModelReactionDynamics::MomentumCheck(const G4ReactionProduct &modifiedOriginal,
                                              G4ReactionProduct &currentParticle,
                                              G4ReactionProduct &targetParticle,
                                              G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                                              G4int &vecLen) {
  const G4double pOriginal = modifiedOriginal.GetTotalMomentum() / MeV;
  G4double testMomentum = currentParticle.GetMomentum().mag() / MeV;
  G4double pMass;
  if (testMomentum >= pOriginal) {
    pMass = currentParticle.GetMass() / MeV;
    currentParticle.SetTotalEnergy(std::sqrt(pMass * pMass + pOriginal * pOriginal) * MeV);
    currentParticle.SetMomentum(currentParticle.GetMomentum() * (pOriginal / testMomentum));
  }
  testMomentum = targetParticle.GetMomentum().mag() / MeV;
  if (testMomentum >= pOriginal) {
    pMass = targetParticle.GetMass() / MeV;
    targetParticle.SetTotalEnergy(std::sqrt(pMass * pMass + pOriginal * pOriginal) * MeV);
    targetParticle.SetMomentum(targetParticle.GetMomentum() * (pOriginal / testMomentum));
  }
  for (G4int i = 0; i < vecLen; ++i) {
    testMomentum = vec[i]->GetMomentum().mag() / MeV;
    if (testMomentum >= pOriginal) {
      pMass = vec[i]->GetMass() / MeV;
      vec[i]->SetTotalEnergy(std::sqrt(pMass * pMass + pOriginal * pOriginal) * MeV);
      vec[i]->SetMomentum(vec[i]->GetMomentum() * (pOriginal / testMomentum));
    }
  }
}

void FullModelReactionDynamics::ProduceStrangeParticlePairs(G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                                                            G4int &vecLen,
                                                            const G4ReactionProduct &modifiedOriginal,
                                                            const G4DynamicParticle *originalTarget,
                                                            G4ReactionProduct &currentParticle,
                                                            G4ReactionProduct &targetParticle,
                                                            G4bool &incidentHasChanged,
                                                            G4bool &targetHasChanged) {
  // derived from original FORTRAN code STPAIR by H. Fesefeldt (16-Dec-1987)
  //
  // Choose charge combinations K+ K-, K+ K0B, K0 K0B, K0 K-,
  //                            K+ Y0, K0 Y+,  K0 Y-
  // For antibaryon induced reactions half of the cross sections KB YB
  // pairs are produced.  Charge is not conserved, no experimental data available
  // for exclusive reactions, therefore some average behaviour assumed.
  // The ratio L/SIGMA is taken as 3:1 (from experimental low energy)
  //
  if (vecLen == 0)
    return;
  //
  // the following protects against annihilation processes
  //
  if (currentParticle.GetMass() == 0.0 || targetParticle.GetMass() == 0.0)
    return;

  const G4double etOriginal = modifiedOriginal.GetTotalEnergy() / GeV;
  const G4double mOriginal = modifiedOriginal.GetDefinition()->GetPDGMass() / GeV;
  G4double targetMass = originalTarget->GetDefinition()->GetPDGMass() / GeV;
  G4double centerofmassEnergy =
      std::sqrt(mOriginal * mOriginal + targetMass * targetMass + 2.0 * targetMass * etOriginal);  // GeV
  G4double currentMass = currentParticle.GetMass() / GeV;
  G4double availableEnergy = centerofmassEnergy - (targetMass + currentMass);
  if (availableEnergy <= 1.0)
    return;

  G4ParticleDefinition *aProton = G4Proton::Proton();
  G4ParticleDefinition *anAntiProton = G4AntiProton::AntiProton();
  G4ParticleDefinition *aNeutron = G4Neutron::Neutron();
  G4ParticleDefinition *anAntiNeutron = G4AntiNeutron::AntiNeutron();
  G4ParticleDefinition *aSigmaMinus = G4SigmaMinus::SigmaMinus();
  G4ParticleDefinition *aSigmaPlus = G4SigmaPlus::SigmaPlus();
  G4ParticleDefinition *aSigmaZero = G4SigmaZero::SigmaZero();
  G4ParticleDefinition *anAntiSigmaMinus = G4AntiSigmaMinus::AntiSigmaMinus();
  G4ParticleDefinition *anAntiSigmaPlus = G4AntiSigmaPlus::AntiSigmaPlus();
  G4ParticleDefinition *anAntiSigmaZero = G4AntiSigmaZero::AntiSigmaZero();
  G4ParticleDefinition *aKaonMinus = G4KaonMinus::KaonMinus();
  G4ParticleDefinition *aKaonPlus = G4KaonPlus::KaonPlus();
  G4ParticleDefinition *aKaonZL = G4KaonZeroLong::KaonZeroLong();
  G4ParticleDefinition *aKaonZS = G4KaonZeroShort::KaonZeroShort();
  G4ParticleDefinition *aLambda = G4Lambda::Lambda();
  G4ParticleDefinition *anAntiLambda = G4AntiLambda::AntiLambda();

  const G4double protonMass = aProton->GetPDGMass() / GeV;
  const G4double sigmaMinusMass = aSigmaMinus->GetPDGMass() / GeV;
  //
  // determine the center of mass energy bin
  //
  const G4double avrs[] = {3., 4., 5., 6., 7., 8., 9., 10., 20., 30., 40., 50.};

  G4int ibin, i3, i4;
  G4double avk, avy, avn, ran;
  G4int i = 1;
  while ((i < 12) && (centerofmassEnergy > avrs[i]))
    ++i;
  if (i == 12)
    ibin = 11;
  else
    ibin = i;
  //
  // the fortran code chooses a random replacement of produced kaons
  //  but does not take into account charge conservation
  //
  if (vecLen == 1)  // we know that vecLen > 0
  {
    i3 = 0;
    i4 = 1;  // note that we will be adding a new secondary particle in this case only
  } else     // otherwise  0 <= i3,i4 < vecLen
  {
    G4double ran = G4UniformRand();
    while (ran == 1.0)
      ran = G4UniformRand();
    i4 = i3 = G4int(vecLen * ran);
    while (i3 == i4) {
      ran = G4UniformRand();
      while (ran == 1.0)
        ran = G4UniformRand();
      i4 = G4int(vecLen * ran);
    }
  }
  //
  // use linear interpolation or extrapolation by y=centerofmassEnergy*x+b
  //
  const G4double avkkb[] = {0.0015, 0.005, 0.012, 0.0285, 0.0525, 0.075, 0.0975, 0.123, 0.28, 0.398, 0.495, 0.573};
  const G4double avky[] = {0.005, 0.03, 0.064, 0.095, 0.115, 0.13, 0.145, 0.155, 0.20, 0.205, 0.210, 0.212};
  const G4double avnnb[] = {0.00001, 0.0001, 0.0006, 0.0025, 0.01, 0.02, 0.04, 0.05, 0.12, 0.15, 0.18, 0.20};

  avk = (std::log(avkkb[ibin]) - std::log(avkkb[ibin - 1])) * (centerofmassEnergy - avrs[ibin - 1]) /
            (avrs[ibin] - avrs[ibin - 1]) +
        std::log(avkkb[ibin - 1]);
  avk = std::exp(avk);

  avy = (std::log(avky[ibin]) - std::log(avky[ibin - 1])) * (centerofmassEnergy - avrs[ibin - 1]) /
            (avrs[ibin] - avrs[ibin - 1]) +
        std::log(avky[ibin - 1]);
  avy = std::exp(avy);

  avn = (std::log(avnnb[ibin]) - std::log(avnnb[ibin - 1])) * (centerofmassEnergy - avrs[ibin - 1]) /
            (avrs[ibin] - avrs[ibin - 1]) +
        std::log(avnnb[ibin - 1]);
  avn = std::exp(avn);

  if (avk + avy + avn <= 0.0)
    return;

  if (currentMass < protonMass)
    avy /= 2.0;
  if (targetMass < protonMass)
    avy = 0.0;
  avy += avk + avn;
  avk += avn;
  ran = G4UniformRand();
  if (ran < avn) {
    if (availableEnergy < 2.0)
      return;
    if (vecLen == 1)  // add a new secondary
    {
      G4ReactionProduct *p1 = new G4ReactionProduct;
      if (G4UniformRand() < 0.5) {
        vec[0]->SetDefinition(aNeutron);
        p1->SetDefinition(anAntiNeutron);
        (G4UniformRand() < 0.5) ? p1->SetSide(-1) : p1->SetSide(1);
        vec[0]->SetMayBeKilled(false);
        p1->SetMayBeKilled(false);
      } else {
        vec[0]->SetDefinition(aProton);
        p1->SetDefinition(anAntiProton);
        (G4UniformRand() < 0.5) ? p1->SetSide(-1) : p1->SetSide(1);
        vec[0]->SetMayBeKilled(false);
        p1->SetMayBeKilled(false);
      }
      vec.SetElement(vecLen++, p1);
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    } else {  // replace two secondaries
      if (G4UniformRand() < 0.5) {
        vec[i3]->SetDefinition(aNeutron);
        vec[i4]->SetDefinition(anAntiNeutron);
        vec[i3]->SetMayBeKilled(false);
        vec[i4]->SetMayBeKilled(false);
      } else {
        vec[i3]->SetDefinition(aProton);
        vec[i4]->SetDefinition(anAntiProton);
        vec[i3]->SetMayBeKilled(false);
        vec[i4]->SetMayBeKilled(false);
      }
    }
  } else if (ran < avk) {
    if (availableEnergy < 1.0)
      return;

    const G4double kkb[] = {0.2500, 0.3750, 0.5000, 0.5625, 0.6250, 0.6875, 0.7500, 0.8750, 1.000};
    const G4int ipakkb1[] = {10, 10, 10, 11, 11, 12, 12, 11, 12};
    const G4int ipakkb2[] = {13, 11, 12, 11, 12, 11, 12, 13, 13};
    ran = G4UniformRand();
    i = 0;
    while ((i < 9) && (ran >= kkb[i]))
      ++i;
    if (i == 9)
      return;
    //
    // ipakkb[] = { 10,13, 10,11, 10,12, 11,11, 11,12, 12,11, 12,12, 11,13, 12,13 };
    // charge       +  -   +  0   +  0   0  0   0  0   0  0   0  0   0  -   0  -
    //
    switch (ipakkb1[i]) {
      case 10:
        vec[i3]->SetDefinition(aKaonPlus);
        vec[i3]->SetMayBeKilled(false);
        break;
      case 11:
        vec[i3]->SetDefinition(aKaonZS);
        vec[i3]->SetMayBeKilled(false);
        break;
      case 12:
        vec[i3]->SetDefinition(aKaonZL);
        vec[i3]->SetMayBeKilled(false);
        break;
    }
    if (vecLen == 1)  // add a secondary
    {
      G4ReactionProduct *p1 = new G4ReactionProduct;
      switch (ipakkb2[i]) {
        case 11:
          p1->SetDefinition(aKaonZS);
          p1->SetMayBeKilled(false);
          break;
        case 12:
          p1->SetDefinition(aKaonZL);
          p1->SetMayBeKilled(false);
          break;
        case 13:
          p1->SetDefinition(aKaonMinus);
          p1->SetMayBeKilled(false);
          break;
      }
      (G4UniformRand() < 0.5) ? p1->SetSide(-1) : p1->SetSide(1);
      vec.SetElement(vecLen++, p1);
      // DEBUGGING --> DumpFrames::DumpFrame(vec, vecLen);
    } else  // replace
    {
      switch (ipakkb2[i]) {
        case 11:
          vec[i4]->SetDefinition(aKaonZS);
          vec[i4]->SetMayBeKilled(false);
          break;
        case 12:
          vec[i4]->SetDefinition(aKaonZL);
          vec[i4]->SetMayBeKilled(false);
          break;
        case 13:
          vec[i4]->SetDefinition(aKaonMinus);
          vec[i4]->SetMayBeKilled(false);
          break;
      }
    }
  } else if (ran < avy) {
    if (availableEnergy < 1.6)
      return;

    const G4double ky[] = {0.200, 0.300, 0.400, 0.550, 0.625, 0.700, 0.800, 0.850, 0.900, 0.950, 0.975, 1.000};
    const G4int ipaky1[] = {18, 18, 18, 20, 20, 20, 21, 21, 21, 22, 22, 22};
    const G4int ipaky2[] = {10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12};
    const G4int ipakyb1[] = {19, 19, 19, 23, 23, 23, 24, 24, 24, 25, 25, 25};
    const G4int ipakyb2[] = {13, 12, 11, 13, 12, 11, 13, 12, 11, 13, 12, 11};
    ran = G4UniformRand();
    i = 0;
    while ((i < 12) && (ran > ky[i]))
      ++i;
    if (i == 12)
      return;
    if ((currentMass < protonMass) || (G4UniformRand() < 0.5)) {
      switch (ipaky1[i]) {
        case 18:
          targetParticle.SetDefinition(aLambda);
          break;
        case 20:
          targetParticle.SetDefinition(aSigmaPlus);
          break;
        case 21:
          targetParticle.SetDefinition(aSigmaZero);
          break;
        case 22:
          targetParticle.SetDefinition(aSigmaMinus);
          break;
      }
      targetHasChanged = true;
      switch (ipaky2[i]) {
        case 10:
          vec[i3]->SetDefinition(aKaonPlus);
          vec[i3]->SetMayBeKilled(false);
          break;
        case 11:
          vec[i3]->SetDefinition(aKaonZS);
          vec[i3]->SetMayBeKilled(false);
          break;
        case 12:
          vec[i3]->SetDefinition(aKaonZL);
          vec[i3]->SetMayBeKilled(false);
          break;
      }
    } else  // (currentMass >= protonMass) && (G4UniformRand() >= 0.5)
    {
      if ((currentParticle.GetDefinition() == anAntiProton) || (currentParticle.GetDefinition() == anAntiNeutron) ||
          (currentParticle.GetDefinition() == anAntiLambda) || (currentMass > sigmaMinusMass)) {
        switch (ipakyb1[i]) {
          case 19:
            currentParticle.SetDefinitionAndUpdateE(anAntiLambda);
            break;
          case 23:
            currentParticle.SetDefinitionAndUpdateE(anAntiSigmaPlus);
            break;
          case 24:
            currentParticle.SetDefinitionAndUpdateE(anAntiSigmaZero);
            break;
          case 25:
            currentParticle.SetDefinitionAndUpdateE(anAntiSigmaMinus);
            break;
        }
        incidentHasChanged = true;
        switch (ipakyb2[i]) {
          case 11:
            vec[i3]->SetDefinition(aKaonZS);
            vec[i3]->SetMayBeKilled(false);
            break;
          case 12:
            vec[i3]->SetDefinition(aKaonZL);
            vec[i3]->SetMayBeKilled(false);
            break;
          case 13:
            vec[i3]->SetDefinition(aKaonMinus);
            vec[i3]->SetMayBeKilled(false);
            break;
        }
      } else {
        switch (ipaky1[i]) {
          case 18:
            currentParticle.SetDefinitionAndUpdateE(aLambda);
            break;
          case 20:
            currentParticle.SetDefinitionAndUpdateE(aSigmaPlus);
            break;
          case 21:
            currentParticle.SetDefinitionAndUpdateE(aSigmaZero);
            break;
          case 22:
            currentParticle.SetDefinitionAndUpdateE(aSigmaMinus);
            break;
        }
        incidentHasChanged = true;
        switch (ipaky2[i]) {
          case 10:
            vec[i3]->SetDefinition(aKaonPlus);
            vec[i3]->SetMayBeKilled(false);
            break;
          case 11:
            vec[i3]->SetDefinition(aKaonZS);
            vec[i3]->SetMayBeKilled(false);
            break;
          case 12:
            vec[i3]->SetDefinition(aKaonZL);
            vec[i3]->SetMayBeKilled(false);
            break;
        }
      }
    }
  } else
    return;
  //
  //  check the available energy
  //   if there is not enough energy for kkb/ky pair production
  //   then reduce the number of secondary particles
  //  NOTE:
  //        the number of secondaries may have been changed
  //        the incident and/or target particles may have changed
  //        charge conservation is ignored (as well as strangness conservation)
  //
  currentMass = currentParticle.GetMass() / GeV;
  targetMass = targetParticle.GetMass() / GeV;

  G4double energyCheck = centerofmassEnergy - (currentMass + targetMass);
  for (i = 0; i < vecLen; ++i) {
    energyCheck -= vec[i]->GetMass() / GeV;
    if (energyCheck < 0.0)  // chop off the secondary List
    {
      vecLen = std::max(0, --i);  // looks like a memory leak @@@@@@@@@@@@
      G4int j;
      for (j = i; j < vecLen; j++)
        delete vec[j];
      break;
    }
  }
  return;
}

void FullModelReactionDynamics::NuclearReaction(G4FastVector<G4ReactionProduct, 4> &vec,
                                                G4int &vecLen,
                                                const G4HadProjectile *originalIncident,
                                                const G4Nucleus &targetNucleus,
                                                const G4double theAtomicMass,
                                                const G4double *mass) {
  // derived from original FORTRAN code NUCREC by H. Fesefeldt (12-Feb-1987)
  //
  // Nuclear reaction kinematics at low energies
  //
  G4ParticleDefinition *aGamma = G4Gamma::Gamma();
  G4ParticleDefinition *aProton = G4Proton::Proton();
  G4ParticleDefinition *aNeutron = G4Neutron::Neutron();
  G4ParticleDefinition *aDeuteron = G4Deuteron::Deuteron();
  G4ParticleDefinition *aTriton = G4Triton::Triton();
  G4ParticleDefinition *anAlpha = G4Alpha::Alpha();

  const G4double aProtonMass = aProton->GetPDGMass() / MeV;
  const G4double aNeutronMass = aNeutron->GetPDGMass() / MeV;
  const G4double aDeuteronMass = aDeuteron->GetPDGMass() / MeV;
  const G4double aTritonMass = aTriton->GetPDGMass() / MeV;
  const G4double anAlphaMass = anAlpha->GetPDGMass() / MeV;

  G4ReactionProduct currentParticle;
  currentParticle = *originalIncident;
  //
  // Set beam particle, take kinetic energy of current particle as the
  // fundamental quantity.  Due to the difficult kinematic, all masses have to
  // be assigned the best measured values
  //
  G4double p = currentParticle.GetTotalMomentum();
  G4double pp = currentParticle.GetMomentum().mag();
  if (pp <= 0.001 * MeV) {
    G4double phinve = twopi * G4UniformRand();
    G4double rthnve = std::acos(std::max(-1.0, std::min(1.0, -1.0 + 2.0 * G4UniformRand())));
    currentParticle.SetMomentum(
        p * std::sin(rthnve) * std::cos(phinve), p * std::sin(rthnve) * std::sin(phinve), p * std::cos(rthnve));
  } else
    currentParticle.SetMomentum(currentParticle.GetMomentum() * (p / pp));
  //
  // calculate Q-value of reactions
  //
  G4double currentKinetic = currentParticle.GetKineticEnergy() / MeV;
  G4double currentMass = currentParticle.GetDefinition()->GetPDGMass() / MeV;
  G4double qv = currentKinetic + theAtomicMass + currentMass;

  G4double qval[9];
  qval[0] = qv - mass[0];
  qval[1] = qv - mass[1] - aNeutronMass;
  qval[2] = qv - mass[2] - aProtonMass;
  qval[3] = qv - mass[3] - aDeuteronMass;
  qval[4] = qv - mass[4] - aTritonMass;
  qval[5] = qv - mass[5] - anAlphaMass;
  qval[6] = qv - mass[6] - aNeutronMass - aNeutronMass;
  qval[7] = qv - mass[7] - aNeutronMass - aProtonMass;
  qval[8] = qv - mass[8] - aProtonMass - aProtonMass;

  if (currentParticle.GetDefinition() == aNeutron) {
    const G4double A = targetNucleus.GetN_asInt();  // atomic weight
    if (G4UniformRand() > ((A - 1.0) / 230.0) * ((A - 1.0) / 230.0))
      qval[0] = 0.0;
    if (G4UniformRand() >= currentKinetic / 7.9254 * A)
      qval[2] = qval[3] = qval[4] = qval[5] = qval[8] = 0.0;
  } else
    qval[0] = 0.0;

  G4int i;
  qv = 0.0;
  for (i = 0; i < 9; ++i) {
    if (mass[i] < 500.0 * MeV)
      qval[i] = 0.0;
    if (qval[i] < 0.0)
      qval[i] = 0.0;
    qv += qval[i];
  }
  G4double qv1 = 0.0;
  G4double ran = G4UniformRand();
  G4int index;
  for (index = 0; index < 9; ++index) {
    if (qval[index] > 0.0) {
      qv1 += qval[index] / qv;
      if (ran <= qv1)
        break;
    }
  }
  if (index == 9)  // loop continued to the end
  {
    throw G4HadronicException(
        __FILE__,
        __LINE__,
        "FullModelReactionDynamics::NuclearReaction: inelastic reaction kinematically not possible");
  }
  G4double ke = currentParticle.GetKineticEnergy() / GeV;
  G4int nt = 2;
  if ((index >= 6) || (G4UniformRand() < std::min(0.5, ke * 10.0)))
    nt = 3;

  G4ReactionProduct **v = new G4ReactionProduct *[3];
  v[0] = new G4ReactionProduct;
  v[1] = new G4ReactionProduct;
  v[2] = new G4ReactionProduct;

  v[0]->SetMass(mass[index] * MeV);
  switch (index) {
    case 0:
      v[1]->SetDefinition(aGamma);
      v[2]->SetDefinition(aGamma);
      break;
    case 1:
      v[1]->SetDefinition(aNeutron);
      v[2]->SetDefinition(aGamma);
      break;
    case 2:
      v[1]->SetDefinition(aProton);
      v[2]->SetDefinition(aGamma);
      break;
    case 3:
      v[1]->SetDefinition(aDeuteron);
      v[2]->SetDefinition(aGamma);
      break;
    case 4:
      v[1]->SetDefinition(aTriton);
      v[2]->SetDefinition(aGamma);
      break;
    case 5:
      v[1]->SetDefinition(anAlpha);
      v[2]->SetDefinition(aGamma);
      break;
    case 6:
      v[1]->SetDefinition(aNeutron);
      v[2]->SetDefinition(aNeutron);
      break;
    case 7:
      v[1]->SetDefinition(aNeutron);
      v[2]->SetDefinition(aProton);
      break;
    case 8:
      v[1]->SetDefinition(aProton);
      v[2]->SetDefinition(aProton);
      break;
  }
  //
  // calculate centre of mass energy
  //
  G4ReactionProduct pseudo1;
  pseudo1.SetMass(theAtomicMass * MeV);
  pseudo1.SetTotalEnergy(theAtomicMass * MeV);
  G4ReactionProduct pseudo2 = currentParticle + pseudo1;
  pseudo2.SetMomentum(pseudo2.GetMomentum() * (-1.0));
  //
  // use phase space routine in centre of mass system
  //
  G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> tempV;
  tempV.Initialize(nt);
  G4int tempLen = 0;
  tempV.SetElement(tempLen++, v[0]);
  tempV.SetElement(tempLen++, v[1]);
  if (nt == 3)
    tempV.SetElement(tempLen++, v[2]);
  G4bool constantCrossSection = true;
  GenerateNBodyEvent(pseudo2.GetMass() / MeV, constantCrossSection, tempV, tempLen);
  v[0]->Lorentz(*v[0], pseudo2);
  v[1]->Lorentz(*v[1], pseudo2);
  if (nt == 3)
    v[2]->Lorentz(*v[2], pseudo2);

  G4bool particleIsDefined = false;
  if (v[0]->GetMass() / MeV - aProtonMass < 0.1) {
    v[0]->SetDefinition(aProton);
    particleIsDefined = true;
  } else if (v[0]->GetMass() / MeV - aNeutronMass < 0.1) {
    v[0]->SetDefinition(aNeutron);
    particleIsDefined = true;
  } else if (v[0]->GetMass() / MeV - aDeuteronMass < 0.1) {
    v[0]->SetDefinition(aDeuteron);
    particleIsDefined = true;
  } else if (v[0]->GetMass() / MeV - aTritonMass < 0.1) {
    v[0]->SetDefinition(aTriton);
    particleIsDefined = true;
  } else if (v[0]->GetMass() / MeV - anAlphaMass < 0.1) {
    v[0]->SetDefinition(anAlpha);
    particleIsDefined = true;
  }
  currentParticle.SetKineticEnergy(std::max(0.001, currentParticle.GetKineticEnergy() / MeV));
  p = currentParticle.GetTotalMomentum();
  pp = currentParticle.GetMomentum().mag();
  if (pp <= 0.001 * MeV) {
    G4double phinve = twopi * G4UniformRand();
    G4double rthnve = std::acos(std::max(-1.0, std::min(1.0, -1.0 + 2.0 * G4UniformRand())));
    currentParticle.SetMomentum(
        p * std::sin(rthnve) * std::cos(phinve), p * std::sin(rthnve) * std::sin(phinve), p * std::cos(rthnve));
  } else
    currentParticle.SetMomentum(currentParticle.GetMomentum() * (p / pp));

  if (particleIsDefined) {
    v[0]->SetKineticEnergy(std::max(0.001, 0.5 * G4UniformRand() * v[0]->GetKineticEnergy() / MeV));
    p = v[0]->GetTotalMomentum();
    pp = v[0]->GetMomentum().mag();
    if (pp <= 0.001 * MeV) {
      G4double phinve = twopi * G4UniformRand();
      G4double rthnve = std::acos(std::max(-1.0, std::min(1.0, -1.0 + 2.0 * G4UniformRand())));
      v[0]->SetMomentum(
          p * std::sin(rthnve) * std::cos(phinve), p * std::sin(rthnve) * std::sin(phinve), p * std::cos(rthnve));
    } else
      v[0]->SetMomentum(v[0]->GetMomentum() * (p / pp));
  }
  if ((v[1]->GetDefinition() == aDeuteron) || (v[1]->GetDefinition() == aTriton) || (v[1]->GetDefinition() == anAlpha))
    v[1]->SetKineticEnergy(std::max(0.001, 0.5 * G4UniformRand() * v[1]->GetKineticEnergy() / MeV));
  else
    v[1]->SetKineticEnergy(std::max(0.001, v[1]->GetKineticEnergy() / MeV));

  p = v[1]->GetTotalMomentum();
  pp = v[1]->GetMomentum().mag();
  if (pp <= 0.001 * MeV) {
    G4double phinve = twopi * G4UniformRand();
    G4double rthnve = std::acos(std::max(-1.0, std::min(1.0, -1.0 + 2.0 * G4UniformRand())));
    v[1]->SetMomentum(
        p * std::sin(rthnve) * std::cos(phinve), p * std::sin(rthnve) * std::sin(phinve), p * std::cos(rthnve));
  } else
    v[1]->SetMomentum(v[1]->GetMomentum() * (p / pp));

  if (nt == 3) {
    if ((v[2]->GetDefinition() == aDeuteron) || (v[2]->GetDefinition() == aTriton) ||
        (v[2]->GetDefinition() == anAlpha))
      v[2]->SetKineticEnergy(std::max(0.001, 0.5 * G4UniformRand() * v[2]->GetKineticEnergy() / MeV));
    else
      v[2]->SetKineticEnergy(std::max(0.001, v[2]->GetKineticEnergy() / MeV));

    p = v[2]->GetTotalMomentum();
    pp = v[2]->GetMomentum().mag();
    if (pp <= 0.001 * MeV) {
      G4double phinve = twopi * G4UniformRand();
      G4double rthnve = std::acos(std::max(-1.0, std::min(1.0, -1.0 + 2.0 * G4UniformRand())));
      v[2]->SetMomentum(
          p * std::sin(rthnve) * std::cos(phinve), p * std::sin(rthnve) * std::sin(phinve), p * std::cos(rthnve));
    } else
      v[2]->SetMomentum(v[2]->GetMomentum() * (p / pp));
  }
  G4int del;
  for (del = 0; del < vecLen; del++)
    delete vec[del];
  vecLen = 0;
  if (particleIsDefined) {
    vec.SetElement(vecLen++, v[0]);
  } else {
    delete v[0];
  }
  vec.SetElement(vecLen++, v[1]);
  if (nt == 3) {
    vec.SetElement(vecLen++, v[2]);
  } else {
    delete v[2];
  }
  delete[] v;
  return;
}

/* end of file */

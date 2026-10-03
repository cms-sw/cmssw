import FWCore.ParameterSet.Config as cms

# Quirk simulation (port of the Athena Simulation/G4Extensions/Quirks).
# A quirk pair Q Qbar is bound by an infracolor string of tension
# F = Lambda^2 / (hbar c); 1 keV gives 5.07e3 MeV/mm.

def customiseQuirk(process, mass, lambdaEV, pdgId=17, charge=-1., stringForce=0., debugStep=0., verbose=0,
                   dumpEvery=0, dumpFile="quirkSteps", keepStopped=False):
    """mass in GeV, lambdaEV in eV, charge of the positive PDG code in e,
    stringForce in MeV/mm (overrides lambdaEV if > 0), debugStep in mm (0 = no debug watcher),
    dumpEvery: write every N-th quirk step to dumpFile_<thread>.txt,
    keepStopped: a quirk stopped by energy loss stays alive while its partner moves"""
    if not hasattr(process, 'g4SimHits'):
        return process
    g4 = process.g4SimHits
    g4.Physics.QuirkMass = cms.untracked.double(mass)
    g4.Physics.QuirkLambda = cms.untracked.double(lambdaEV)
    g4.Physics.QuirkStringForce = cms.untracked.double(stringForce)
    g4.Physics.QuirkPDGID = cms.untracked.int32(pdgId)
    g4.Physics.QuirkCharge = cms.untracked.double(charge)
    g4.Physics.QuirkVerbose = cms.untracked.int32(verbose)
    g4.Physics.QuirkKeepStopped = cms.untracked.bool(keepStopped)
    # suspend/resume of the two quirks needs the Geant4 event manager
    g4.UseG4EventManager = True
    if debugStep > 0 or verbose > 0 or dumpEvery > 0:
        g4.Watchers.append(cms.PSet(
            type = cms.string('QuirkDebugWatcher'),
            QuirkDebugWatcher = cms.PSet(
                DebugStep = cms.untracked.double(debugStep),
                Verbose = cms.untracked.int32(verbose),
                DumpEvery = cms.untracked.int32(dumpEvery),
                DumpFile = cms.untracked.string(dumpFile)
            )
        ))
    return process

def customise(process):
    """parameters from the generator fragment: quirkMass (GeV), quirkLambda (eV), optional quirkPDGID;
    for a SIM-only step use customiseQuirk in --customise_commands instead"""
    if not hasattr(process, 'generator') or not hasattr(process.generator, 'quirkMass'):
        raise RuntimeError("Exotica_Quirk_SIM_cfi.customise needs generator.quirkMass/quirkLambda; "
                           "without the generator call customiseQuirk(process, mass, lambdaEV)")
    gen = process.generator
    pdgId = gen.quirkPDGID.value() if hasattr(gen, 'quirkPDGID') else 17
    return customiseQuirk(process, gen.quirkMass.value(), gen.quirkLambda.value(), pdgId=pdgId)

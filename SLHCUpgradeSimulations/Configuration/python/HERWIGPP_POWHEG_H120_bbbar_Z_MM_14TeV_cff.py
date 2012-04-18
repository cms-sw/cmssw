import FWCore.ParameterSet.Config as cms
# Modified from Configuration/GenProduction/python/HERWIGPP_POWHEG_H120_bbbar_Z_ll_7TeV_cff.py
# Use with Tags V01-00-77 or newer of Configuration/GenProduction

from Configuration.Generator.HerwigppDefaults_cfi import *

generator = cms.EDFilter(
    "ThePEGGeneratorFilter",
    herwigDefaultsBlock,
    configFiles = cms.vstring(),

    parameterSets = cms.vstring(
    'cm14TeV',
    'powhegNewDefaults',
    'HbbZllParameters',
    'basicSetup',
    'setParticlesStableForDetector',
    ),
    
    powhegNewDefaults = cms.vstring(
    '# Need to use an NLO PDF',
    'cp /Herwig/Partons/MRST-NLO /cmsPDFSet',
    '# and strong coupling',
    'create Herwig::O2AlphaS O2AlphaS',
    'set /Herwig/Generators/LHCGenerator:StandardModelParameters:QCD/RunningAlphaS O2AlphaS',
    '# Setup the POWHEG shower',
    'cd /Herwig/Shower',
    '# use the general recon for now',
    'set KinematicsReconstructor:ReconstructionOption General',
    '# create the Powheg evolver and use it instead of the default one',
    'create Herwig::PowhegEvolver PowhegEvolver HwPowhegShower.so',
    'set ShowerHandler:Evolver PowhegEvolver',
    'set PowhegEvolver:ShowerModel ShowerModel',
    'set PowhegEvolver:SplittingGenerator SplittingGenerator',
    'set PowhegEvolver:MECorrMode 0',
    '# create and use the Drell-yan hard emission generator',
    'create Herwig::DrellYanHardGenerator DrellYanHardGenerator',
    'set DrellYanHardGenerator:ShowerAlpha AlphaQCD',
    'insert PowhegEvolver:HardGenerator 0 DrellYanHardGenerator',
    '# create and use the gg->H hard emission generator',
    'create Herwig::GGtoHHardGenerator GGtoHHardGenerator',
    'set GGtoHHardGenerator:ShowerAlpha AlphaQCD',
    'insert PowhegEvolver:HardGenerator 0 GGtoHHardGenerator',
    ),

    pdfCTEQ6M = cms.vstring(
    'mkdir /LHAPDF',
    'cd /LHAPDF',
    'create ThePEG::LHAPDF CTEQ6M',
    'set CTEQ6M:PDFName cteq6mE.LHgrid',
    'set CTEQ6M:RemnantHandler /Herwig/Partons/HadronRemnants',
    'cp CTEQ6M /cmsPDFSet',
    'cd /'
    ),
    
    HbbZllParameters = cms.vstring(
    'cd /Herwig/MatrixElements/',
    'insert SimpleQCD:MatrixElements[0] PowhegMEPP2ZH',
    'set /Herwig/Cuts/JetKtCut:MinKT 0.0*GeV',
    
    'set /Herwig/Particles/h0:NominalMass 120.*GeV',
    'set /Herwig/Particles/h0/h0->b,bbar;:OnOff On',
    'set /Herwig/Particles/h0/h0->b,bbar;:BranchingRatio 0.6649',
    'set /Herwig/Particles/h0/h0->W+,W-;:OnOff Off',
    'set /Herwig/Particles/h0/h0->tau-,tau+;:OnOff Off',
    'set /Herwig/Particles/h0/h0->g,g;:OnOff Off',
    'set /Herwig/Particles/h0/h0->c,cbar;:OnOff Off',
    'set /Herwig/Particles/h0/h0->Z0,Z0;:OnOff Off',
    'set /Herwig/Particles/h0/h0->gamma,gamma;:OnOff Off',
    'set /Herwig/Particles/h0/h0->mu-,mu+;:OnOff Off',
    'set /Herwig/Particles/h0/h0->t,tbar;:OnOff Off',

    'set /Herwig/Particles/Z0/Z0->d,dbar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->s,sbar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->b,bbar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->u,ubar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->c,cbar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->nu_e,nu_ebar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->nu_mu,nu_mubar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->nu_tau,nu_taubar;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->e-,e+;:OnOff Off',
    'set /Herwig/Particles/Z0/Z0->mu-,mu+;:OnOff On',
    'set /Herwig/Particles/Z0/Z0->tau-,tau+;:OnOff Off',
    ),
    
    crossSection     = cms.untracked.double(0.0230), # Did not adjust
    filterEfficiency = cms.untracked.double(1.0)
    )


configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('\$Revision: 1.1 $'),
    name = cms.untracked.string('\$Source: /cvs/CMSSW/CMSSW/SLHCUpgradeSimulations/Configuration/python/HERWIGPP_POWHEG_H120_bbbar_Z_MM_14TeV_cff.py,v $'),
	annotation = cms.untracked.string('HERWIGPP/POWHEG: (H->bb)(Z->MM), m(H)=120 GeV')
)

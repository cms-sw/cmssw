import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.TauTagValidation_cfi import *

# Apply the signal <--> background parameter transformation function 
#  (defined in TauTagValidation_cfi) to each of the Validation EDAnalyzers
map(ChangeAModuleToBackground, TauTagValidationPackages)

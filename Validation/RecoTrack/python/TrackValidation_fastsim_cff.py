import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
TrackAssociatorByHitsRecoDenom.associateStrip = False
TrackAssociatorByHitsRecoDenom.associatePixel = False
TrackAssociatorByHitsRecoDenom.ROUList = ['famosSimHitsTrackerHits']
multiTrackValidator.UseAssociators = True
###must be commented in normal running
###multiTrackValidator.outputFile='validationPlots.root'



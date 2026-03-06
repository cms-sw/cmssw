from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
import math
import os
import ROOT


ROOT.PyConfig.IgnoreCommandLineOptions = True


class NanoAODDirectSimProducer(Module):
    beam_energy: float = 0.

    def __init__(self, profile, verbosity=0):
        self.worker = ROOT.NanoAODDirectSimulator()
        self.worker.setVerbosity(verbosity)
        # feed the worker with configuration retrieved from conditions
        if not hasattr(profile, 'ctppsLHCInfo'):
            raise AttributeError('Failed to retrieve the LHC info from the profile.')
        if hasattr(profile.ctppsLHCInfo, 'beamEnergy'):
            self.beam_energy = float(profile.ctppsLHCInfo.beamEnergy.value())
            self.worker.setBeamEnergy(self.beam_energy)
        if hasattr(profile.ctppsLHCInfo, 'betaStar'):
            self.worker.setBetaStar(float(profile.ctppsLHCInfo.betaStar.value()))
        if hasattr(profile.ctppsLHCInfo, 'xangle'):
            self.worker.setCrossingAngle(float(profile.ctppsLHCInfo.xangle.value()))
        if not hasattr(profile, 'ctppsOpticalFunctions'):
            raise AttributeError('Failed to retrieve the optics configuration from the profile.')
        if not hasattr(profile.ctppsOpticalFunctions, 'opticalFunctions'):
            raise AttributeError('Failed to retrieve the list of optical functions from the optics configuration.')
        if not hasattr(profile.ctppsOpticalFunctions, 'scoringPlanes'):
            raise AttributeError('Failed to retrieve the list of scoring planes from the optics configuration.')
        for crossing_angle_functions in profile.ctppsOpticalFunctions.opticalFunctions:
            if hasattr(crossing_angle_functions, 'fileName') and hasattr(crossing_angle_functions, 'xangle'):
                self.worker.addCrossingAngleOpticalFunctions(float(crossing_angle_functions.xangle.value()),
                                                             str(crossing_angle_functions.fileName.value()))
        for scoring_plane in profile.ctppsOpticalFunctions.scoringPlanes:
            if hasattr(scoring_plane, 'rpId') and hasattr(scoring_plane, 'z') and hasattr(scoring_plane, 'dirName'):
                self.worker.addScoringPlane(int(scoring_plane.rpId.value()),
                                            float(scoring_plane.z.value()),
                                            str(scoring_plane.dirName.value()))

    def beginJob(self):
        self.worker.initialise()
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch("nPPSLocalTracks", "I")
        self.out.branch("PPSLocalTracks_x", "F", lenVar="nPPSLocalTracks")
        self.out.branch("PPSLocalTracks_y", "F", lenVar="nPPSLocalTracks")
        self.out.branch("PPSLocalTracks_time", "F", lenVar="nPPSLocalTracks")
        self.out.branch("PPSLocalTracks_timeUnc", "F", lenVar="nPPSLocalTracks")
        self.out.branch("PPSLocalTracks_multiRPProtonIdx", "F", lenVar="nPPSLocalTracks")
        self.out.branch("PPSLocalTracks_singleRPProtonIdx", "F", lenVar="nPPSLocalTracks")
        self.out.branch("PPSLocalTracks_decRPId", "I", lenVar="nPPSLocalTracks")
        self.out.branch("PPSLocalTracks_rpType", "I", lenVar="nPPSLocalTracks")
        self.out.branch("nProton_multiRP", "I")
        self.out.branch("Proton_multiRP_arm", "F", lenVar="nProton_multiRP")
        self.out.branch("Proton_multiRP_xi", "F", lenVar="nProton_multiRP")
        self.out.branch("Proton_multiRP_thetaX", "F", lenVar="nProton_multiRP")
        self.out.branch("Proton_multiRP_thetaY", "F", lenVar="nProton_multiRP")
        self.out.branch("Proton_multiRP_t", "F", lenVar="nProton_multiRP")
        self.out.branch("Proton_multiRP_time", "F", lenVar="nProton_multiRP")
        self.out.branch("Proton_multiRP_timeUnc", "F", lenVar="nProton_multiRP")
        self.out.branch("nProton_singleRP", "I")
        self.out.branch("Proton_singleRP_arm", "F", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_decRPId", "I", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_rpType", "I", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_x", "F", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_y", "F", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_time", "F", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_timeUnc", "F", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_xi", "F", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_thetaX", "F", lenVar="nProton_singleRP")
        self.out.branch("Proton_singleRP_thetaY", "F", lenVar="nProton_singleRP")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        proton_candidates = []
        genparts = Collection(event, "GenPart")
        for part in genparts:
            if part.pdgId == 2212 and (part.status == 1 or part.status > 83):
                proton_candidates.append(part)
            #if part.pdgId == 22 and part.status == 21:  # photon as incoming parton
            #    photon_mom = ROOT.TLorentzVector(part.pt * math.cos(part.phi), part.pt * math.sin(part.phi), 0., part.mass)
        if len(proton_candidates) == 0:
            return False

        # initialise collections to be filled
        nPPSLocalTracks = 0
        PPSLocalTracks_x = []
        PPSLocalTracks_y = []
        PPSLocalTracks_time = []
        PPSLocalTracks_timeUnc = []
        PPSLocalTracks_multiRPProtonIdx = []
        PPSLocalTracks_singleRPProtonIdx = []
        PPSLocalTracks_decRPId = []
        PPSLocalTracks_rpType = []
        nProton_multiRP = 0
        Proton_multiRP_arm = []
        Proton_multiRP_xi = []
        Proton_multiRP_thetaX = []
        Proton_multiRP_thetaY = []
        Proton_multiRP_t = []
        Proton_multiRP_time = []
        Proton_multiRP_timeUnc = []
        nProton_singleRP = 0
        Proton_singleRP_arm = []
        Proton_singleRP_decRPId = []
        Proton_singleRP_rpType = []
        Proton_singleRP_x = []
        Proton_singleRP_y = []
        Proton_singleRP_time = []
        Proton_singleRP_timeUnc = []
        Proton_singleRP_xi = []
        Proton_singleRP_thetaX = []
        Proton_singleRP_thetaY = []

        for proton in proton_candidates:
            proton_mom = ROOT.TLorentzVector()
            proton_mom.SetPtEtaPhiM(proton.pt, proton.eta, proton.phi, proton.mass)
            xi = 1. - proton_mom.Pz() / self.beam_energy  #FIXME
            for track in self.worker.computeProton(ROOT.TLorentzVector(), proton_mom):
                detid = ROOT.CTPPSDetId(track.rpId())
                dec_detid = 100 * detid.arm() + 10 * detid.station() + detid.rp()
                rp_type = detid.subdetId()
                nPPSLocalTracks += 1
                PPSLocalTracks_x.append(track.x())
                PPSLocalTracks_y.append(track.y())
                PPSLocalTracks_time.append(track.time())
                PPSLocalTracks_timeUnc.append(track.timeUnc())
                PPSLocalTracks_multiRPProtonIdx.append(nProton_multiRP)
                PPSLocalTracks_singleRPProtonIdx.append(nProton_singleRP)
                PPSLocalTracks_decRPId.append(dec_detid)
                PPSLocalTracks_rpType.append(rp_type)
                # TODO: multi-RP proton objects
                # for now, single-RP protons are assumed to be a clone of local tracks
                nProton_singleRP += 1
                Proton_singleRP_arm.append(detid.arm())
                Proton_singleRP_decRPId.append(dec_detid)
                Proton_singleRP_rpType.append(rp_type)
                Proton_singleRP_x.append(track.x())
                Proton_singleRP_y.append(track.y())
                Proton_singleRP_time.append(track.time())
                Proton_singleRP_timeUnc.append(track.timeUnc())
                Proton_singleRP_xi.append(xi)
                Proton_singleRP_thetaX.append(0.)  #FIXME
                Proton_singleRP_thetaY.append(0.)  #FIXME

        self.out.fillBranch("nPPSLocalTracks", nPPSLocalTracks)
        self.out.fillBranch("PPSLocalTracks_x", PPSLocalTracks_x)
        self.out.fillBranch("PPSLocalTracks_y", PPSLocalTracks_y)
        self.out.fillBranch("PPSLocalTracks_time", PPSLocalTracks_time)
        self.out.fillBranch("PPSLocalTracks_timeUnc", PPSLocalTracks_timeUnc)
        self.out.fillBranch("PPSLocalTracks_multiRPProtonIdx", PPSLocalTracks_multiRPProtonIdx)
        self.out.fillBranch("PPSLocalTracks_singleRPProtonIdx", PPSLocalTracks_singleRPProtonIdx)
        self.out.fillBranch("PPSLocalTracks_decRPId", PPSLocalTracks_decRPId)
        self.out.fillBranch("PPSLocalTracks_rpType", PPSLocalTracks_rpType)
        self.out.fillBranch("nProton_multiRP", nProton_multiRP)
        self.out.fillBranch("Proton_multiRP_arm", Proton_multiRP_arm)
        self.out.fillBranch("Proton_multiRP_xi", Proton_multiRP_xi)
        self.out.fillBranch("Proton_multiRP_thetaX", Proton_multiRP_thetaX)
        self.out.fillBranch("Proton_multiRP_thetaY", Proton_multiRP_thetaY)
        self.out.fillBranch("Proton_multiRP_t", Proton_multiRP_t)
        self.out.fillBranch("Proton_multiRP_time", Proton_multiRP_time)
        self.out.fillBranch("Proton_multiRP_timeUnc", Proton_multiRP_timeUnc)
        self.out.fillBranch("nProton_singleRP", nProton_singleRP)
        self.out.fillBranch("Proton_singleRP_arm", Proton_singleRP_arm)
        self.out.fillBranch("Proton_singleRP_decRPId", Proton_singleRP_decRPId)
        self.out.fillBranch("Proton_singleRP_rpType", Proton_singleRP_rpType)
        self.out.fillBranch("Proton_singleRP_x", Proton_singleRP_x)
        self.out.fillBranch("Proton_singleRP_y", Proton_singleRP_y)
        self.out.fillBranch("Proton_singleRP_time", Proton_singleRP_time)
        self.out.fillBranch("Proton_singleRP_timeUnc", Proton_singleRP_timeUnc)
        self.out.fillBranch("Proton_singleRP_xi", Proton_singleRP_xi)
        self.out.fillBranch("Proton_singleRP_thetaX", Proton_singleRP_thetaX)
        self.out.fillBranch("Proton_singleRP_thetaY", Proton_singleRP_thetaY)
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
directSimModuleConstr = lambda profile, verbosity=0: NanoAODDirectSimProducer(profile, verbosity)

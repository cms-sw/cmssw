from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
import math
import os
import ROOT


ROOT.PyConfig.IgnoreCommandLineOptions = True


class NanoAODDirectSimProducer(Module):
    beam_energy: float = 0.

    def __init__(self, profile):
        self.worker = ROOT.NanoAODDirectSimulator()
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
                print(track.x())
                nProton_singleRP += 1
                Proton_singleRP_arm.append(detid.arm())
                Proton_singleRP_decRPId.append(100 * detid.arm() + 10 * detid.station() + detid.rp())
                Proton_singleRP_rpType.append(0)  #FIXME
                Proton_singleRP_x.append(track.x())
                Proton_singleRP_y.append(track.y())
                Proton_singleRP_time.append(track.time())
                Proton_singleRP_timeUnc.append(track.timeUnc())
                Proton_singleRP_xi.append(xi)
                Proton_singleRP_thetaX.append(0.)  #FIXME
                Proton_singleRP_thetaY.append(0.)  #FIXME

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
directSimModuleConstr = lambda profile: NanoAODDirectSimProducer(profile)

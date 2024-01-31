import FWCore.ParameterSet.Config as cms

def customise(process):
    process.mtdTracksValid.inputTagV = 'offlinePrimaryVertices'
    process.mtdTracksValid.t0SafePID = 'tofPID3D:t0safe'
    process.mtdTracksValid.sigmat0SafePID = 'tofPID3D:sigmat0safe'
    process.mtdTracksValid.sigmat0PID = 'tofPID3D:sigmat0'
    process.mtdTracksValid.t0PID = 'tofPID3D:t0'
    
    process.vertices4DValid.offline4DPV = 'offlinePrimaryVertices'
    process.vertices4DValid.t0PID = 'tofPID3D:t0'
    process.vertices4DValid.t0SafePID = 'tofPID3D:t0safe'
    process.vertices4DValid.sigmat0SafePID = 'tofPID3D:sigmat0safe'
    process.vertices4DValid.probPi = 'tofPID3D:probPi'
    process.vertices4DValid.probK = 'tofPID3D:probK'
    process.vertices4DValid.probP = 'tofPID3D:probP'  
    return(process)

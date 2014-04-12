class BeamSpot:
    def __init__(self):
        self.Type = -1
        self.X = 0.
        self.Xerr = 0.
        self.Y = 0.
        self.Yerr = 0.
        self.Z = 0.
        self.Zerr = 0.
        self.sigmaZ = 0.
        self.sigmaZerr = 0.
        self.dxdz = 0.
        self.dxdzerr = 0.
        self.dydz = 0.
        self.dydzerr = 0.
        self.beamWidthX = 0.
        self.beamWidthXerr = 0.
        self.beamWidthY = 0.
        self.beamWidthYerr = 0.
        self.EmittanceX = 0.
        self.EmittanceY = 0.
        self.betastar = 0.
        self.IOVfirst = 0
        self.IOVlast = 0
        self.IOVBeginTime = 0
        self.IOVEndTime = 0
        self.Run = 0
    def Reset(self):
       self.Type = -1
       self.X = self.Y = self.Z = 0.
       self.Xerr = self.Yerr = self.Zerr = 0.
       self.sigmaZ = self.sigmaZerr = 0.
       self.dxdz = self.dydz = 0.
       self.dxdzerr = self.dydzerr = 0.
       self.beamWidthX = self.beamWidthY = 0.
       self.beamWidthXerr = self.beamWidthYerr = 0.
       self.EmittanceX = self.EmittanceY = self.betastar = 0.
       self.IOVfirst = self.IOVlast = 0
       self.Run = 0

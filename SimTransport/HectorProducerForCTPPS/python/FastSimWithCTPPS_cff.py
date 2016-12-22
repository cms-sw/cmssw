import FWCore.ParameterSet.Config as cms

def customise(process):
	if hasattr(process.VtxSmeared,"X0"):
		VertexX = process.VtxSmeared.X0
		VertexY = process.VtxSmeared.Y0
		VertexZ = process.VtxSmeared.Z0

	if hasattr(process.VtxSmeared,"MeanX"):
		VertexX = process.VtxSmeared.MeanX
		VertexY = process.VtxSmeared.MeanY
		VertexZ = process.VtxSmeared.MeanZ


	process.load('SimTransport.HectorProducerForCTPPS.HectorTransport_cfi')
	process.LHCTransport.HectorForCTPPS.VtxMeanX  = VertexX
	process.LHCTransport.HectorForCTPPS.VtxMeanY  = VertexY
	process.LHCTransport.HectorForCTPPS.VtxMeanZ = VertexZ

	# SimTransport on path
	process.transport_step = cms.Path(process.generator+process.LHCTransport)

	process.schedule.insert(2,process.transport_step)

	return(process)


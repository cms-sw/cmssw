import FWCore.ParameterSet.Config as cms

from RecoVertex.PixelVertexFinding.pixelVertexProducer_cfi import pixelVertexProducer as _pixelVertexProducer
pixelVertices = _pixelVertexProducer.clone()

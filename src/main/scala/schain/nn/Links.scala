package schain.nn

import org.nd4j.autodiff.samediff.{SDVariable, SameDiff}
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig
import org.nd4j.weightinit.impl.{XavierInitScheme, ZeroInitScheme}

class Linear2(w: SDVariable, b: SDVariable)(implicit sd: SameDiff) extends Chain(sd) {
  override def forward(x: SDVariable): SDVariable = {
    x.getSameDiff.linear(x, w, b)
  }
}

class Conv2D(w: SDVariable, b: SDVariable, k: Int)(implicit sd: SameDiff) extends Chain(sd) {
  private val config = Conv2DConfig.builder()
    .kH(k).kW(k)
    .pH(0).pW(0)
    .sH(1).sW(1)
    .dH(1).dW(1)
    .isSameMode(true)
    .isNHWC(true)
    .build()

  override def forward(x: SDVariable): SDVariable = {
    x.getSameDiff.conv2d(Array(x, w, b), config)
  }
}

object Links {
  def linear(in: Int, out: Int)(implicit sd: SameDiff): Chain = {
    val W = sd.`var`(in, out)
    W.setWeightInitScheme(XavierInitScheme.builder().order('c').fanIn(in).fanOut(out).build())
    val b = sd.`var`(out)
    b.setWeightInitScheme(ZeroInitScheme.builder().order('c').build())

    W.storeAndAllocateNewArray()
    b.storeAndAllocateNewArray()

    new Linear2(W, b)
  }

  def conv2d(inChannels: Int, outChannels: Int, kernelSize: Int)(implicit sd: SameDiff): Chain = {
    val W = sd.`var`(outChannels, inChannels, kernelSize, kernelSize)
    W.setWeightInitScheme(XavierInitScheme.builder().order('c').fanIn(inChannels).fanOut(outChannels).build())

    val b = sd.`var`(1, outChannels * inChannels)
    b.setWeightInitScheme(ZeroInitScheme.builder().order('c').build())

    W.storeAndAllocateNewArray()
    b.storeAndAllocateNewArray()

    new Conv2D(W, b, kernelSize)
  }
}

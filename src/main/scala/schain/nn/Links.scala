package schain.nn

import org.nd4j.autodiff.samediff.{SDVariable, SameDiff, VariableType}
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig
import org.nd4j.weightinit.impl.{XavierInitScheme, ZeroInitScheme}

class Linear2(w: SDVariable, b: SDVariable)(implicit sd: SameDiff) extends Chain(sd) {
  override def forward(x: SDVariable): SDVariable = {
    x.getSameDiff.nn.linear(x, w, b)
  }
}

class Conv2D(w: SDVariable, b: SDVariable, k: Int)(implicit sd: SameDiff) extends Chain(sd) {
  private val config = Conv2DConfig.builder()
    .kH(k).kW(k)
    .pH(0).pW(0)
    .sH(1).sW(1)
    .dH(1).dW(1)
    .isSameMode(false)
    .dataFormat("NHWC")
    .build()

  override def forward(x: SDVariable): SDVariable = {
    x.getSameDiff.cnn.conv2d(Array(x, w, b), config)
  }
}

object Links {
  def linear(in: Int, out: Int)(implicit sd: SameDiff): Chain = {
    val weightInitScheme = XavierInitScheme.builder().order('c').fanIn(in).fanOut(out).build()
    val W = sd.`var`(weightInitScheme, DataType.FLOAT, in, out)

    val biasWeightInitScheme = ZeroInitScheme.builder().order('c').build()
    val b = sd.`var`(biasWeightInitScheme, DataType.FLOAT, out)

    W.storeAndAllocateNewArray()
    b.storeAndAllocateNewArray()

    new Linear2(W, b)
  }

  def conv2d(inChannels: Int, outChannels: Int, kernelSize: Int)(implicit sd: SameDiff): Chain = {
    val weightInitScheme = XavierInitScheme.builder().order('c').fanIn(inChannels).fanOut(outChannels).build()
    val W = sd.`var`(weightInitScheme, DataType.FLOAT, kernelSize, kernelSize, inChannels, outChannels)

    val biasInitScheme = ZeroInitScheme.builder().order('c').build()
    val b = sd.`var`(biasInitScheme, DataType.FLOAT, 1, outChannels)

    W.storeAndAllocateNewArray()
    b.storeAndAllocateNewArray()

    new Conv2D(W, b, kernelSize)
  }
}

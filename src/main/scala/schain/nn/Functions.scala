package schain.nn

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig

object Functions {
  def maxPool2d(v: SDVariable, size: (Int, Int), stride: (Int, Int)): SDVariable = {
    val conf = Pooling2DConfig.builder()
      .kH(size._1).kW(size._2)
      .sH(stride._1).sW(stride._2)
      .isNHWC(true)
      .isSameMode(false)
      .build()
    v.getSameDiff.cnn.maxPooling2d(v, conf)
  }

  def relu(v: SDVariable): SDVariable = {
    v.getSameDiff.nn.relu(v, 0.0)
  }

  def softmax(v: SDVariable): SDVariable = {
    v.getSameDiff.softmax(v)
  }

  def logSoftmax(v: SDVariable): SDVariable = {
    v.getSameDiff.logSoftmax(v)
  }

  def view(v: SDVariable, x: Int, y: Int): SDVariable = {
    v.getSameDiff.reshape(v, x, y)
  }
}

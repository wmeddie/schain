package schain.nn

import org.nd4j.autodiff.samediff.{SDVariable, SameDiff}
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig

object Functions {
  def maxPool2d(v: SDVariable, size: (Int, Int))(implicit sd: SameDiff): SDVariable = {
    sd.maxPooling2d(v, Pooling2DConfig.builder().kH(size._2).kW(size._1).build())
  }

  def relu(v: SDVariable)(implicit sd: SameDiff): SDVariable = {
    sd.relu(v, 0.0)
  }

  def view(v: SDVariable, x: Int, y: Int)(implicit sd: SameDiff): SDVariable = {
    sd.reshape(v, x, y)
  }
}

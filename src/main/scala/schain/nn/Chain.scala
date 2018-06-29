package schain.nn

import org.nd4j.autodiff.samediff.{SDVariable, SameDiff}
import org.nd4j.linalg.api.ndarray.INDArray


abstract class Chain(sd: SameDiff) {
  def forward(x: SDVariable): SDVariable

  def apply(x: INDArray): SDVariable = {
    val tape: SameDiff = sd.dup()
    forward(tape.`var`(x))
  }

  def apply(x: SDVariable): SDVariable = {
    forward(x)
  }
}

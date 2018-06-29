package schain.examples

import org.nd4j.autodiff.samediff.{SDVariable, SameDiff}
import org.nd4j.linalg.factory.Nd4j
import schain.nn.{Chain, Functions => F, Links => L}

class MyNet(implicit sd: SameDiff) extends Chain(sd) {
  private val conv1 = L.conv2d(inChannels = 1, outChannels = 6, kernelSize = 5)
  private val conv2 = L.conv2d(inChannels = 6, outChannels = 16, kernelSize = 5)

  private val fc1 = L.linear(in = 16 * 5 * 5, out = 120)
  private val fc2 = L.linear(in = 120, out = 84)
  private val fc3 = L.linear(in = 84, out = 10)

  override def forward(in: SDVariable): SDVariable = {
    var x = in
    x = F.relu(conv1(in))
    x = F.maxPool2d(x, (2, 2))
    x = F.relu(conv2(x))
    x = F.maxPool2d(x, (2, 2))
    x = F.view(x, 1, 400)
    x = F.relu(fc1(x))
    x = F.relu(fc2(x))
    x = F.relu(fc3(x))

    x
  }
}


object ExampleNet {
  def main(args: Array[String]): Unit = {
    implicit val sd: SameDiff = SameDiff.create()
    val model = new MyNet()

    val out = model(Nd4j.rand(Array(1, 1, 32, 32), 42))

    println("Forwards:")
    println(out.getSameDiff.execAndEndResult())

    println("Backwards:")
    val grad = out.gradient()
    println(grad.eval())
  }
}

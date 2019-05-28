package schain.examples

import org.nd4j.autodiff.samediff.{SDVariable, SameDiff}

import schain.nn.{Chain, Functions => F, Links => L}

import org.nd4j.linalg.factory.Nd4j

class MyNet(implicit sd: SameDiff) extends Chain(sd) {
  private val conv1 = L.conv2d(inChannels = 1, outChannels = 6, kernelSize = 5)
  private val conv2 = L.conv2d(inChannels = 6, outChannels = 16, kernelSize = 5)

  private val fc1 = L.linear(in = 16 * 5 * 5, out = 120)
  private val fc2 = L.linear(in = 120, out = 84)
  private val fc3 = L.linear(in = 84, out = 10)

  override def forward(in: SDVariable): SDVariable = {
    var x = in
    x = F.relu(conv1(in))
    x = F.maxPool2d(x, size = (2, 2), stride = (2, 2))
    x = F.relu(conv2(x))
    x = F.maxPool2d(x, size = (2, 2), stride = (2, 2))
    x = F.view(x, 1, 400)
    x = F.relu(fc1(x))
    x = F.relu(fc2(x))
    x = F.softmax(fc3(x))

    x
  }
}


object ExampleNet {
  def main(args: Array[String]): Unit = {
    //Nd4j.getExecutioner.enableDebugMode(true)
    //Nd4j.getExecutioner.enableVerboseMode(true)

    implicit var graph: SameDiff = SameDiff.create()

    val model = new MyNet()

    val input = Nd4j.ones(1, 32, 32, 1)
    val out = model(input)

    val labels = Nd4j.zeros(1L, 10L)
    labels.putScalar(Array[Int](0, 0), 1.0F)

    println("Init Graph")
    println(graph.summary())

    println("Forward Graph")
    println(out.getSameDiff.summary())

    println("Forwards:")
    println(out.eval())


    val loss = out.getSameDiff.loss.softmaxCrossEntropy("loss", out.getSameDiff.`var`(labels), out)

    println("Loss:")
    println(loss.eval())

    println("Backwards:")
    loss.getSameDiff.execBackwards(null)
    val variable = loss.getSameDiff.grad("loss")
    println(variable.eval())
  }
}

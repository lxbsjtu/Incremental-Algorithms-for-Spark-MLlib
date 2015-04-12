package org.apache.spark.mllib.classification

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, argmax => brzArgmax, sum => brzSum}

import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.{Logging, SparkContext, SparkException}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.util.Utils
/**
 * Created by lxb on 15/4/7.
 */
class StreamingNaiveBayesModel (
  val lambda:Double,
  override val labels: Array[Double],
  override val pi: Array[Double],
   val theta1: Array[Array[Double]],
   val theta2: Array[Array[Double]])extends NaiveBayesModel(labels,pi,theta) {
  // 过来一批数据，就更新模型
  def update(data: DStream[LabeledPoint], decayFactor: Double, timeUnit: String): StreamingNaiveBayesModel = {

    val requireNonnegativeValues: Vector => Unit = (v: Vector) => {
      val values = v match {
        case SparseVector(size, indices, values) =>
          values
        case DenseVector(values) =>
          values
      }
      if (!values.forall(_ >= 0.0)) {
        throw new SparkException(s"Naive Bayes requires nonnegative feature values but found $v.")
      }
    }



    val aggregated = data.map(p => (p.label, p.features)).combineByKey[(Long, BDV[Double])](
      createCombiner = (v: Vector) => {
        requireNonnegativeValues(v)
        (1L, v.toBreeze.toDenseVector)
      },
      mergeValue = (c: (Long, BDV[Double]), v: Vector) => {
        requireNonnegativeValues(v)
        (c._1 + 1L, c._2 += v.toBreeze)
      },
      mergeCombiners = (c1: (Long, BDV[Double]), c2: (Long, BDV[Double])) =>
        (c1._1 + c2._1, c1._2 += c2._2)
    ).collect()
    var numDocuments =
      aggregated.foreach { case (_, (n, _)) =>
        numDocuments += n
      }
    val numFeatures = aggregated.head match {
      case (_, (_, v)) => v.size
    }

    val numLabels = aggregated.length
    val updatedPi = new Array[Double](numLabels)
    val updatedTheta = Array.fill(numLabels)(new Array[Double](numFeatures))


    val updatedNumOfLable = new Array[Double](numLabels)
    var i = 0
    aggregated.foreach { case (label, (n, sumTermFreqs)) =>
      var j = 0
      while (j < numFeatures) {
        val updatedThetaNum = updatedNumOfLable(i) * decayFactor * updatedTheta(i)(j) + sumTermFreqs(j)
        val updatedThetaDed = updatedNumOfLable(i) * decayFactor + n
        updatedTheta(i)(j) = updatedThetaNum / updatedThetaDed
        theta1(i)(j) = updatedThetaNum
        theta2(i)(j) = updatedThetaDed
        j += 1
      }
      val updatedPiNum = (numDocuments - n) * decayFactor * updatedPi(i) + n
      val updatedPiDed = (numDocuments - n) * decayFactor + n
      updatedPi(i) = updatedPiNum / updatedPiDed
      pi(i) = math.log(updatedPi(i))

      updatedNumOfLable(i) += n
      i += 1
    }
    new NaiveBayesModel(labels, pi, theta)

  }

  def predict(data: DStream[Vector]): DStream[Double] = {
    val numFeatures = theta1(0).length
    var i = 0
    while (i < labels.length) {
      theta(i).map(_ => math.log(theta1(i).map(_ => _ + lambda)) - math.log(theta2(i).map(_ => _ + lambda * numFeatures)))

      i ++
    }
    labels(brzArgmax(pi + theta * data.toBreeze))
  }
}

class StreamingNaiveBayes(
                       var decayFactor: Double,
                       var timeUnit: String) {

  def this() = this(1.0, StreamingNaiveBayes.BATCHES)

  protected var model: StreamingNaiveBayesModel = new StreamingNaiveBayesModel(null, null, null)



/*  def setDecayFactor(a: Double): this.type = {
    this.decayFactor = decayFactor
    this
  }

  def setHalfLife(halfLife: Double, timeUnit: String): this.type = {
    if (timeUnit != StreamingNaiveBayes.BATCHES && timeUnit != StreamingNaiveBayes.POINTS) {
      throw new IllegalArgumentException("Invalid time unit for decay: " + timeUnit)
    }
    this.decayFactor = math.exp(math.log(0.5) / halfLife)
    logInfo("Setting decay factor to: %g ".format (this.decayFactor))
    this.timeUnit = timeUnit
    this
  }
  def latestModel(): StreamingNaiveBayesModel = {
    model
  }
*/
  def train(data: DStream[LabeledPoint]) {
    data.foreachRDD { (rdd, time) =>
      model = model.update(rdd, decayFactor, timeUnit)
    }
  }

  def predictOn(data: DStream[Vector]): DStream[Double] = {
    data.map(model.predict)
  }

}

object StreamingNaiveBayes{
  final val BATCHES = "batches"
  final val POINTS = "points"

}
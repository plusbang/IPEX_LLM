/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.dllib.utils.tf.loaders


import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.tf.{PaddingType, TensorflowSpecHelper}
import org.tensorflow.framework.{DataType, NodeDef}
import com.intel.analytics.bigdl.dllib.utils.tf.Tensorflow._

class Conv3DBackpropFilterSpec extends TensorflowSpecHelper {

  "Conv3DBackpropFilter forward with VALID padding" should "be correct" in {

    val builder = NodeDef.newBuilder()
      .setName(s"Conv3DBackpropFilterTest")
      .setOp("Conv3DBackpropFilter")
      .putAttr("T", typeAttr(DataType.DT_FLOAT))
      .putAttr("strides", listIntAttr(Seq(1, 1, 2, 3, 1)))
      .putAttr("padding", PaddingType.PADDING_VALID.value)

    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()
    val outputBackprop = Tensor[Float](4, 19, 14, 13, 4).rand()

    // the output in this case is typical the scale of thousands,
    // so it is ok to have 1e-2 absolute error tolerance
    compare[Float](
      builder,
      Seq(input, filter, outputBackprop),
      0,
      1e-2
    )
  }

  "Conv3DBackpropFilter forward with SAME padding" should "be correct" in {

    val builder = NodeDef.newBuilder()
      .setName(s"Conv3DBackpropFilterTest")
      .setOp("Conv3DBackpropFilter")
      .putAttr("T", typeAttr(DataType.DT_FLOAT))
      .putAttr("strides", listIntAttr(Seq(1, 1, 2, 3, 1)))
      .putAttr("padding", PaddingType.PADDING_SAME.value)

    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()
    val outputBackprop = Tensor[Float](4, 20, 15, 14, 4).rand()

    // the output in this case is typical the scale of thousands,
    // so it is ok to have 1e-2 absolute error tolerance
    compare[Float](
      builder,
      Seq(input, filter, outputBackprop),
      0,
      1e-2
    )
  }
}

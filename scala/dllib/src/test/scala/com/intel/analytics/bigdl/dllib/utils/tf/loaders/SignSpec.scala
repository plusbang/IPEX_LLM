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
import com.intel.analytics.bigdl.dllib.utils.tf.Tensorflow.typeAttr
import com.intel.analytics.bigdl.dllib.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class SignSpec extends TensorflowSpecHelper {
  "Sign" should "be correct for float tensor" in {
    val t = Tensor[Float](32, 32).rand()
    t.setValue(1, 1, -1)
    t.setValue(1, 2, -3)
    t.setValue(1, 4, -2)
    t.setValue(2, 2, Float.PositiveInfinity)
    t.setValue(2, 3, Float.NegativeInfinity)
    t.setValue(4, 5, Float.NaN)
    compare[Float](
      NodeDef.newBuilder()
        .setName("sign_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .setOp("Sign"),
      Seq(t),
      0
    )
  }
}

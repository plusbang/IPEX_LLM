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
package com.intel.analytics.bigdl.dllib.nn.ops

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Log4Error

import scala.reflect.ClassTag

class Greater[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Compare[T] {
  override def compareFloat(a: Float, b: Float): Boolean = a > b

  override def compareDouble(a: Double, b: Double): Boolean = a > b

  override def compareChar(a: Char, b: Char): Boolean = a > b

  override def compareLong(a: Long, b: Long): Boolean = a > b

  override def compareShort(a: Short, b: Short): Boolean = a > b

  override def compareInt(a: Int, b: Int): Boolean = a > b

  override def compareBoolean(a: Boolean, b: Boolean): Boolean = {
    Log4Error.invalidOperationError(false, "Does not support Greater on Boolean")
    true
  }

  override def compareByteString(a: ByteString, b: ByteString): Boolean = {
    Log4Error.invalidOperationError(false, "Does not support Greater on ByteString")
    true
  }
}

object Greater {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new Greater())
}

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

package com.intel.analytics.bigdl.orca.python

import org.apache.spark.sql.OrcaArrowUtils

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object PythonOrcaSQLUtils {

  def ofFloat(): PythonOrcaSQLUtils[Float] = new PythonOrcaSQLUtils[Float]()

  def ofDouble(): PythonOrcaSQLUtils[Double] = new PythonOrcaSQLUtils[Double]()
}

class PythonOrcaSQLUtils[T: ClassTag](implicit ev: TensorNumeric[T]) extends OrcaArrowUtils {
}

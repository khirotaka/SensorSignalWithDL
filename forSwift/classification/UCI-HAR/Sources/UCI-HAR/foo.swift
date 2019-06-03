//
//  foo.swift
//  UCI-HAR
//
//  Created by 川島 寛隆 on 2019/06/02.
//

import Python
import TensorFlow

let os = Python.import("os")
let sys = Python.import("sys")


let pd = Python.import("pandas")
let home = os.environ["HOME"] + "/.SensorSignalDatasets/UCI HAR Dataset/"
var x_train = pd.read_csv(home + "train/X_train.txt", header: Python.None, delim_whitespace: true)
var y_train = pd.read_csv(home + "train/y_train.txt", header: Python.None, delim_whitespace: true)

var x_test = pd.read_csv(home + "test/X_test.txt", header: Python.None, delim_whitespace: true)
var y_test = pd.read_csv(home + "test/y_test.txt", header:Python.None, delim_whitespace: true)



//
//  model.swift
//  UCI-HAR
//
//  Created by 川島 寛隆 on 2019/06/03.
//
/*
import Foundation
import TensorFlow

struct CNN: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    var conv = Conv1D<Float>(filterShape: (2, 9, 32))    // [width, inputChannels, outputChannels]
    var pool = MaxPool1D<Float>(poolSize: 2, stride: 1, padding: Padding.valid)
    
    var flat = Flatten<Float>()
    var fc = Dense<Float>(inputSize: 512, outputSize: 9)
    
    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        let tmp = self.pool(relu(self.conv(input)))
        let output = softmax(self.fc(self.flat(tmp)))
        return output
    }
}

// var model = CNN()
// let optimizer = Adam(for: model)

*/

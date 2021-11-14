import { Network } from "./Network";
import { Matrix } from "./types";
import {expect} from "chai";
import mocha from "mocha";
import { arrayOf, createMatrix } from "./utilities";
import { mnist } from "src/mnist";
const { describe, it } = mocha;

function expectDimensions(x:number, y:number) {
  return (matrix: Matrix) => {
    const msg = `Expected matrix to have dimensions ${x}x${y} but it has dimensions ${matrix.length}x${matrix[0].length}`;
    expect(matrix, msg).to.have.lengthOf(y);
    matrix.forEach(row => {
      expect(row, msg).to.have.lengthOf(x);
    })
  }
}


function expectCells(tester: (n: number) => boolean) {
  return (matrix: Matrix) => {
    matrix.forEach(row => {
      row.forEach(cell => {
        expect(tester(cell)).to.be.true;
      })
    })
  }
}


class NiceMatrix {
  inner: Matrix;
  constructor(matrix: Matrix) {
    this.inner = matrix;
  }
  get(x: number, y: number) {
    return this.inner[y][x]
  }
}

describe("Network", () => {
  it("Initialises with correct number and shape of weights", () => {
    const layers = [3, 5, 8, 10, 100];
    const network = new Network(layers);
    expect(network.weights).to.have.lengthOf(4);
    expectDimensions(3, 5)(network.weights[0]);
    expectDimensions(5, 8)(network.weights[1]);
    expectDimensions(8, 10)(network.weights[2]);
    expectDimensions(10, 100)(network.weights[3]);
  })

  it("Initialises with correct number and shape of biases", () => {
    const layers = [3, 5, 8, 10, 100];
    const network = new Network(layers);
    expect(network.biases).to.have.lengthOf(4);
    expectDimensions(1, 5)(network.biases[0]);
    expectDimensions(1, 8)(network.biases[1]);
    expectDimensions(1, 10)(network.biases[2]);
    expectDimensions(1, 100)(network.biases[3]);
  })

  it("Biases should be 0", () => {
    const layers = [2, 3, 10];
    const network = new Network(layers);
    network.biases.forEach(bias => {
      expectCells(cell => cell === 0)(bias);
    })
  })

  it("Weights should be normal", () => {
    const layers = [50, 50, 50];
    const network = new Network(layers);
    let cellCount = 0;
    let cellSum = 0;
    network.weights.forEach(weight => {
      expectCells(cell => {
        cellCount++;
        cellSum += cell;
        return cell !== 0 && cell > -5 && cell < 5; //within expected range
      })(weight);
    });
    const average = cellSum / cellCount;
    const allowedError = 0.05;
    expect(average).greaterThan(average - allowedError);
    expect(average).lessThan(average + allowedError);
  })

  it("Feeds forward something sensible", () => {
    const layers = [1000, 500, 2];
    const network = new Network(layers);
    const result = network.feed_forward(createMatrix(1, 1000, Math.random));
    expect(result[0][0]).greaterThan(0.05).and.lessThan(0.95);
    expect(result[1][0]).greaterThan(0.05).and.lessThan(0.95);
  })

  function topNumberIsLarger(input: Matrix): Matrix {
    if (input[0][0] > input[1][0]) return [[1]];
    return [[0]];
  }

  it("Can learn", async (done) => {
    const layers = [2, 5, 5, 1];
    const network = new Network(layers, {learningRate: 1, batchSize: 5});
    const inputs = arrayOf(100000, () => createMatrix(1, 2, Math.random));
    const outputs = inputs.map(topNumberIsLarger); // want to know if top number is larger
    network.learn(inputs, outputs);

    const result1 = network.feed_forward([[0.6], [0.45]]);
    // console.log(result1);
    
    const result2 = network.feed_forward([[0.2], [0.8]]);
    // console.log(result2);

    expect(result1[0][0]).greaterThan(0.95).and.lessThanOrEqual(1);
    expect(result2[0][0]).greaterThan(0).and.lessThanOrEqual(0.05);
    done();
  }).timeout(10000)

  it("Can be serialized and loaded", () => {
    const layers = [2, 5, 5, 1];
    const network = new Network(layers, {learningRate: 1, batchSize: 5});
    const serial = network.serialize();

    const loaded = new Network([]);
    loaded.load(serial);
    expect(loaded.layers).to.deep.equal(network.layers);
    expect(loaded.weights).to.deep.equal(network.weights);
    expect(loaded.biases).to.deep.equal(network.biases);
    expect(loaded.options).to.deep.equal(network.options);
  })

  it("can update options", () => {
    const layers = [2, 5, 5, 1];
    const network = new Network(layers, {learningRate: 1, batchSize: 5});
    network.updateOptions({learningRate: 5, batchSize: 50, progress: true});
    expect(network.options.batchSize).to.equal(50);
    expect(network.options.learningRate).to.equal(5);
    expect(network.options.progress).to.be.true;
  })

})
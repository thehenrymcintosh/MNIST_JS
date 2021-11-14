import { Matrix, FiniteIterator, SerializedNetwork } from "./types";
import { arrayOf, createMatrix, standard_normal, zip, msqErr, ArrayIterator } from "./utilities";
// import cliProgress from "cli-progress";

function createWeights(layers: number[]): Matrix[] {
  const weightsLength = layers.length - 1;
  return arrayOf(weightsLength, (i) => {
    const fromSize = layers[i];
    const toSize = layers[i + 1];
    const initial = () => standard_normal() / fromSize ** 0.5;
    return createMatrix( fromSize, toSize, initial )
  })
}

function createBiases(layers: number[]): Matrix[] {
  const biasLength = layers.length - 1;
  return arrayOf(biasLength, (i) => createMatrix( 1, layers[i + 1], () => 0 ))
}

function matMul(a: Matrix, b: Matrix) : Matrix {
  if (a[0].length !== b.length) throw new Error("Width of A must match height of B!");
  const sharedLength = b.length;
  return createMatrix(b[0].length, a.length, (x,y) => {
    let sum = 0;
    for (let i = 0; i < sharedLength; i++) {
      sum += (a[y][i] * b[i][x]);
    }
    return sum;
  });
}

function matAdd(a: Matrix, b: Matrix) : Matrix {
  if (a.length !== b.length) throw new Error("Height of A must match height of B!");
  if (a[0].length !== b[0].length) throw new Error("Width of A must match width of B!");
  return createMatrix(a[0].length, a.length, (x,y) => a[y][x] + b[y][x]);
}

function scalarMul(a: Matrix, scalar: number) : Matrix {
  return createMatrix(a[0].length, a.length, (x,y) => a[y][x] * scalar );
}

function elMul(a: Matrix, b: Matrix) : Matrix {
  if (a.length !== b.length) throw new Error("Height of A must match height of B!");
  if (a[0].length !== b[0].length) throw new Error("Width of A must match width of B!");
  return createMatrix(a[0].length, a.length, (x,y) => a[y][x] * b[y][x]);
}

function matActivation(a: Matrix) : Matrix {
  return createMatrix(a[0].length, a.length, (x,y) => activation(a[y][x]));
}


function matTranspose(a: Matrix) : Matrix {
  return createMatrix(a.length, a[0].length, (x,y) => a[x][y]);
}

function matDerivActivation(a: Matrix) : Matrix {
  return createMatrix(a[0].length, a.length, (x,y) => {
    const sig = activation(a[y][x])
    return sig * (1 - sig);
  });
}

function activation(input: number): number {
  return 1 / ( 1 + Math.exp(-input) );
}

function dims(input: Matrix): [number, number] {
  return [input[0].length, input.length];
}

interface NetworkOptions {
  learningRate?: number,
  batchSize?: number,
  progress?: boolean,
}

interface NetworkOptionsRequired extends NetworkOptions {
  learningRate: number,
  batchSize: number,
}

export class Network {
  layers: number[];
  weights: Matrix[];
  biases: Matrix[];
  options: NetworkOptionsRequired;

  constructor(layers: number[], options: NetworkOptions = {}) {
    this.layers = layers;
    this.options = {
      learningRate: 1,
      batchSize: 1,
      progress: false,
      ...options
    };

    this.weights = createWeights(layers);
    this.biases = createBiases(layers);
  }

  serialize(): SerializedNetwork {
    const {layers, weights, biases, options} = this;
    return JSON.parse(JSON.stringify({
      layers,
      weights,
      biases, 
      options,
    }))
  }

  load(network: SerializedNetwork){
    const {layers, weights, biases, options} = network;
    this.layers = layers;
    this.weights = weights;
    this.biases = biases;
    this.options = options;
  }


  feed_forward(input: Matrix) : Matrix {
    return zip(this.weights, this.biases)
      .reduce((a, [w,b]) => {
        const z = matAdd( matMul(w, a), b );
        return matActivation( z )
      }, input );
  }

  learn(mixedInputs: FiniteIterator<Matrix> | Matrix[], mixedOutputs: FiniteIterator<Matrix> | Matrix[]) {
    let batchWeightUpdates: Matrix[][] = [];
    let batchBiasUpdates: Matrix[][] = [];
    let batchError: number[] = [];
    // let bar: cliProgress.Bar | undefined;
    const inputs = Array.isArray(mixedInputs) ? new ArrayIterator(mixedInputs) : mixedInputs;
    const outputs = Array.isArray(mixedOutputs) ? new ArrayIterator(mixedOutputs) : mixedOutputs;
    // if (this.options.progress){
    //   bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic); 
    //   bar.start(inputs.length, 0);
    // }

    let input = inputs.next();
    let output = outputs.next();
    let i = 0;
    while (!input.done && !output.done) {
      const {weightUpdates, biasUpdates, E} = this.learn_example(input.value, output.value);
      batchWeightUpdates.push(weightUpdates);
      batchBiasUpdates.push(biasUpdates);
      batchError.push(E);
      if (i % this.options.batchSize === 0) {
        this.apply_batch(batchWeightUpdates, batchBiasUpdates, batchError);
        batchBiasUpdates = [];
        batchWeightUpdates = [];
        batchError = [];
      }
      // bar?.update(i);
      i++;
      input = inputs.next();
      output = outputs.next();
    }
    // bar?.stop();
    if (batchWeightUpdates.length > 0) {
      this.apply_batch(batchWeightUpdates, batchBiasUpdates, batchError);
    }
  }

  get_E(input: Matrix, expectation: Matrix) : number {
    const output = this.feed_forward(input);
    const error = matAdd(expectation, scalarMul(output, -1));
    return msqErr(error);
  }

  updateOptions(options: NetworkOptions) {
    this.options = {
      ...this.options,
      ...options
    };
  }

  private apply_batch(weightsUpdates: Matrix[][], biasesUpdates: Matrix[][], batchError: number[]) {
    let newBiases = this.biases;
    for (let u = 0; u < biasesUpdates.length; u++) {
      const biasesUpdate = biasesUpdates[u];
      const E = batchError[u];
      newBiases = newBiases.map((bias, i) => matAdd(bias, scalarMul(biasesUpdate[i], E * this.options.learningRate) ))
    }

    let newWeights = this.weights;
    for (let u = 0; u < weightsUpdates.length; u++) {
      const weightsUpdate = weightsUpdates[u];
      const E = batchError[u];
      newWeights = newWeights.map((weight, i) => matAdd(weight, scalarMul(weightsUpdate[i], E * this.options.learningRate) ))
    }
    this.biases = newBiases;
    this.weights = newWeights;
  }

  private learn_example(input: Matrix, expectation: Matrix) {
    const activations : Matrix[] = [];
    const layerInputs : Matrix[] = [];

    const weightUpdates = new Array<Matrix>(this.weights.length);
    const biasUpdates = new Array<Matrix>(this.biases.length);
    const wb = zip(this.weights, this.biases)
    const output = wb.reduce((a, [w,b]) => {
        layerInputs.push(a);
        const z = matAdd( matMul(w, a), b );
        const output = matActivation( z )
        activations.push(z);
        return output;
      }, input );
    let error = matAdd(expectation, scalarMul(output, -1));
    const E = msqErr(error);
    for (let layer = activations.length - 1; layer >= 0; layer--) {
      const zPrime = matDerivActivation(activations[layer]);
      const errZPrime = elMul(zPrime, error);
      biasUpdates[layer] = errZPrime;
      weightUpdates[layer] = matMul( errZPrime, matTranspose(layerInputs[layer]) );
      error = matMul(matTranspose(this.weights[layer]), errZPrime);
    }
    return {
      weightUpdates,
      biasUpdates,
      E,
    }
  }
}
